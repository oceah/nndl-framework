#include "gpu/gmath.h"

#include "gpu/ocuda.h"

namespace ocean::cuda_kernel {

    __global__ void exp(float *x, int N) {
        auto index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (auto i = index; i < N; i += stride)
            x[i] = expf(x[i]);
    }

    __global__ void sum(const float *x, float *s, int N, int C) {
        // 列优先存储: x(i, j) = x[j * N + i]
        auto index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (auto i = index; i < N; i += stride) {
            float sum = 0;
            for (auto j = 0; j < C; j++)
                sum += x[j * N + i];
            s[i] = sum;
        }
    }

} // namespace ocean::cuda_kernel

namespace ocean {

    /* +-+-+-+-+-+-+-+-+-+- API -+-+-+-+-+-+-+-+-+-+ */

    __global__ static void _softmax(float *expx, const float *s, int N, int C) {
        auto index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;
        auto NC = N * C;

        for (auto i = index; i < NC; i += stride)
            expx[i] /= s[i % N];
    }

    void softmax_g(float *x, int N, int C) {
        // 减去 max_x 提高数值稳定性
        for (int k = 0; k < N; k++) {
            // 防止数值溢出
            float max_z = x[k];
            for (int i = 1; i < C; i++)
                max_z = std::max(max_z, x[k + i * N]);

            for (int i = 0; i < C; i++)
                x[k + i * N] -= max_z;
        }

        auto NC = N * C;

        float *dx, *ds;
        cudaMalloc(&dx, NC * sizeof(float));
        cudaMalloc(&ds, N * sizeof(float));
        cudaMemcpy(dx, x, NC * sizeof(float), cudaMemcpyHostToDevice);

        auto &prop = CUDA_GetCurrentDeviceProp();
        auto gridDim = prop.maxBlocksPerMultiProcessor;
        auto blockDim = prop.maxThreadsPerBlock;

        cuda_kernel::exp<<<gridDim, blockDim>>>(dx, NC);
        cuda_kernel::sum<<<gridDim, blockDim>>>(dx, ds, N, C);
        _softmax<<< gridDim, blockDim>>>(dx, ds, N, C);

        cudaMemcpy(x, dx, NC * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(dx);
        cudaFree(ds);
    }

    void conv_g(const float *x, const float *h, float *y, int Nx, int Nh,
                int N, int Rh,
                int pad, int stride, int dilation) {
        cudnnHandle_t cudnn;
        cudnnCreate(&cudnn);

        // 创建输入张量描述符
        cudnnTensorDescriptor_t xDesc;
        cudnnCreateTensorDescriptor(&xDesc);
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, 1, Nx, 1);

        // 创建输出张量描述符
        cudnnTensorDescriptor_t yDesc;
        cudnnCreateTensorDescriptor(&yDesc);
        int Ny = (Nx + 2 * pad - dilation * (Nh - 1) - 1) / stride + 1;
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, 1, Ny, 1);

        // 创建卷积核描述符
        cudnnFilterDescriptor_t hDesc;
        cudnnCreateFilterDescriptor(&hDesc);
        cudnnSetFilter4dDescriptor(hDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, Rh, 1, Nh, 1);

        // 创建卷积描述符
        cudnnConvolutionDescriptor_t convDesc;
        cudnnCreateConvolutionDescriptor(&convDesc);
        cudnnSetConvolution2dDescriptor(convDesc, pad, 0, stride, 1, dilation, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

        // 选择卷积算法
        const int maxAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        int returnedAlgoCount;
        cudnnConvolutionFwdAlgoPerf_t algoPerf[maxAlgoCount];

        cudnnFindConvolutionForwardAlgorithm(cudnn, xDesc, hDesc, convDesc, yDesc,
                                             maxAlgoCount, &returnedAlgoCount, algoPerf);

        // 选择性能最优的算法
        cudnnConvolutionFwdAlgo_t algo = algoPerf[0].algo;

        // 获取所需工作空间大小
        size_t workspace_bytes;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn, xDesc, hDesc, convDesc, yDesc, algo, &workspace_bytes);
        float *workspace;
        cudaMalloc(&workspace, workspace_bytes);

        float *d_x, *d_h, *d_y;
        cudaMalloc(&d_x, N * std::max(Nx, Ny) * sizeof(float));
        cudaMalloc(&d_h, Rh * Nh * sizeof(float));
        cudaMalloc(&d_y, N * std::max(Ny, Nx) * sizeof(float));
        cudaMemcpy(d_x, x, N * Nx * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_h, h, Rh * Nh * sizeof(float), cudaMemcpyHostToDevice);

        constexpr float alpha = 1.0f, beta = 0.0f;

        cublasHandle_t handle;
        cublasCreate(&handle);
        if (N > 1) {
            // C = alpha * A' + beta * A
            cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nx, N, &alpha, d_x, N, &beta, d_x, Nx, d_y, Nx);

            cudaMemcpy(d_x, d_y, N * Nx * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        if (Rh > 1) {
            // C = alpha * A' + beta * A
            cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nh, Rh, &alpha, d_h, Rh, &beta, d_h, Nh, d_y, Nh);

            cudaMemcpy(d_h, d_y, Rh * Nh * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        // 执行卷积运算
        cudnnConvolutionForward(cudnn, &alpha, xDesc, d_x, hDesc, d_h, convDesc, algo, workspace, workspace_bytes,
                                &beta, yDesc, d_y);

        if (N > 1)  // C = alpha * A' + beta * A
            cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, Ny, &alpha, d_y, Ny, &beta, d_y, N, d_x, N);

        // 销毁 cuBLAS 句柄
        cublasDestroy(handle);

        cudaMemcpy(y, d_x, N * Ny * sizeof(float), cudaMemcpyDeviceToHost);

        // 释放资源
        cudaFree(workspace);
        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroyFilterDescriptor(hDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroy(cudnn);

        cudaFree(d_x);
        cudaFree(d_h);
        cudaFree(d_y);
    }

    void transpose_g(const float *x, float *y, int R, int C) {
        cublasHandle_t handle;
        cublasCreate(&handle);

        // 在设备上分配内存
        float *d_x, *d_y;
        cudaMalloc(&d_x, R * C * sizeof(float));
        cudaMalloc(&d_y, C * R * sizeof(float));

        // 将输入数据拷贝到设备
        cudaMemcpy(d_x, x, R * C * sizeof(float), cudaMemcpyHostToDevice);

        // C = alhpa * A + beta * B
        constexpr float alpha = 1.0f;   // 输入矩阵的权重
        constexpr float beta = 0.0f;    // 输出矩阵的初始值

        // 执行转置操作
        // C = alpha * A' + beta * A
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, C, R, &alpha, d_x, R, &beta, d_x, C, d_y, C);

        // 将结果从设备拷贝回主机
        cudaMemcpy(y, d_y, C * R * sizeof(float), cudaMemcpyDeviceToHost);

        // 释放设备内存
        cudaFree(d_x);
        cudaFree(d_y);

        // 销毁 cuBLAS 句柄
        cublasDestroy(handle);
    }

} // namespace ocean
