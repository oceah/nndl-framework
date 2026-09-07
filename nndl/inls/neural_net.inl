#include "../neural_net.h"

#include "math/dsp.hpp"

#if ONN_USE_CUDA

#include "gpu/ocuda.h"
#include "gpu/gmath.h"

#endif

using std::cout;
using std::endl;

namespace ocean {
#if ONN_USE_CUDA

    /// @brief 神经网络 CUDA 核函数
    namespace onn_gpu_kernel {
        /* +-+-+-+-+-+-+-+-+-+- 通用核 -+-+-+-+-+-+-+-+-+-+ */

        /// @brief PReLU
        /// @param x 数据[N, D]
        /// @param gamma 参数[D]
        /// @param N 数据量
        /// @param D 数据维度
        __global__ void prelu(float *x, const float *gamma, int N, int D) {
            auto index = blockIdx.x * blockDim.x + threadIdx.x;
            auto stride = blockDim.x * gridDim.x;
            auto ND = N * D;

            for (auto i = index; i < ND; i += stride)
                x[i] = x[i] >= 0 ? x[i] : x[i] * gamma[i % N];
        }

        /// @brief 张量逐行加向量
        /// @param x 张量[N, C]
        /// @param b 向量[C]
        /// @param N 数据量
        /// @param C 数据维度
        __global__ void tensor_add_vector(float *x, float *b, int N, int C) {
            for (int i = blockIdx.x; i < C; i += gridDim.x)
                for (int j = threadIdx.x; j < N; j += blockDim.x)
                    x[i * N + j] += b[i];
        }

        /* +-+-+-+-+-+-+-+-+-+- 专用核 -+-+-+-+-+-+-+-+-+-+ */

        /// @brief softmax-call_inerr CUDA 核函数
        /// @param dx 输入误差
        /// @param dl 输出误差
        /// @param y 输出
        /// @param N 数据量
        /// @param D 数据维度
        __global__ static void
        __cudakernel_softmax_call_inerr(float *dx, const float *dl, const float *y, int N, int D) {
            auto index = blockIdx.x * blockDim.x + threadIdx.x;
            auto stride = blockDim.x * gridDim.x;
            auto ND = N * D;

            for (auto idx = index; idx < ND; idx += stride) {
                auto k = idx % N;
                dx[idx] = dl[idx];
                for (int j = 0; j < D; j++)
                    dx[idx] -= y[j * N + k] * dl[j * N + k];
                dx[idx] *= y[idx];
            }
        }

        /// @brief prelu-cal_inerr CUDA 核函数
        /// @param dx 输入误差
        ///     @dx must be initialized with dl before calling this kernel
        /// @param x 输入
        /// @param gamma 参数
        /// @param N 数据量
        /// @param D 数据维度
        __global__ static void
        __cudakernel_prelu_cal_inerr(float *dx, const float *x, const float *gamma, int N, int D) {
            auto index = blockIdx.x * blockDim.x + threadIdx.x;
            auto stride = blockDim.x * gridDim.x;
            auto ND = N * D;

            for (auto idx = index; idx < ND; idx += stride)
                dx[idx] = x[idx] >= 0 ? dx[idx] : dx[idx] * gamma[idx % N];
        }

        /// @brief prelu-cal_grad CUDA 核函数
        /// @param dg 参数梯度[D]
        /// @param x 输入[N, D]
        /// @param dy 输出误差[N, D]
        __global__ static void
        __cudakernel_prelu_cal_grad(float *dg, const float *x, const float *dy, int N, int D) {
            auto index = blockIdx.x * blockDim.x + threadIdx.x;
            auto stride = blockDim.x * gridDim.x;

            for (auto i = index; i < D; i += stride) {
                float sum = 0;
                unsigned iN = i * N;
                for (auto j = 0; j < N; j++)
                    sum += x[iN + j] >= 0.0 ? 0.0 : dy[iN + j] * x[iN + j];
                dg[i] = sum / (float) N;
            }
        }
    } // namespace onn_gpu_kernel

#endif

    /* +-+-+-+-+-+-+-+-+-+- 前馈神经网络 -+-+-+-+-+-+-+-+-+-+ */

    template<class Float>
    void FNN<Float>::train(const tensor_type &x, const tensor_type &y, Float alpha) {
        if (x.dim() > 1 && x.size(0) != y.size(0))
            throw std::runtime_error("FNN::train: size mismatch.");
        // assert x.size == y.size
        tensor_type dl;
        switch (x.dim()) {
            case 1: // 单例训练
            case 2: // 批量训练
                input(x, true);
                dl = dL(y, output());
                backprop(dl);
                update(alpha);
                break;
            default:
                throw std::runtime_error("FNN::train: invalid input dimension.");
        }
    }

    /* +-+-+-+-+-+-+-+-+-+- 参数管理器 -+-+-+-+-+-+-+-+-+-+ */

    template<class Float>
    template<class... Args> requires (std::is_integral_v<std::remove_cvref_t<Args> > && ...)

    NNParam<Float>::NNParam(Args &&... args) : _param(std::forward<Args>(args)...) {
        auto &size = _param.size();
        gparam.resize(size);
        clear_grad();

        Gt.resize(size);
        Gt.fill(0.0);
        Dt.resize(size);
        Dt.fill(0.0);

        Mt.resize(size);
        Mt.fill(0.0);
    }

    template<class Float>
    void NNParam<Float>::set(Float _beta, Float _rho, Float _clip, Float _lambda, Float _eps) {
        beta = _beta;
        rho = _rho;
        clip = _clip;
        lambda = _lambda;
        eps = _eps;
    }

    template<class Float>
    void NNParam<Float>::set(const string &opt, Float val) {
        if (opt == "beta")
            beta = val;
        else if (opt == "rho")
            rho = val;
        else if (opt == "clip")
            clip = val;
        else if (opt == "lambda")
            lambda = val;
        else if (opt == "eps")
            eps = val;
        else
            throw std::runtime_error("NNParam::set: invalid option.");
    }

    template<class Float>
    [[nodiscard]] typename NNParam<Float>::tensor_type &NNParam<Float>::inertia() {
        if (paramh.empty())
            paramh = _param;
        return paramh;
    }

    template<class Float>
    inline void NNParam<Float>::clear_grad() {
        gparam.fill(0.0);
        K = 0.0;
    }

    template<class Float>
    inline void NNParam<Float>::push_grad(const tensor_type &g) {
        // lazy initialization
        if (gparam.empty()) {
            auto &size = _param.empty() ? paramh.size() : _param.size();
            Gt.resize(size);
            Gt.fill(0.0);
            Dt.resize(size);
            Dt.fill(0.0);
            Mt.resize(size);
            Mt.fill(0.0);
            gparam.resize(size);
            gparam.fill(0.0);
            if (_param.empty()) {
                _param.resize(size);
                _param.fill(0.0);
            }
            if (paramh.empty()) {
                paramh.resize(size);
                paramh.fill(0.0);
            }
        }

        gparam = (K * gparam + g) / (K + 1.0);
        K += 1.0;
    }

    template<class Float>
    void NNParam<Float>::update(Float alpha) {
        // 梯度截断
        Float gm = gparam.norm();
        if (gm > clip)
            gparam *= clip / gm;
        /// TODO: 根据一段时间内的平均梯度来自动调整截断阈值

        // RMSProp
        Gt = beta * Gt + (1.0 - beta) * gparam.scalar_mul(gparam);
        Dt = gparam.scalar_div((Gt + eps).scalar_op([](Float x) { return sqrt(x); }));

        // NAG
        Mt = rho * Mt - alpha * Dt;

        _param += Mt;
        clear_grad();

        // 更新惯性参数
        paramh = _param + rho * Mt;
    }

    template<class T>
    std::ostream &operator<<(std::ostream &os, const NNParam<T> &pm) {
        os << pm.param();
        return os;
    }

    /* +-+-+-+-+-+-+-+-+-+- SISO神经元 -+-+-+-+-+-+-+-+-+-+ */

    template<class Float>
    void SISONeuron<Float>::input(const tensor_type &x, Neuron<Float> *p, bool inertial) {
        auto pgetter = [&](size_type i) -> tensor_type & {
            return param[i].param();
        };

        auto igetter = [&](size_type i) -> tensor_type & {
            return param[i].inertia();
        };

        pin = &x;
        if (inertial)
            cal_out(x, igetter, out);
        else
            cal_out(x, pgetter, out);
        // 下一层输入
        if (next != nullptr)
            next->input(out, this, inertial);
    }

    template<class Float>
    void SISONeuron<Float>::backprop(const tensor_type &dl, Neuron<Float> *p) {
        auto pg = [&](size_type i) -> tensor_type & {
            return param[i].inertia();
        };

        pdl = &dl;
        // 计算上一层的误差
        if (prev != nullptr)
            cal_inerr(*pin, out, dl, pg, inerr);
        // 参数更新
        auto g = cal_grad(*pin, out, dl, pg);
        size_type i = 0;
        for (auto &pi: param)
            pi.push_grad(g[i++]);
        // 反向传播误差
        if (prev != nullptr)
            prev->backprop(inerr, this);
    }

    template<class Float>
    void SISONeuron<Float>::update(const Float &alpha) {
        for (auto &pi: param)
            pi.update(alpha);

        if (this->prev != nullptr)
            this->prev->update(alpha);
    }

    template<class Float>
    inline void SISONeuron<Float>::connect(Neuron<Float> *p) {
        next = p;
        this->aux_accept_connect(p);
    }

    /* +-+-+-+-+- 神经分路器 -+-+-+-+-+ */

    template<class Float>
    void NNBrancher<Float>::input(const tensor_type &x, Neuron<Float> *p, bool inertial) {
        pin = &x;
        // 下一层输入
        for (auto &[next, _]: pdls)
            ((Neuron<Float> *) next)->input(x, this, inertial);
    }

    template<class Float>
    void NNBrancher<Float>::backprop(const tensor_type &dl, Neuron<Float> *p) {
        auto &pdl = pdls[p];
        pdl = &dl;

        if (prev == nullptr)
            return;
        // assert prev is not null

        ++ready;
        if (ready == pdls.size()) {
            ready = 0;
            // 计算上一层的误差
            inerr.resize(pin->size());
            inerr.fill(0.0);
            for (auto &[_, _dl]: pdls)
                inerr += *_dl;
            inerr /= (Float) pdls.size();
            // 反向传播误差
            prev->backprop(inerr, this);
        }
    }

    template<class Float>
    void NNBrancher<Float>::update(const Float &alpha) {
        if (prev == nullptr)
            return;
        // assert prev is not null

        ++ready;
        if (ready == pdls.size()) {
            ready = 0;
            prev->update(alpha);
        }
    }

    template<class Float>
    inline void NNBrancher<Float>::connect(Neuron<Float> *p) {
        pdls[p] = nullptr;
        this->aux_accept_connect(p);
    }

    /* +-+-+-+-+- 神经并路器 -+-+-+-+-+ */

    template<class Float>
    void NNMerger<Float>::input(const tensor_type &x, Neuron<Float> *p, bool inertial) {
        auto &[_pin, _] = prevs[p];
        _pin = &x;
        ++ready;
        if (ready == prevs.size()) {
            ready = 0;

            // merge
            size_type N = x.dim() > 1 ? x.size(0) : 1;
            size_type D = 0;
            if (N > 1)
                for (auto &[_, info]: prevs) {
                    auto &[pin, __] = info;
                    D += pin->size(1);
                }
            else
                for (auto &[_, info]: prevs) {
                    auto &[pin, __] = info;
                    D += pin->size(0);
                }
            if (N > 1) {
                out.resize(N, D);
                size_type i = 0;
                for (size_type k = 0; k < N; k++)
                    for (auto &[_, info]: prevs) {
                        auto &[pin, __] = info;
                        auto &xk = *pin;
                        size_type Dk = xk.size(1);
                        for (size_type j = 0; j < Dk; j++)
                            out[i++] = xk(k, j);
                    }
            } else {
                out.resize(D);
                size_type i = 0;
                for (auto &[_, info]: prevs) {
                    auto &[pin, __] = info;
                    for (const auto &xi: *pin)
                        out[i++] = xi;
                }
            }

            // 下一层输入
            if (next != nullptr)
                next->input(out, this, inertial);
        }
    }

    template<class Float>
    void NNMerger<Float>::backprop(const tensor_type &dl, Neuron<Float> *p) {
        pdl = &dl;

        // 计算上一层的误差
        size_type Doffset = 0;
        size_type N = dl.dim() > 1 ? dl.size(0) : 1;
        for (auto &[prev, info]: prevs) {
            auto &[pin, inerr] = info;
            size_type D = N > 1 ? pin->size(1) : pin->size(0);
            if (N > 1) {
                inerr.resize(N, D);
                for (size_type k = 0; k < N; k++)
                    for (size_type j = 0; j < D; j++)
                        inerr(k, j) = dl(k, Doffset + j);
            } else {
                inerr.resize(D);
                for (size_type j = 0; j < D; j++)
                    inerr[j] = dl[Doffset + j];
            }
            Doffset += D;
            ((Neuron<Float> *) prev)->backprop(inerr, this);
        }
    }

    template<class Float>
    void NNMerger<Float>::update(const Float &alpha) {
        for (auto &[prev, _]: prevs)
            ((Neuron<Float> *) prev)->update(alpha);
    }

    template<class Float>
    inline void NNMerger<Float>::connect(Neuron<Float> *p) {
        next = p;
        this->aux_accept_connect(p);
    }

    template<class Float>
    inline void NNMerger<Float>::accept_connect(Neuron<Float> *p) {
        prevs[p] = {nullptr, {}};
    }

    /* +-+-+-+-+-+-+-+-+-+- 数据管理器 -+-+-+-+-+-+-+-+-+-+ */

    template<class Float>
    tensor<Float> to_one_hat(const tensor<Float> &Y, size_t C) {
        using size_type = tensor<Float>::size_type;

        if (Y.dim() > 1) {
            size_type N = Y.size(0);
            tensor<Float> Yh(N, C);
            Yh.fill(0);
            for (size_type i = 0; i < N; i++)
                Yh(i, (size_type) Y(i)) = 1;
            return Yh;
        } else {
            tensor<Float> Yh(C);
            Yh.fill(0);
            Yh((size_type) Y[0]) = 1;
            return Yh;
        }
    }

    /// @brief one-hot 行向量 => 标签
    /// @param Y one-hot 行向量 Nx1
    /// @note 类别标签从 0 开始
    template<class Float>
    tensor<Float> argmax(const tensor<Float> &Y) {
        using size_type = tensor<Float>::size_type;

        if (Y.dim() > 1) {
            size_type N = Y.size(0);
            tensor<Float> Yh(N);
            for (size_type i = 0; i < N; i++) {
                Float max = Y(i, 0);
                size_type idx = 0;
                for (size_type j = 1; j < Y.size(1); j++)
                    if (Y(i, j) > max) {
                        max = Y(i, j);
                        idx = j;
                    }
                Yh[i] = idx;
            }
            return Yh;
        } else {
            Float max = Y[0];
            size_type idx = 0;
            for (size_type i = 1; i < Y.size(0); i++)
                if (Y[i] > max) {
                    max = Y[i];
                    idx = i;
                }
            return tensor<Float>(idx);
        }
    }

    /* +-+-+-+-+- 数据归一化器 -+-+-+-+-+ */

    template<class Float>
    void DataNormalizer<Float>::cal_out(const tensor_type &x, const param_getter &pg, tensor_type &y) {
        y.resize(x.size());
        std::copy(x.begin(), x.end(), y.begin());
        normalize(y);
    }

    template<class Float>
    void DataNormalizer<Float>::cal_inerr(const tensor_type &x, const tensor_type &y, const tensor_type &dl,
                                          const param_getter &pg, tensor_type &dx) {
        dx.resize(dl.size());
        std::copy(dl.begin(), dl.end(), dx.begin());
        denormalize(dx);
    }

    /* +-+-+-+-+- 最小最大值归一化 -+-+-+-+-+ */

    /// @brief 设置数据集
    /// @param X: 数据集
    /// @reflect 最小值 => minval
    /// @reflect 最大值 => maxval
    template<class Float>
    void MinMaxNormalizer<Float>::set(const tensor<Float> &X, Float minval, Float maxval) {
        min = X.min();
        range = X.max();
        min = (min + range) / (Float) 2.0; // min <= mid
        range = (range - min) * ((Float) 2 / (maxval - minval));
        min -= range / (Float) 2.0;
    }

    /// @brief 归一化
    /// @param x: 数据
    template<class Float>
    void MinMaxNormalizer<Float>::normalize(tensor_type &x) const {
        // x = (x - min).scalar_div(range);
        size_type N = x.size(0);
        for (size_type i = 0; i < N; i++)
            for (size_type j = 0; j < x.size(1); j++)
                x(i, j) = (x(i, j) - min[j]) / range[j];
    }

    /// @brief 反归一化
    /// @param x: 数据
    template<class Float>
    void MinMaxNormalizer<Float>::denormalize(tensor_type &x) const {
        // x = x.scalar_mul(range) + min;
        size_type N = x.size(0);
        for (size_type i = 0; i < N; i++)
            for (size_type j = 0; j < x.size(1); j++)
                x(i, j) = x(i, j) * range[j] + min[j];
    }

    /* +-+-+-+-+-+-+-+-+-+- 激活函数 -+-+-+-+-+-+-+-+-+-+ */

    /* +-+-+-+-+- Softmax函数 -+-+-+-+-+ */

    template<class Float>
    void Softmax<Float>::cal_out(const tensor_type &x, const param_getter &pg, tensor_type &y) {
        size_type N = x.dim() > 1 ? x.size(0) : 1;
        size_type C = x.dim() > 1 ? x.size(1) : x.size(0);

        bool use_cuda = false;
#if ONN_USE_CUDA
        constexpr bool is_float = std::is_same_v<Float, float>;

        if constexpr (is_float) {
            if (CUDA_IsAvailable() && N * C > 25000) {
                use_cuda = true;
                if (y.size() != x.size())
                    y = x;
                else std::copy(x.begin(), x.end(), y.begin());
                softmax_g(y.data(), N, C);
            }
        }
#endif

        if (!use_cuda) {
            if (N > 1) {
                size_type D = x.size(1);
                y.resize(N, D);
                for (size_type k = 0; k < N; k++) {
                    // 防止数值溢出
                    Float max_z = x(k, 0);
                    for (size_type i = 1; i < D; i++)
                        max_z = std::max(max_z, x(k, i));

                    Float sum = 0.0;
                    for (size_type i = 0; i < D; i++) {
                        y(k, i) = exp(x(k, i) - max_z); // 减去 max_z 提高数值稳定性
                        sum += y(k, i);
                    }
                    // 归一化, 使所有输出之和为 1
                    for (size_type i = 0; i < D; i++)
                        y(k, i) /= sum;
                }
            } else {
                Float max_z = *std::max_element(x.cbegin(), x.cend()); // 防止数值溢出
                Float sum = 0.0;
                size_type D = x.size(0);
                y.resize(D);
                for (size_type i = 0; i < D; i++) {
                    y[i] = exp(x[i] - max_z); // 减去 max_z 提高数值稳定性
                    sum += y[i];
                }
                y /= sum; // 归一化, 使所有输出之和为 1
            }
        }

#if ONN_DBG
        cout << "[Softmax - cal_out]" << endl;
        cout << "x = [" << x << "];" << endl;
        cout << "y = [" << y << "];" << endl;
#endif
    }

    template<class Float>
    void Softmax<Float>::cal_inerr(const tensor_type &x, const tensor_type &y,
                                   const tensor_type &dl, const param_getter &pg,
                                   tensor_type &dx) {
        size_type N = x.dim() > 1 ? x.size(0) : 1;
        size_type D = x.size(N > 1 ? 1 : 0);

        bool use_cuda = false;
#if ONN_USE_CUDA
        constexpr bool is_float = std::is_same_v<Float, float>;

        if constexpr (is_float) {
            if (CUDA_IsAvailable() && N * D > 800) {
                use_cuda = true;
                if (dx.size() != dl.size())
                    dx = dl;
                else std::copy(dl.begin(), dl.end(), dx.begin());

                float *ddx, *ddl, *dy;
                cudaMalloc(&ddx, N * D * sizeof(float));
                cudaMalloc(&ddl, N * D * sizeof(float));
                cudaMalloc(&dy, N * D * sizeof(float));
                cudaMemcpy(ddl, dl.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(dy, y.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);

                auto &prop = CUDA_GetCurrentDeviceProp();
                int gridDim = std::min(prop.maxBlocksPerMultiProcessor, (int) N);
                int blockDim = std::min(prop.maxThreadsPerBlock, (int) D);

                onn_gpu_kernel::__cudakernel_softmax_call_inerr<<<gridDim, blockDim>>>(ddx, ddl, dy, N, D);

                cudaMemcpy(dx.data(), ddx, N * D * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(ddx);
                cudaFree(ddl);
                cudaFree(dy);
            }
        }
#endif

        if (!use_cuda) {
            if (N > 1) {
                dx.resize(N, D);
                for (size_type k = 0; k < N; k++)
                    for (size_type i = 0; i < D; i++) {
                        dx(k, i) = dl(k, i);
                        for (size_type j = 0; j < D; j++)
                            dx(k, i) -= y(k, j) * dl(k, j);
                        dx(k, i) *= y(k, i);
                    }
            } else {
                dx.resize(D);
                for (size_type i = 0; i < D; i++) {
                    dx[i] = dl[i];
                    for (size_type j = 0; j < D; j++)
                        dx[i] -= y[j] * dl[j];
                    dx[i] *= y[i];
                }
            }
        }

#if ONN_DBG
        cout << "[Softmax - cal_inerr]" << endl;
        cout << "dy = [" << dl << "];" << endl;
        cout << "y = [" << y << "];" << endl;
        cout << "dx = [" << dx << "];" << endl;
#endif
    }

    template<class T>
    std::ostream &operator<<(std::ostream &os, const Softmax<T> &n) {
        os << "[Softmax]" << std::endl;
        return os;
    }

    /* +-+-+-+-+- PReLU函数 -+-+-+-+-+ */

    template<class Float>
    PReLU<Float>::PReLU() {
        auto &param = this->param;
        param.reserve(1);
        param.emplace_back();
    }

    template<class Float>
    void PReLU<Float>::cal_out(const tensor_type &x, const param_getter &pg, tensor_type &y) {
        auto &gamma = pg(0);

        size_type N = x.dim() > 1 ? x.size(0) : 1;
        size_type D = x.dim() > 1 ? x.size(1) : x.size(0);

        // lazy initialization
        if (gamma.empty()) {
            gamma.resize(D);
            gamma.fill(0.001);
        }

        bool use_cuda = false;
#if ONN_USE_CUDA
        constexpr bool is_float = std::is_same_v<Float, float>;
        if constexpr (is_float) {
            if (CUDA_IsAvailable() && N * D > 49000) {
                use_cuda = true;
                if (y.size() != x.size())
                    y.resize(x.size());

                float *dx, *dg;
                cudaMalloc(&dx, N * D * sizeof(float));
                cudaMalloc(&dg, D * sizeof(float));
                cudaMemcpy(dx, x.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(dg, gamma.data(), D * sizeof(float), cudaMemcpyHostToDevice);

                auto &prop = CUDA_GetCurrentDeviceProp();
                int gridDim = std::min(prop.maxBlocksPerMultiProcessor, (int) D);
                int blockDim = std::min(prop.maxThreadsPerBlock, (int) N);

                onn_gpu_kernel::prelu<<<gridDim, blockDim>>>(dx, dg, N, D);

                cudaMemcpy(y.data(), dx, N * D * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(dx);
                cudaFree(dg);
            }
        }
#endif

        if (!use_cuda) {
            if (N > 1) {
                y = x;
                for (size_type i = 0; i < N; i++)
                    for (size_type j = 0; j < D; j++)
                        if (y(i, j) < 0.0)
                            y(i, j) *= gamma[j];
            } else {
                y = x;
                for (size_type i = 0; i < D; i++)
                    if (y[i] < 0.0)
                        y[i] *= gamma[i];
            }
        }

#if ONN_DBG
        cout << "[PReLU - cal_out]" << endl;
        cout << "gamma = [" << gamma << "];" << endl;
        cout << "x = [" << x << "];" << endl;
        cout << "y = [" << y << "];" << endl;
#endif
    }

    template<class Float>
    void PReLU<Float>::cal_inerr(const tensor_type &x, const tensor_type &y,
                                 const tensor_type &dl, const param_getter &pg,
                                 tensor_type &dx) {
        auto &gamma = pg(0);

        size_type D = gamma.size(0);
        size_type N = x.dim() > 1 ? x.size(0) : 1;

        bool use_cuda = false;
#if ONN_USE_CUDA
        constexpr bool is_float = std::is_same_v<Float, float>;
        if constexpr (is_float) {
            if (CUDA_IsAvailable() && N * D > 49000) {
                use_cuda = true;
                if (dx.size() != dl.size())
                    dx.resize(dl.size());

                float *ddl, *_dx, *dg;
                cudaMalloc(&ddl, N * D * sizeof(float));
                cudaMalloc(&_dx, N * D * sizeof(float));
                cudaMalloc(&dg, D * sizeof(float));
                cudaMemcpy(ddl, dl.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(_dx, x.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(dg, gamma.data(), D * sizeof(float), cudaMemcpyHostToDevice);

                auto &prop = CUDA_GetCurrentDeviceProp();
                int gridDim = std::min(prop.maxBlocksPerMultiProcessor, (int) D);
                int blockDim = std::min(prop.maxThreadsPerBlock, (int) N);

                onn_gpu_kernel::__cudakernel_prelu_cal_inerr<<<gridDim, blockDim>>>(ddl, _dx, dg, N, D);

                cudaMemcpy(dx.data(), ddl, N * D * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(ddl);
                cudaFree(_dx);
                cudaFree(dg);
            }
        }
#endif

        if (!use_cuda) {
            if (N > 1) {
                dx = dl;
                for (size_type i = 0; i < N; i++)
                    for (size_type j = 0; j < D; j++)
                        if (x(i, j) < 0.0)
                            dx(i, j) *= gamma[j];
            } else {
                dx = dl;
                for (size_type i = 0; i < D; i++)
                    if (x[i] < 0.0)
                        dx[i] *= gamma[i];
            }
        }

#if ONN_DBG
        cout << "[PReLU - cal_inerr]" << endl;
        cout << "gamma = [" << gamma << "];" << endl;
        cout << "x = [" << x << "];" << endl;
        cout << "dy = [" << dl << "];" << endl;
        cout << "dx = [" << dx << "];" << endl;
#endif
    }

    template<class Float>
    vector<typename PReLU<Float>::tensor_type>
    PReLU<Float>::cal_grad(const tensor_type &x, const tensor_type &y,
                           const tensor_type &dl, const param_getter &pg) {
        auto &gamma = pg(0);

        size_type D = gamma.size(0);
        size_type N = x.dim() > 1 ? x.size(0) : 1;

        tensor_type dgamma(D);

        bool use_cuda = false;
#if ONN_USE_CUDA
        constexpr bool is_float = std::is_same_v<Float, float>;
        if constexpr (is_float) {
            if (CUDA_IsAvailable()) {
                use_cuda = true;

                float *ddl, *_dx, *dg;
                cudaMalloc(&ddl, N * D * sizeof(float));
                cudaMalloc(&_dx, N * D * sizeof(float));
                cudaMalloc(&dg, D * sizeof(float));
                cudaMemcpy(ddl, dl.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(_dx, x.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);

                auto &prop = CUDA_GetCurrentDeviceProp();
                int gridDim = std::min(prop.maxBlocksPerMultiProcessor, (int) D);
                int blockDim = std::min(prop.maxThreadsPerBlock, (int) N);

                onn_gpu_kernel::__cudakernel_prelu_cal_grad<<<gridDim, blockDim>>>(dg, _dx, ddl, N, D);

                cudaMemcpy(dgamma.data(), dg, D * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(ddl);
                cudaFree(_dx);
                cudaFree(dg);
            }
        }
#endif

        if (!use_cuda) {
            dgamma.fill(0.0);
            if (N > 1) {
                for (size_type i = 0; i < N; i++)
                    for (size_type j = 0; j < D; j++)
                        dgamma[j] += x(i, j) >= 0.0 ? 0.0 : dl(i, j) * x(i, j);
                dgamma /= (Float) N;
            } else {
                for (size_type i = 0; i < D; i++)
                    dgamma[i] += x[i] >= 0.0 ? 0.0 : dl[i] * x[i];
            }
        }

#if ONN_DBG
        cout << "[PReLU - cal_grad]" << endl;
        cout << "gamma = [" << gamma << "];" << endl;
        cout << "x = [" << x << "];" << endl;
        cout << "dy = [" << dl << "];" << endl;
        cout << "dgamma = [" << dgamma << "];" << endl;
#endif

        return {dgamma};
    }

    template<class T>
    std::ostream &operator<<(std::ostream &os, const PReLU<T> &n) {
        auto &gamma = n.param[0].param();

        os << "[PReLU]" << std::endl;
        os << "gamma = [" << gamma << "];" << std::endl;
        return os;
    }

    /* +-+-+-+-+-+-+-+-+-+- 线性变换单元 -+-+-+-+-+-+-+-+-+-+ */

    template<class Float>
    LinearTransUnit<Float>::LinearTransUnit(size_type D, size_type C) {
        auto &param = this->param;
        param.reserve(2);
        param.emplace_back(D, C);
        param.emplace_back(C);

        auto &W = param[0];
        auto &b = param[1];
        W.set("lambda", 0.0005);

        // He初始化
        Float r = sqrt(6.0 / (Float) D);

        using rand_gen_type = std::conditional_t<sizeof(void *) == 4, std::mt19937, std::mt19937_64>;
        rand_gen_type gen = rand_gen_type(std::random_device{}());
        std::uniform_real_distribution<Float> dis = std::uniform_real_distribution<Float>(-r, r);

        for (size_type i = 0; i < C; i++) {
            for (size_type j = 0; j < D; j++)
                W(j, i) = dis(gen);
            b[i] = 0.0;
        }
    }

    template<class Float>
    void LinearTransUnit<Float>::cal_out(const tensor_type &x, const param_getter &pg, tensor_type &y) {
        auto &W = pg(0);
        auto &b = pg(1);

        size_type D = W.size(0), C = W.size(1);
        size_type N = x.dim() > 1 ? x.size(0) : 1;
        y = x * W;

        constexpr bool is_float = std::is_same_v<Float, float>;
        bool use_cuda = false;

        if constexpr (is_float) {
            size_type n = N * C;
            if (CUDA_IsAvailable() && n > 48000) {
                use_cuda = true;

                Float *dx, *db;
                cudaMalloc(&dx, N * C * sizeof(Float));
                cudaMalloc(&db, C * sizeof(Float));
                cudaMemcpy(dx, y.data(), N * C * sizeof(Float), cudaMemcpyHostToDevice);
                cudaMemcpy(db, b.data(), C * sizeof(Float), cudaMemcpyHostToDevice);

                auto &prop = CUDA_GetCurrentDeviceProp();
                int gridDim = std::min(prop.maxBlocksPerMultiProcessor, (int) C);
                int blockDim = std::min(prop.maxThreadsPerBlock, (int) N);

                onn_gpu_kernel::tensor_add_vector<<<gridDim, blockDim>>>(dx, db, N, C);

                cudaMemcpy(y.data(), dx, N * C * sizeof(Float), cudaMemcpyDeviceToHost);

                cudaFree(dx);
                cudaFree(db);
            }
        }

        if (!use_cuda) {
            if (N > 1)
                for (size_type i = 0; i < N; i++)
                    for (size_type j = 0; j < C; j++)
                        y(i, j) += b[j];
            else
                y += b;
        }

#if ONN_DBG
        cout << "[LinearTransUnit - cal_out]" << endl;
        cout << "W = [" << W << "];" << endl;
        cout << "b = [" << b << "];" << endl;
        cout << "x = [" << x << "];" << endl;
        cout << "y = [" << y << "];" << endl;
#endif
    }

    template<class Float>
    void LinearTransUnit<Float>::cal_inerr(const tensor_type &x, const tensor_type &y,
                                           const tensor_type &dl, const param_getter &pg,
                                           tensor_type &dx) {
        auto &W = pg(0);

        // dx = dl * W.t();
        // tensor<Float>::mul(dl, W, dx, CUBLAS_OP_N, CUBLAS_OP_T);
        dx = dl * W.t();

#if ONN_DBG
        cout << "[LinearTransUnit - cal_inerr]" << endl;
        cout << "W = [" << W << "];" << endl;
        cout << "dy = [" << dl << "];" << endl;
        cout << "dx = [" << dx << "];" << endl;
#endif
    }

    template<class Float>
    vector<typename LinearTransUnit<Float>::tensor_type>
    LinearTransUnit<Float>::cal_grad(const tensor_type &x, const tensor_type &y,
                                     const tensor_type &dl, const param_getter &pg) {
        // auto dW = x.t() * dl;
        tensor<Float> dW;
        // tensor<Float>::mul(x, dl, dW, CUBLAS_OP_T, CUBLAS_OP_N);
        dW = x.t() * dl;

        auto db = dl.dim() > 1 ? dl.mean(0) : dl;
        /// TODO: 使用 CUDA 加速 mean 的计算

#if ONN_DBG
        cout << "[LinearTransUnit - cal_inerr]" << endl;
        cout << "x = [" << x << "];" << endl;
        cout << "dy = [" << dl << "];" << endl;
        cout << "dW = [" << dW << "];" << endl;
        cout << "db = [" << db << "];" << endl;
#endif

        return {dW, db};
    }

    template<class T>
    std::ostream &operator<<(std::ostream &os, const LinearTransUnit<T> &n) {
        using size_type = typename LinearTransUnit<T>::size_type;

        auto &param = n.param;
        auto &W = param[0].param();
        auto &b = param[1].param();
        size_type C = W.size(0), D = W.size(1);
        os << "[LinearTransUnit]" << std::endl;
        os << "W = [";
        for (size_type i = 0; i < C; i++) {
            os << W(i, 0);
            for (size_type j = 1; j < D; j++)
                os << ',' << W(i, j);
            if (i == C - 1)
                os << ']';
            os << ';' << std::endl;
        }
        os << "b = [" << b << "];" << std::endl;
        return os;
    }

    /* +-+-+-+-+-+-+-+-+-+- 一维卷积单元 -+-+-+-+-+-+-+-+-+-+ */

    /// @pramm D: 输入维度
    /// @pramm K: 卷积核大小
    template<class Float>
    ConvUnit<Float>::ConvUnit(size_type D, size_type K) {
        size_type C = D - K + 1;

        auto &param = this->param;
        param.reserve(2);
        param.emplace_back(K);
        param.emplace_back(C);

        auto &h = param[0];
        auto &b = param[1];

        h.set("lambda", 0.0005);

        // 设: 输入 x ~ N(0, 1)
        // h(i) ~ U(-r, r)
        // 激活函数使用 ReLU
        Float r = sqrt(6.0 / (Float) K);

        using rand_gen_type = std::conditional_t<sizeof(void *) == 4, std::mt19937, std::mt19937_64>;
        rand_gen_type gen = rand_gen_type(std::random_device{}());
        std::uniform_real_distribution<Float> dis = std::uniform_real_distribution<Float>(-r, r);

        for (size_type i = 0; i < K; i++)
            h[i] = dis(gen);
        for (size_type i = 0; i < C; i++)
            b[i] = 0.0;
    }

    template<class Float>
    void ConvUnit<Float>::cal_out(const tensor_type &x, const param_getter &pg, tensor_type &y) {
        auto &h = pg(0);
        auto &b = pg(1);

        size_type K = h.size(0), C = b.size(0), D = C + K - 1;
        size_type N = x.dim() > 1 ? x.size(0) : 1;
        if (N > 1)
            y.resize(N, C);
        else
            y.resize(C);
        _Conv(x, h, y, ValidConv);
        if (N > 1)
            for (size_type i = 0; i < N; i++)
                for (size_type j = 0; j < C; j++)
                    y(i, j) += b[j];
        else
            y += b;
    }

    template<class Float>
    void ConvUnit<Float>::cal_inerr(const tensor_type &x, const tensor_type &y,
                                    const tensor_type &dl, const param_getter &pg,
                                    tensor_type &dx) {
        auto &h = pg(0);
        auto &b = pg(1);

        size_type K = h.size(0), C = b.size(0), D = C + K - 1;
        size_type N = x.dim() > 1 ? x.size(0) : 1;

        dx.resize(x.size());

        _Conv(dl, h, dx, FullConv, false);
    }

    template<class Float>
    vector<typename ConvUnit<Float>::tensor_type>
    ConvUnit<Float>::cal_grad(const tensor_type &x, const tensor_type &y,
                              const tensor_type &dl, const param_getter &pg) {
        auto &h = pg(0);
        auto &b = pg(1);

        size_type N = x.dim() > 1 ? x.size(0) : 1;
        size_type K = h.size(0), C = b.size(0), D = C + K - 1;

        tensor_type dh, db;
        if (N > 1) {
            dh.resize(N, K);
            db.resize(N, C);
        } else {
            dh.resize(K);
            db.resize(C);
        }

        _Conv(x, dl, dh, ValidConv);
        if (N > 1)
            db = dl.mean(0);
        else db = dl;
        dh = dh.mean();

        return {dh, db};
    }

    template<class T>
    std::ostream &operator<<(std::ostream &os, const ConvUnit<T> &n) {
        using size_type = typename ConvUnit<T>::size_type;

        auto &h = n.param[0].param();
        auto &b = n.param[1].param();

        os << "[ConvUnit]" << std::endl;
        os << "h = [" << h << "];" << std::endl;
        os << "b = [" << b << "];" << std::endl;
        return os;
    }

    /* +-+-+-+-+-+-+-+-+-+- 一维汇聚单元 -+-+-+-+-+-+-+-+-+-+ */

    template<class Float>
    void AverPoolUnit<Float>::cal_out(const tensor_type &x, const param_getter &pg, tensor_type &y) {
        size_type N = x.dim() > 1 ? x.size(0) : 1;
        size_type C = (D + K - 1) / K;
        if (N > 1)
            y.resize(N, C);
        else
            y.resize(C);
        y.fill(0.0);
        if (N > 1) {
            for (size_type p = 0; p < N; p++) {
                size_type i = 0;
                // 处理可以整除的部分
                for (size_type j = 0; j < D / K; j++) {
                    Float sum = 0.0;
                    for (size_type k = 0; k < K; k++)
                        sum += x(p, i++);
                    y(p, j) = sum / (Float) K;
                }
                // 处理不能整除的部分
                if (D % K != 0) {
                    Float sum = 0.0;
                    for (size_type k = 0; k < D % K; k++)
                        sum += x(p, i++);
                    y(p, D / K) = sum / (Float) (D % K);
                }
            }
        } else {
            size_type i = 0;
            // 处理可以整除的部分
            for (size_type j = 0; j < D / K; j++) {
                Float sum = 0.0;
                for (size_type k = 0; k < K; k++)
                    sum += x[i++];
                y[j] = sum / (Float) K;
            }
            // 处理不能整除的部分
            if (D % K != 0) {
                Float sum = 0.0;
                for (size_type k = 0; k < D % K; k++)
                    sum += x[i++];
                y[D / K] = sum / (Float) (D % K);
            }
        }
    }

    template<class Float>
    void AverPoolUnit<Float>::cal_inerr(const tensor_type &x, const tensor_type &y,
                                        const tensor_type &dl, const param_getter &pg,
                                        tensor_type &dx) {
        size_type N = dl.dim() > 1 ? dl.size(0) : 1;
        size_type C = dl.size(N > 1 ? 1 : 0);
        dx.resize(x.size());
        if (N > 1) {
            for (size_type p = 0; p < N; p++) {
                size_type i = 0;
                // 处理可以整除的部分
                for (size_type j = 0; j < D / K; j++) {
                    Float d = dl[j] / (Float) K;
                    for (size_type k = 0; k < K; k++)
                        dx(p, i++) = d;
                }
                // 处理不能整除的部分
                if (D % K != 0) {
                    Float d = dl[D / K] / (Float) (D % K);
                    for (size_type k = 0; k < D % K; k++)
                        dx(p, i++) = d;
                }
            }
        } else {
            size_type i = 0;
            // 处理可以整除的部分
            for (size_type j = 0; j < D / K; j++) {
                Float d = dl[j] / (Float) K;
                for (size_type k = 0; k < K; k++)
                    dx[i++] = d;
            }
            // 处理不能整除的部分
            if (D % K != 0) {
                Float d = dl[D / K] / (Float) (D % K);
                for (size_type k = 0; k < D % K; k++)
                    dx[i++] = d;
            }
        }
    }

    template<class T>
    std::ostream &operator<<(std::ostream &os, const AverPoolUnit<T> &n) {
        os << "[AverPoolUnit]" << std::endl;
        return os;
    }
} // namespace ocean
