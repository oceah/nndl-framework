/**
******************************************************************************
* @file         gmath.h
* @brief        CUDA数学库
* @author       透小犹
* @create       2024-10-29
* @lastupdate   2024-10-30
******************************************************************************
* @update
* 2024-10-30    添加一维卷积 API
******************************************************************************
*/

#ifndef GMATH_H
#define GMATH_H

namespace ocean {

    namespace cuda_kernel {

        /// @brief 指数函数
        /// @param x 输入张量[N, C]
        /// @param N 数据量
        __global__ void exp(float *x, int N);

        /// @brief 求和
        /// @param x 输入张量[N, C]
        /// @param s 数据和向量[N]
        /// @param N 数据量
        /// @param C 数据维度
        __global__ void sum(const float *x, float *s, int N, int C);

    } // namespace cuda_kernel


    /* +-+-+-+-+-+-+-+-+-+- API -+-+-+-+-+-+-+-+-+-+ */

    /// @brief softmax 函数
    /// @param x 输入张量[N, C]
    /// @param N 数据量
    /// @param C 数据维度
    /// @assert x 列优先存储
    void softmax_g(float *x, int N, int C);

    /// @brief 卷积
    /// @param x 输入序列[N, Nx]
    /// @param h 卷积核[Rh, Nh]
    /// @param y 输出序列[N, Ny]
    /// @param Nx 输入序列长度
    /// @param Nh 卷积核长度
    /// @param N 数据量
    /// @param Rh 卷积核数量 1 or N
    /// @param pad 填充 卷积核在边界处如何处理输入(0: 不填充)
    /// @param stride 步长 每次卷积滑动的步长(1: 卷积窗口逐点移动)
    /// @param dilation 扩展 卷积核的扩展(1: 普通卷积)
    void conv_g(const float *x, const float *h, float *y, int Nx, int Nh,
                int N = 1, int Rh = 1,
                int pad = 0, int stride = 1, int dilation = 1);

    /// @brief 矩阵转置
    /// @param x 输入矩阵[R, C]
    /// @param y 输出矩阵[C, R]
    void transpose_g(const float *x, float *y, int R, int C);

} // namespace ocean

#endif // GMATH_H
