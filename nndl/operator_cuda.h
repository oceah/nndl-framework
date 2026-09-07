/**
******************************************************************************
* @file             operator_cuda.h
* @brief            operator library[CUDA]
* @author           OceanH
* @create           2025-01-22
* @latestupdate     2025-02-17
******************************************************************************
*/

#ifndef OPERATOR_CUDA_H
#define OPERATOR_CUDA_H

namespace ocean::nndl {
    /* +-+-+-+-+-+-+-+-+-+- basic[unary] -+-+-+-+-+-+-+-+-+-+ */

    /// @brief tensor assignment
    template<class T>
    void assign_cuda(T *dst, const T *src, size_t size);

    /// @brief tensor assignment with scalar
    template<class T>
    void assign_cuda(T *dst, const T &src, size_t size);

    /// @brief tensor negation
    template<class T>
    void neg_cuda(T *dst, const T *src, size_t size);

    /* +-+-+-+-+-+-+-+-+-+- basic[binary] -+-+-+-+-+-+-+-+-+-+ */

    /// @brief tensor addition
    template<class T>
    void add_cuda(const T *a, const T *b, T *c, size_t size);

    /// @brief tensor addition with scalar
    template<class T>
    void add_cuda(const T *a, T *b, const T &s, size_t size);

    /// @brief tensor subtraction
    template<class T>
    void sub_cuda(const T *a, const T *b, T *c, size_t size);

    template <class T>
    void scalar_mul_cuda(const T *a, T *b, const T &s, size_t size);

    /// @brief matrix multiplication
    /// @param a matrix A[m, k]
    /// @param b matrix B[k, n]
    /// @param c matrix C[m, n]
    /// @param m logical row of matrix A
    /// @param k logical column of matrix A and row of matrix B
    /// @param n logical column of matrix B
    /// @param trans_a matrix A is transposed[colume-major]
    /// @param trans_b matrix B is transposed[colume-major]
    template<class T>
    void matmul_cuda(const T *a, const T *b, T *c, size_t m, size_t k, size_t n,
                     bool trans_a = false, bool trans_b = false);
} // namespace ocean::nndl

#include "inls/operator_cuda.inl"

#endif //OPERATOR_CUDA_H
