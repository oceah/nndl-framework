/**
******************************************************************************
* @file             operator_cpu.h
* @brief            operator library[CPU]
* @author           OceanH
* @create           2025-01-14
* @latestupdate     2025-02-17
******************************************************************************
*/

#ifndef OPERATOR_CPU_H
#define OPERATOR_CPU_H

namespace ocean::nndl {
    /* +-+-+-+-+-+-+-+-+-+- basic[unary] -+-+-+-+-+-+-+-+-+-+ */

    /// @brief tensor assignment
    template<class T>
    void assign_cpu(T *dst, const T *src, size_t size);

    /// @brief tensor assignment with scalar
    template<class T>
    void assign_cpu(T *dst, const T &src, size_t size);

    /// @brief tensor negation
    template<class T>
    void neg_cpu(T *dst, const T *src, size_t size);

    /* +-+-+-+-+-+-+-+-+-+- basic[binary] -+-+-+-+-+-+-+-+-+-+ */

    /// @brief tensor addition
    template<class T>
    void add_cpu(const T *a, const T *b, T *c, size_t size);

    /// @brief scalar addition
    template<class T>
    void add_cpu(const T *a, T *b, const T &s, size_t size);

    /// @brief tensor subtraction
    template<class T>
    void sub_cpu(const T *a, const T *b, T *c, size_t size);

    template<class T>
    void scalar_mul_cpu(const T *a, T *b, const T &s, size_t size);

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
    void matmul_cpu(const T *a, const T *b, T *c, size_t m, size_t k, size_t n,
                    bool trans_a = false, bool trans_b = false);
} // namespace ocean

#include "inls/operator_cpu.inl"

#endif // OPERATOR_CPU_H
