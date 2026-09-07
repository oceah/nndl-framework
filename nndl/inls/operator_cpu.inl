#pragma once

#include "../operator_cpu.h"

namespace ocean::nndl {
    /* +-+-+-+-+-+-+-+-+-+- basic[unary] -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    void fill_cpu(T *dst, T v, size_t size) {
        while (size--)
            *dst++ = v;
    }

    template<class T>
    void assign_cpu(T *dst, const T *src, size_t size) {
        while (size--)
            *dst++ = *src++;
    }

    template<class T>
    void assign_cpu(T *dst, const T &src, size_t size) {
        while (size--)
            *dst++ = src;
    }

    template<class T>
    void neg_cpu(T *dst, const T *src, size_t size) {
        while (size--)
            *dst++ = -(*src++);
    }

    /* +-+-+-+-+-+-+-+-+-+- basic[binary] -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    void add_cpu(const T *a, const T *b, T *c, size_t size) {
        while (size--)
            *c++ = *a++ + *b++;
    }

    template<class T>
    void add_cpu(const T *a, T *b, const T &s, size_t size) {
        while (size--)
            *b++ = *a++ + s;
    }

    template<class T>
    void sub_cpu(const T *a, const T *b, T *c, size_t size) {
        while (size--)
            *c++ = *a++ - *b++;
    }

    template<class T>
    void scalar_mul_cpu(const T *a, T *b, const T &s, size_t size) {
        while (size--)
            *b++ = *a++ * s;
    }

    /// implementation of matrix binary scalar operation begin

    /// @brief matrix binary scalar operation
    /// @param op scalar operation
    /// @param a matrix A[m, n] with physical row la
    /// @param la
    /// @param b matrix B[m, n] with physical row lb
    /// @param lb
    /// @param c matrix C[m, n] with physical row lc
    /// @param lc
    /// @param m logical row of matrix
    /// @param n logical column of matrix
    /// @param transa matrix A is transposed[colume-major]
    /// @param transb matrix B is transposed[colume-major]
    template<class T>
    void __matbinscalar_cpu(const std::function<T(const T &, const T &)> &op,
                            const T *a, size_t la, const T *b, size_t lb,
                            T *c, size_t lc, size_t m, size_t n,
                            bool transa, bool transb) {
        if (!transa) {
            if (!transb) {
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n; j++)
                        c[j] = op(a[j], b[j]);
                    a += la;
                    b += lb;
                    c += lc;
                }
            } else {
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n; j++)
                        c[j] = op(a[j], b[j * lb]);
                    a += la;
                    ++b;
                    c += lc;
                }
            }
        } else {
            if (!transb) {
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n; j++)
                        c[j] = op(a[j * la], b[j]);
                    ++a;
                    b += lb;
                    c += lc;
                }
            } else {
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n; j++)
                        c[j] = op(a[j * la], b[j * lb]);
                    ++a;
                    ++b;
                    c += lc;
                }
            }
        }
    }

    template<class T>
    void __matadd_cpu(const T *a, size_t la, const T *b, size_t lb,
                      T *c, size_t lc, size_t m, size_t n,
                      bool transa, bool transb) {
        std::function<T(const T &, const T &)> op =
                [](const T &x, const T &y) -> T { return x + y; };
        __matbinscalar_cpu(op, a, la, b, lb, c, lc, m, n, transa, transb);
    }

    template<class T>
    void __matsub_cpu(const T *a, size_t la, const T *b, size_t lb,
                      T *c, size_t lc, size_t m, size_t n,
                      bool transa, bool transb) {
        std::function<T(const T &, const T &)> op =
                [](const T &x, const T &y) -> T { return x - y; };
        __matbinscalar_cpu(op, a, la, b, lb, c, lc, m, n, transa, transb);
    }

    /// implementation of matrix binary scalar operation end

    /// implementation of matrix multiplication begin

    /// @brief matrix multiplication
    /// @param a matrix A[m, k] with physical row la
    /// @param la
    /// @param b matrix B[k, n] with physical row lb
    /// @param lb
    /// @param c matrix C[m, n] with physical row lc
    /// @param lc
    /// @param m logical row of matrix A
    /// @param k logical column of matrix A and row of matrix B
    /// @param n logical column of matrix
    /// @param transa matrix A is transposed[colume-major]
    /// @param transb matrix B is transposed[colume-major]
    template<class T>
    void __matmul_cpu(const T *a, size_t la, const T *b, size_t lb,
                      T *c, size_t lc, size_t m, size_t k, size_t n,
                      bool transa, bool transb);

    template<class T>
    void __matmul_cpu_generate(const T *a, size_t la, const T *b, size_t lb,
                               T *c, size_t lc, size_t m, size_t k, size_t n,
                               bool transa, bool transb);

    template<class T>
    void __matmul_cpu_divide(const T *a, size_t la, const T *b, size_t lb,
                             T *c, size_t lc, size_t m, size_t k, size_t n,
                             bool transa, bool transb);

    template<class T>
    void __matmul_cpu_strassen(const T *a, size_t la, const T *b, size_t lb,
                               T *c, size_t lc, size_t n, bool transa, bool transb);

    template<class T>
    void __matmul_cpu(const T *a, size_t la, const T *b, size_t lb,
                      T *c, size_t lc, size_t m, size_t k, size_t n,
                      bool transa, bool transb) {
        constexpr size_t threshold = 128;
        if (m >= threshold && k >= threshold && n >= threshold) {
            if (m == k && k == n && !(n & 1))
                __matmul_cpu_strassen(a, la, b, lb, c, lc, n, transa, transb);
            else __matmul_cpu_divide(a, la, b, lb, c, lc, m, k, n, transa, transb);
        } else __matmul_cpu_generate(a, la, b, lb, c, lc, m, k, n, transa, transb);
    }

    template<class T>
    void __matmul_cpu_generate(const T *a, size_t la, const T *b, size_t lb,
                               T *c, size_t lc, size_t m, size_t k, size_t n,
                               bool transa, bool transb) {
        if (!transa) {
            if (!transb) {
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n; j++) {
                        c[j] = 0;
                        const T *bj = b + j;
                        for (size_t l = 0; l < k; l++)
                            c[j] += a[l] * bj[l * lb];
                    }
                    a += la;
                    c += lc;
                }
            } else {
                for (size_t i = 0; i < m; i++) {
                    const T *bj = b;
                    for (size_t j = 0; j < n; j++) {
                        c[j] = 0;
                        for (size_t l = 0; l < k; l++)
                            c[j] += a[l] * bj[l];
                        bj += lb;
                    }
                    a += la;
                    c += lc;
                }
            }
        } else {
            if (!transb) {
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n; j++) {
                        c[j] = 0;
                        const T *bj = b + j;
                        for (size_t l = 0; l < k; l++)
                            c[j] += a[l * la] * bj[l * lb];
                    }
                    ++a;
                    c += lc;
                }
            } else {
                for (size_t i = 0; i < m; i++) {
                    const T *bj = b;
                    for (size_t j = 0; j < n; j++) {
                        c[j] = 0;
                        for (size_t l = 0; l < k; l++)
                            c[j] += a[l * la] * bj[l];
                        bj += lb;
                    }
                    ++a;
                    c += lc;
                }
            }
        }
    }

    template<class T>
    void __matmul_cpu_divide(const T *a, size_t la, const T *b, size_t lb,
                             T *c, size_t lc, size_t m, size_t k, size_t n,
                             bool transa, bool transb) {
        // find the max pow of 2, which is less than or equal to m, k, n
        size_t p2 = std::min({m, k, n});
        while (p2 & (p2 - 1))
            p2 &= p2 - 1;

        const T *a11, *a12, *a21, *a22, *b11, *b12, *b21, *b22;
        if (!transa) {
            a11 = a;
            a12 = a + p2;
            a21 = a + p2 * la;
            a22 = a + p2 * la + p2;
        } else {
            a11 = a;
            a21 = a + p2;
            a12 = a + p2 * la;
            a22 = a + p2 * la + p2;
        }
        if (!transb) {
            b11 = b;
            b12 = b + p2;
            b21 = b + p2 * lb;
            b22 = b + p2 * lb + p2;
        } else {
            b11 = b;
            b21 = b + p2;
            b12 = b + p2 * lb;
            b22 = b + p2 * lb + p2;
        }
        T *c11 = c;
        T *c12 = c + p2;
        T *c21 = c + p2 * lc;
        T *c22 = c + p2 * lc + p2;

        size_t buf_row = std::max(p2, m - p2);
        size_t buf_col = std::max(p2, n - p2);
        auto buf = new T[buf_row * buf_col];

        // -                                -
        // |  A11(p2,p2)    A12(p2,k-p2)    |
        // |  A21(m-p2,p2)  A22(m-p2,k-p2)  |
        // -                                -

        // -                                -
        // |  B11(p2,p2)    B12(p2,n-p2)    |
        // |  B21(k-p2,p2)  B22(k-p2,n-p2)  |
        // -                                -

        // -                                -
        // |  C11(p2,p2)    C12(p2,n-p2)    |
        // |  C21(m-p2,p2)  C22(m-p2,n-p2)  |
        // -                                -

        // C11 = A11 * B11 + A12 * B21
        __matmul_cpu(a11, la, b11, lb, c11, lc, p2, p2, p2, transa, transb);
        __matmul_cpu(a12, la, b21, lb, buf, p2, p2, k - p2, p2, transa, transb);
        __matadd_cpu(c11, lc, buf, p2, c11, lc, p2, p2, false, false);

        // C12 = A11 * B12 + A12 * B22
        __matmul_cpu(a11, la, b12, lb, c12, lc, p2, p2, n - p2, transa, transb);
        __matmul_cpu(a12, la, b22, lb, buf, n - p2, p2, k - p2, n - p2, transa, transb);
        __matadd_cpu(c12, lc, buf, n - p2, c12, lc, p2, n - p2, false, false);

        // C21 = A21 * B11 + A22 * B21
        __matmul_cpu(a21, la, b11, lb, c21, lc, m - p2, p2, p2, transa, transb);
        __matmul_cpu(a22, la, b21, lb, buf, p2, m - p2, k - p2, p2, transa, transb);
        __matadd_cpu(c21, lc, buf, p2, c21, lc, m - p2, p2, false, false);

        // C22 = A21 * B12 + A22 * B22
        __matmul_cpu(a21, la, b12, lb, c22, lc, m - p2, p2, n - p2, transa, transb);
        __matmul_cpu(a22, la, b22, lb, buf, n - p2, m - p2, k - p2, n - p2, transa, transb);
        __matadd_cpu(c22, lc, buf, n - p2, c22, lc, m - p2, n - p2, false, false);

        delete[] buf;
    }

    template<class T>
    void __matmul_cpu_strassen(const T *a, size_t la, const T *b, size_t lb,
                               T *c, size_t lc, size_t n, bool transa, bool transb) {
        // assert a[n, n] b[n, n] c[n, n] && !(n & 1)
        if (n & 1)
            throw std::invalid_argument("assert failed: n is odd");
        n >>= 1;

        const T *a11, *a12, *a21, *a22, *b11, *b12, *b21, *b22;
        if (!transa) {
            a11 = a;
            a12 = a + n;
            a21 = a + n * la;
            a22 = a + n * la + n;
        } else {
            a11 = a;
            a21 = a + n;
            a12 = a + n * la;
            a22 = a + n * la + n;
        }
        if (!transb) {
            b11 = b;
            b12 = b + n;
            b21 = b + n * lb;
            b22 = b + n * lb + n;
        } else {
            b11 = b;
            b21 = b + n;
            b12 = b + n * lb;
            b22 = b + n * lb + n;
        }
        T *c11 = c;
        T *c12 = c + n;
        T *c21 = c + n * lc;
        T *c22 = c + n * lc + n;

        auto buf1 = new T[n * n];
        auto buf2 = new T[n * n];

        // M1 = (A11 + A22) * (B11 + B22)
        auto m1 = new T[n * n];
        __matadd_cpu(a11, la, a22, la, buf1, n, n, n, transa, transa);
        __matadd_cpu(b11, lb, b22, lb, buf2, n, n, n, transb, transb);
        __matmul_cpu(buf1, n, buf2, n, m1, n, n, n, n, false, false);

        // M2 = (A21 + A22) * B11
        auto m2 = new T[n * n];
        __matadd_cpu(a21, la, a22, la, buf1, n, n, n, transa, transa);
        __matmul_cpu(buf1, n, b11, lb, m2, n, n, n, n, false, transb);

        // M3 = A11 * (B12 - B22)
        auto m3 = new T[n * n];
        __matsub_cpu(b12, lb, b22, lb, buf1, n, n, n, transb, transb);
        __matmul_cpu(a11, la, buf1, n, m3, n, n, n, n, transa, false);

        // M4 = A22 * (B21 - B11)
        auto m4 = new T[n * n];
        __matsub_cpu(b21, lb, b11, lb, buf1, n, n, n, transb, transb);
        __matmul_cpu(a22, la, buf1, n, m4, n, n, n, n, transa, false);

        // M5 = (A11 + A12) * B22
        auto m5 = new T[n * n];
        __matadd_cpu(a11, la, a12, la, buf1, n, n, n, transa, transa);
        __matmul_cpu(buf1, n, b22, lb, m5, n, n, n, n, false, transb);

        // M6 = (A21 - A11) * (B11 + B12)
        auto m6 = new T[n * n];
        __matsub_cpu(a21, la, a11, la, buf1, n, n, n, transa, transa);
        __matadd_cpu(b11, lb, b12, lb, buf2, n, n, n, transb, transb);
        __matmul_cpu(buf1, n, buf2, n, m6, n, n, n, n, false, false);

        // M7 = (A12 - A22) * (B21 + B22)
        auto m7 = new T[n * n];
        __matsub_cpu(a12, la, a22, la, buf1, n, n, n, transa, transa);
        __matadd_cpu(b21, lb, b22, lb, buf2, n, n, n, transb, transb);
        __matmul_cpu(buf1, n, buf2, n, m7, n, n, n, n, false, false);

        // C11 = M1 + M4 - M5 + M7
        __matadd_cpu(m1, n, m4, n, buf1, n, n, n, false, false);
        __matsub_cpu(buf1, n, m5, n, buf2, n, n, n, false, false);
        __matadd_cpu(buf2, n, m7, n, c11, lc, n, n, false, false);

        // C12 = M3 + M5
        __matadd_cpu(m3, n, m5, n, c12, lc, n, n, false, false);

        // C21 = M2 + M4
        __matadd_cpu(m2, n, m4, n, c21, lc, n, n, false, false);

        // C22 = M1 - M2 + M3 + M6
        __matsub_cpu(m1, n, m2, n, buf1, n, n, n, false, false);
        __matadd_cpu(buf1, n, m3, n, buf2, n, n, n, false, false);
        __matadd_cpu(buf2, n, m6, n, c22, lc, n, n, false, false);

        delete[] buf1;
        delete[] buf2;
        delete[] m1;
        delete[] m2;
        delete[] m3;
        delete[] m4;
        delete[] m5;
        delete[] m6;
        delete[] m7;
    }

    /// implementation of matrix multiplication end

    template<class T>
    void matmul_cpu(const T *a, const T *b, T *c, size_t m, size_t k, size_t n,
                    bool trans_a, bool trans_b) {
        auto la = trans_a ? m : k;
        auto lb = trans_b ? k : n;
        __matmul_cpu(a, la, b, lb, c, n, m, k, n, trans_a, trans_b);
    }
} // namespace ocean::nndl
