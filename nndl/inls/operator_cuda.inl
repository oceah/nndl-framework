#pragma once

#include "../operator_cuda.h"

#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

namespace ocean::nndl {
    /* +-+-+-+-+-+-+-+-+-+- basic[unary] -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    void fill_cuda(T *dst, T v, size_t size) {
        thrust::fill(dst, dst + size, v);
    }

    template<class T>
    void assign_cuda(T *dst, const T *src, size_t size) {
        cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    template<class T>
    void assign_cuda(T *dst, const T &src, size_t size) {
        thrust::device_ptr<T> dd(dst);
        thrust::fill(dd, dd + size, src);
    }

    template<class T>
    void neg_cuda(T *dst, const T *src, size_t size) {
        thrust::device_ptr<const T> ds(src);
        thrust::device_ptr<T> dd(dst);
        thrust::transform(ds, ds + size, dd, cuda::std::negate<T>());
    }

    /* +-+-+-+-+-+-+-+-+-+- basic[binary] -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    void add_cuda(const T *a, const T *b, T *c, size_t size) {
        // Create device pointers
        thrust::device_ptr<const T> da(a);
        thrust::device_ptr<const T> db(b);
        thrust::device_ptr<T> dc(c);
        // Perform addition using thrust::transform
        thrust::transform(da, da + size, db, dc, cuda::std::plus<T>());
    }

    /// add_cuda begin

    /// @brief unary functor: y = x * s
    template<class T>
    struct add_functor {
        T s;

        __host__ __device__ T operator()(const T &x) const {
            return x + s;
        }
    };

    template<class T>
    void add_cuda(const T *a, T *b, const T &s, size_t size) {
        thrust::device_ptr<const T> da(a);
        thrust::device_ptr<T> db(b);
        thrust::transform(da, da + size, db, add_functor<T>{s});
    }

    /// add_cuda end

    template<class T>
    void sub_cuda(const T *a, const T *b, T *c, size_t size) {
        // Create device pointers
        thrust::device_ptr<const T> da(a);
        thrust::device_ptr<const T> db(b);
        thrust::device_ptr<T> dc(c);
        // Perform subtraction using thrust::transform
        thrust::transform(da, da + size, db, dc, cuda::std::minus<T>());
    }

    /// @brief unary functor: y = x * s
    template<class T>
    struct mul_scalar_functor {
        T s;

        __host__ __device__ T operator()(const T &x) const {
            return x * s;
        }
    };

    template<class T>
    void scalar_mul_cuda(const T *a, T *b, const T &s, size_t size) {
        thrust::device_ptr<const T> da(a);
        thrust::device_ptr<T> db(b);
        thrust::transform(da, da + size, db, mul_scalar_functor<T>{s});
    }

    template<class T>
    void matmul_cuda(const T *a, const T *b, T *c, size_t m, size_t k, size_t n,
                     bool trans_a, bool trans_b) {
        cublasHandle_t handle;
        cublasCreate(&handle);

        auto transa = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        auto transb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

        auto la = transa == CUBLAS_OP_N ? k : m;
        auto lb = transb == CUBLAS_OP_N ? n : k;

        T alpha = 1.0;
        T beta = 0.0;
        if constexpr (std::is_same_v<T, float>) {
            transa = transa == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
            transb = transb == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
            T *p = nullptr;
            if (cudaMalloc(&p, m * n * sizeof(T)) != cudaSuccess)
                throw std::runtime_error("cudaMalloc failed in matmul_cuda");
            cublasSgemm(handle, transa, transb,
                        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                        &alpha, a, static_cast<int>(la), b, static_cast<int>(lb),
                        &beta, p, static_cast<int>(m));
            cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(n), static_cast<int>(m),
                        &alpha, p, static_cast<int>(m), &beta, p, static_cast<int>(n),
                        c, static_cast<int>(n));
            cudaFree(p);
        } else
            throw std::invalid_argument("unsupported data type");

        cublasDestroy(handle);
    }
} // namespace ocean::nndl
