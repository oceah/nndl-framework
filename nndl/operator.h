/**
******************************************************************************
* @file             operator.h
* @brief            operator library[CPU/GPU]
* @author           OceanH
* @create           2025-01-07
* @latestupdate     2025-02-17
******************************************************************************
*/

#ifndef OPERATOR_H
#define OPERATOR_H

#include "utility.h"

#include <functional>

using std::function;

namespace ocean::nndl {
    /* +-+-+-+-+-+-+-+-+-+- basic -+-+-+-+-+-+-+-+-+-+ */

    /// @brief tensor assignment
    template<class T>
    void assign(DeviceTypeDef device, T *dst, const T *src, size_t size);

    /// @brief tensor assignment with scalar
    template<class T>
    void assign(DeviceTypeDef device, T *dst, const T &src, size_t size);

    /// @brief tensor negationg
    template<class T>
    void neg(DeviceTypeDef device, T *dst, const T *src, size_t size);

    /// @brief tensor addition
    template<class T>
    void add(DeviceTypeDef device, const T *a, const T *b, T *c, size_t size);

    /// @brief tensor addition with scalar
    template<class T>
    void add(DeviceTypeDef device, const T *a, const T &s, T *b, size_t size);

    /// @brief tensor subtraction
    template<class T>
    void sub(DeviceTypeDef device, const T *a, const T *b, T *c, size_t size);

    template<class T>
    void scalar_mul(DeviceTypeDef device, const T *a, T *b, const T &s, size_t size);

    /// @brief matrix multiplication
    /// @param device
    /// @param a matrix A[m, k]
    /// @param b matrix B[k, n]
    /// @param c matrix C[m, n]
    /// @param m logical row of matrix A
    /// @param k logical column of matrix A and row of matrix B
    /// @param n logical column of matrix B
    /// @param trans_a matrix A is transposed[colume-major]
    /// @param trans_b matrix B is transposed[colume-major]
    template<class T>
    void matmul(DeviceTypeDef device, const T *a, const T *b, T *c,
                size_t m, size_t k, size_t n,
                bool trans_a = false, bool trans_b = false);

    /* +-+-+-+-+-+-+-+-+-+- unary scalar function -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    void transform(DeviceTypeDef device, const function<T(const T &)> &f,
                   const T *x, T *y, size_t size);

    /* +-+-+-+-+-+-+-+-+-+- operator base -+-+-+-+-+-+-+-+-+-+ */

    /// @brief tensor operator base class
    /// @design not responsible for memory management and any pre-checks
    template<class T>
    class tensor_op {
    public:
        using tensor_type = tensor_kernel<T>;
        using tensor_ptr = tensor_type *;
        using size_type = tensor_type::size_type;

        /// @brief operator execution
        virtual void forward(DeviceTypeDef device,
                             const tensor_ptr *ins, size_type in_num,
                             tensor_ptr *outs, size_type out_num,
                             const T *params) = 0;

        /// @brief back propagation of gradients
        virtual void backward(DeviceTypeDef device,
                              const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                              const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                              const T *params) = 0;

        virtual ~tensor_op() = default;
    };

    /* +-+-+-+-+-+-+-+-+-+- basic operators -+-+-+-+-+-+-+-+-+-+ */

    /// @brief tensor assignment operator
    template<class T>
    class assign_op : public tensor_op<T> {
    public:
        using tensor_type = tensor_op<T>::tensor_type;
        using tensor_ptr = tensor_op<T>::tensor_ptr;
        using size_type = tensor_op<T>::size_type;

        /// @brief operator execution
        virtual void forward(DeviceTypeDef device,
                             const tensor_ptr *ins, size_type in_num,
                             tensor_ptr *outs, size_type out_num,
                             const T *params) override;

        /// @brief back propagation of gradients
        /// @param device
        /// @param outs
        /// @param out_grads
        /// @param out_num
        /// @param ins
        /// @param in_grads set null if no gradient required
        /// @param in_num
        /// @param params
        void backward(DeviceTypeDef device,
                      const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                      const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                      const T *params) override;
    };

    /// @brief tensor negation operator
    template<class T>
    class neg_op : public tensor_op<T> {
    public:
        using tensor_type = tensor_op<T>::tensor_type;
        using tensor_ptr = tensor_op<T>::tensor_ptr;
        using size_type = tensor_op<T>::size_type;

        /// @brief operator execution
        virtual void forward(DeviceTypeDef device,
                             const tensor_ptr *ins, size_type in_num,
                             tensor_ptr *outs, size_type out_num,
                             const T *params) override;

        /// @brief back propagation of gradients
        virtual void backward(DeviceTypeDef device,
                              const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                              const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                              const T *params) override;
    };

    /// @brief tensor addition operator
    template<class T>
    class add_op : public tensor_op<T> {
    public:
        using tensor_type = tensor_op<T>::tensor_type;
        using tensor_ptr = tensor_op<T>::tensor_ptr;
        using size_type = tensor_op<T>::size_type;

        /// @brief operator execution
        virtual void forward(DeviceTypeDef device,
                             const tensor_ptr *ins, size_type in_num,
                             tensor_ptr *outs, size_type out_num,
                             const T *params) override;

        /// @brief back propagation of gradients
        virtual void backward(DeviceTypeDef device,
                              const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                              const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                              const T *params) override;
    };

    /// @brief tensor subtraction operator
    template<class T>
    class sub_op : public tensor_op<T> {
    public:
        using tensor_type = tensor_op<T>::tensor_type;
        using tensor_ptr = tensor_op<T>::tensor_ptr;
        using size_type = tensor_op<T>::size_type;

        /// @brief operator execution
        virtual void forward(DeviceTypeDef device,
                             const tensor_ptr *ins, size_type in_num,
                             tensor_ptr *outs, size_type out_num,
                             const T *params) override;

        /// @brief back propagation of gradients
        virtual void backward(DeviceTypeDef device,
                              const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                              const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                              const T *params) override;
    };

    /// @brief matrix multiplication operator
    template<class T>
    class matmul_op : public tensor_op<T> {
    public:
        using tensor_type = tensor_op<T>::tensor_type;
        using tensor_ptr = tensor_op<T>::tensor_ptr;
        using size_type = tensor_op<T>::size_type;

        /// @brief operator execution
        virtual void forward(DeviceTypeDef device,
                             const tensor_ptr *ins, size_type in_num,
                             tensor_ptr *outs, size_type out_num,
                             const T *params) override;

        /// @brief back propagation of gradients
        virtual void backward(DeviceTypeDef device,
                              const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                              const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                              const T *params) override;
    };
} // namespace ocean::nndl

#include "inls/operator.inl"

#endif // OPERATOR_H
