/**
******************************************************************************
* @file             math.h
* @brief            math library[CPU/GPU]
* @author           OceanH
* @create           2025-02-19
* @latestupdate     2025-02-19
******************************************************************************
*/

#ifndef NNDL_MATH_H
#define NNDL_MATH_H

#include "tensor.h"

namespace ocean::nndl
{
    /* +-+-+-+-+-+-+-+-+-+- operators -+-+-+-+-+-+-+-+-+-+ */

    /// @brief tensor scalar exp
    template <class T>
    class exp_op : public tensor_op<T>
    {
    public:
        using tensor_type = tensor_op<T>::tensor_type;
        using tensor_ptr = tensor_op<T>::tensor_ptr;
        using size_type = tensor_op<T>::size_type;

        /// @brief operator execution
        void forward(DeviceTypeDef device,
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

    /// @brief tensor power y = x^N, N is a compile-time constant
    template <class T, int N>
    class power_op : public tensor_op<T>
    {
    public:
        using tensor_type = tensor_op<T>::tensor_type;
        using tensor_ptr = tensor_op<T>::tensor_ptr;
        using size_type = tensor_op<T>::size_type;

        /// @brief operator execution
        void forward(DeviceTypeDef device,
                     const tensor_ptr *ins, size_type in_num,
                     tensor_ptr *outs, size_type out_num,
                     const T *params) override;

        /// @brief back propagation: dx = dy * N * x^(N-1)
        void backward(DeviceTypeDef device,
                      const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                      const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                      const T *params) override;
    };

    /* +-+-+-+-+-+-+-+-+-+- functions -+-+-+-+-+-+-+-+-+-+ */

    /// @brief tensor scalar exp
    template <class T>
    tensor<T> exp(tensor<T> &&x);

    template <class T>
    tensor<T> exp(const tensor<T> &x);

    /// @brief tensor power y = x^N
    template <class T, int N>
    tensor<T> power(tensor<T> &&x);

    template <class T, int N>
    tensor<T> power(const tensor<T> &x);
} // namespace ocean::nndl

#include "inls/math.inl"

#endif // NNDL_MATH_H
