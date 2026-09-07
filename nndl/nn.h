/**
******************************************************************************
* @file             nn.h
* @brief            neural network header file
* @author           OceanH
* @create           2025-02-19
* @latestupdate     2025-02-19
******************************************************************************
*/

#ifndef NNDL_NN_H
#define NNDL_NN_H

#include "tensor.h"

namespace ocean::nn
{
    template <class T>
    class Linear
    {
    public:
        using size_type = tensor<T>::size_type;

        Linear(size_type in_features,
               size_type out_features,
               bool bias = true,
               DeviceTypeDef device = {EDevice::ANY, 0});

        /// @param x N x in_features
        /// @return N x out_features
        tensor<T> operator()(const tensor<T> &x) const;

        /* +-+-+-+-+-+-+-+-+-+- getters -+-+-+-+-+-+-+-+-+-+ */

        tensor<T> &weight() { return w; }

        const tensor<T> &weight() const { return w; }

        tensor<T> &bias() { return b; }

        const tensor<T> &bias() const { return b; }

        /* +-+-+-+-+-+-+-+-+-+- setters -+-+-+-+-+-+-+-+-+-+ */

        template <class... Args>
        void to(Args &&...args) &;

        void update(T lr);

        /* +-+-+-+-+-+-+-+-+-+- friendss -+-+-+-+-+-+-+-+-+-+ */

        template <class U>
        friend std::ostream &operator<<(std::ostream &os, const Linear<U> &linear);

    private:
        size_type _in_features;
        size_type _out_features;
        /// @any auto select the calculation device
        /// @cpu all the calculation is done on the CPU
        /// @cuda all the calculation is done on the NVIDIA GPU
        DeviceTypeDef _device = {EDevice::ANY, 0};

        // parameters
        tensor<T> w, b;
    };
} // namespace ocean::nn

#include "inls/nn.inl"

#endif // NNDL_NN_H
