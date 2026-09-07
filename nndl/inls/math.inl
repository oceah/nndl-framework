#pragma once

#include "../math.h"

#include <cmath>

namespace ocean::nndl {
    /* +-+-+-+-+-+-+-+-+-+- operators -+-+-+-+-+-+-+-+-+-+ */

    /// tensor scalar exp begin

    template<class T>
    void exp_op<T>::forward(DeviceTypeDef device,
                            const tensor_ptr *ins, size_type in_num,
                            tensor_ptr *outs, size_type out_num,
                            const T *params) {
        auto &x = *ins[0];
        auto &y = *outs[0];
        transform<T>(device, [](const T &_x) { return std::exp(_x); }, x.pdata, y.pdata, x.numel(device));
    }

    template<class T>
    void exp_op<T>::backward(DeviceTypeDef device,
                             const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                             const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                             const T *params) {
        if (in_grads == nullptr || in_grads[0] == nullptr)
            return;
        auto &y = *outs[0];
        auto &dy = *out_grads[0];
        auto &dx = *in_grads[0];

        auto n = y.numel(device);
        auto pdx = dx.pdata;
        auto pdy = dy.pdata;
        auto py = y.pdata;

        switch (device.type) {
            case EDevice::CPU:
                while (n--)
                    *pdx++ = *pdy++ * *py++;
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                throw std::runtime_error("insert code here");
#endif
            default:
                throw std::runtime_error("unknown device type");
        }
    }

    /// tensor scalar exp end

    /// tensor power begin

    /// @brief device-safe power helper: x^e
    template<class T>
    __ocean__host__device__ T power_impl(const T &x, const T &e) {
        if constexpr (std::is_same_v<T, float>) {
            return powf(x, e);
        } else if constexpr (std::is_same_v<T, double>) {
            return pow(x, e);
        } else {
            static_assert(!std::is_same_v<T, T>, "unsupported data type");
            return static_cast<T>(0);
        }
    }

    /// @brief unary functor: y = x^N
    template<class T, int N>
    struct power_functor {
        __ocean__host__device__ T operator()(const T &x) const {
            return power_impl(x, static_cast<T>(N));
        }
    };

    /// @brief binary functor: dx = dy * N * x^(N-1)
    template<class T, int N>
    struct power_backward_functor {
        __ocean__host__device__ T operator()(const T &dy, const T &x) const {
            return dy * (N * power_impl(x, static_cast<T>(N - 1)));
        }
    };

    template<class T, int N>
    void power_op<T, N>::forward(DeviceTypeDef device,
                                 const tensor_ptr *ins, size_type in_num,
                                 tensor_ptr *outs, size_type out_num,
                                 const T *params) {
        auto &x = *ins[0];
        auto &y = *outs[0];
        auto n = x.numel(device);
        switch (device.type) {
            case EDevice::CPU:
                for (size_type i = 0; i < n; ++i)
                    y.pdata[i] = power_impl(x.pdata[i], static_cast<T>(N));
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(device.id);
                {
                    thrust::device_ptr<const T> dx(x.pdata);
                    thrust::device_ptr<T> dy(y.pdata);
                    thrust::transform(dx, dx + n, dy, power_functor<T, N>{});
                    break;
                }
#endif
            default:
                throw std::runtime_error("unknown device type");
        }
    }

    template<class T, int N>
    void power_op<T, N>::backward(DeviceTypeDef device,
                                  const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                                  const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                                  const T *params) {
        if (in_grads == nullptr || in_grads[0] == nullptr)
            return;
        auto &x = *ins[0];
        auto &dx = *in_grads[0];
        auto &dy = *out_grads[0];

        auto n = x.numel(device);
        switch (device.type) {
            case EDevice::CPU:
                for (size_type i = 0; i < n; ++i)
                    dx.pdata[i] = dy.pdata[i] * (N * power_impl(x.pdata[i], static_cast<T>(N - 1)));
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(device.id);
                {
                    thrust::device_ptr<const T> pdy(dy.pdata);
                    thrust::device_ptr<const T> px(x.pdata);
                    thrust::device_ptr<T> pdx(dx.pdata);
                    thrust::transform(pdy, pdy + n, px, pdx, power_backward_functor<T, N>{});
                    break;
                }
#endif
            default:
                throw std::runtime_error("unknown device type");
        }
    }

    /// tensor power end

    /* +-+-+-+-+-+-+-+-+-+- functions -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    tensor<T> exp(tensor<T> &&x) {
        if (!x.requires_grad()) {
            // inplace transformation
            exp_op<T> op;
            auto &xk = x.tensor_pointer()->kernel();
            auto xp = &xk;
            op.forward(x.device(), &xp, 1, &xp, 1, nullptr);
            return std::move(x);
        }
        // right value that requires grad is treated as left value
        return exp(x);
    }

    template<class T>
    tensor<T> exp(const tensor<T> &x) {
        tensor<T> y(x.size(), x.device());
        exp_op<T> op;
        auto &xk = x.tensor_pointer()->kernel();
        auto &yk = y.tensor_pointer()->kernel();
        auto xp = &xk;
        auto yp = &yk;
        op.forward(x.device(), &xp, 1, &yp, 1, nullptr);
        // automatic differentiation
        if (x.requires_grad()) {
            y.requires_grad(true);
            // create a new graph node
            // in: x
            // out: y
            auto graph = std::make_shared<ComputationalGraphNode<T> >();

            graph->in_pt = {x.tensor_pointer()};
            graph->in_pg = {x.grad_pointer()};
            graph->in_gen = {x.generator()};
            graph->in_need_grad = {true};

            graph->out_pt = {y.tensor_pointer()};
            graph->out_pg = {y.grad_pointer()};

            graph->op = std::make_unique<exp_op<T> >();

            y.generator(graph);
        }
        return y;
    }

    template<class T, int N>
    tensor<T> power(tensor<T> &&x) {
        if (!x.requires_grad()) {
            // inplace transformation
            power_op<T, N> op;
            auto &xk = x.tensor_pointer()->kernel();
            auto xp = &xk;
            op.forward(x.device(), &xp, 1, &xp, 1, nullptr);
            return std::move(x);
        }
        // right value that requires grad is treated as left value
        return power<T, N>(x);
    }

    template<class T, int N>
    tensor<T> power(const tensor<T> &x) {
        tensor<T> y(x.size(), x.device());
        power_op<T, N> op;
        auto &xk = x.tensor_pointer()->kernel();
        auto &yk = y.tensor_pointer()->kernel();
        auto xp = &xk;
        auto yp = &yk;
        op.forward(x.device(), &xp, 1, &yp, 1, nullptr);
        // automatic differentiation
        if (x.requires_grad()) {
            y.requires_grad(true);
            // create a new graph node
            // in: x
            // out: y
            auto graph = std::make_shared<ComputationalGraphNode<T> >();

            graph->in_pt = {x.tensor_pointer()};
            graph->in_pg = {x.grad_pointer()};
            graph->in_gen = {x.generator()};
            graph->in_need_grad = {true};

            graph->out_pt = {y.tensor_pointer()};
            graph->out_pg = {y.grad_pointer()};

            graph->op = std::make_unique<power_op<T, N> >();

            y.generator(graph);
        }
        return y;
    }
} // namespace ocean::nndl
