#pragma once

#include "../operator.h"
#include "nndl/operator_cpu.h"

#if NNDL_USE_CUDA

#include "../operator_cuda.h"

#include <thrust/transform.h>

#endif

namespace ocean::nndl {
    /* +-+-+-+-+-+-+-+-+-+- basic -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    void fill(const DeviceTypeDef device, T *dst, T v, size_t size) {
        switch (device.type) {
            case EDevice::CPU:
                fill_cpu(dst, v, size);
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(static_cast<int>(device.id));
                fill_cuda(dst, v, size);
                break;
#endif
            default:
                throw std::runtime_error("unsupported device type");
        }
    }

    template<class T>
    void assign(const DeviceTypeDef device, T *dst, const T *src, size_t size) {
        switch (device.type) {
            case EDevice::CPU:
                assign_cpu(dst, src, size);
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(static_cast<int>(device.id));
                assign_cuda(dst, src, size);
                break;
#endif
            default:
                throw std::runtime_error("unsupported device type");
        }
    }

    template<class T>
    void assign(DeviceTypeDef device, T *dst, const T &src, size_t size) {
        switch (device.type) {
            case EDevice::CPU:
                assign_cpu(dst, src, size);
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(device.id);
                assign_cuda(dst, src, size);
                break;
#endif
            default:
                throw std::runtime_error("unsupported device type");
        }
    }

    template<class T>
    void neg(DeviceTypeDef device, T *dst, const T *src, size_t size) {
        switch (device.type) {
            case EDevice::CPU:
                neg_cpu(dst, src, size);
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(device.id);
                neg_cuda(dst, src, size);
                break;
#endif
            default:
                throw std::runtime_error("unsupported device type");
        }
    }

    template<class T>
    void add(DeviceTypeDef device, const T *a, const T *b, T *c, size_t size) {
        switch (device.type) {
            case EDevice::CPU:
                add_cpu(a, b, c, size);
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(device.id);
                add_cuda(a, b, c, size);
                break;
#endif
            default:
                throw std::runtime_error("unsupported device type");
        }
    }

    template<class T>
    void add(DeviceTypeDef device, const T *a, const T &s, T *b, size_t size) {
        switch (device.type) {
            case EDevice::CPU:
                add_cpu(a, b, s, size);
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(device.id);
                add_cuda(a, b, s, size);
                break;
#endif
            default:
                throw std::runtime_error("unsupported device type");
        }
    }

    template<class T>
    void sub(DeviceTypeDef device, const T *a, const T *b, T *c, size_t size) {
        switch (device.type) {
            case EDevice::CPU:
                sub_cpu(a, b, c, size);
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(device.id);
                sub_cuda(a, b, c, size);
                break;
#endif
            default:
                throw std::runtime_error("unsupported device type");
        }
    }

    template<class T>
    void scalar_mul(DeviceTypeDef device, const T *a, T *b, const T &s, size_t size) {
        switch (device.type) {
            case EDevice::CPU:
                scalar_mul_cpu(a, b, s, size);
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(device.id);
                scalar_mul_cuda(a, b, s, size);
                break;
#endif
            default:
                throw std::runtime_error("unsupported device type");
        }
    }

    template<class T>
    void matmul(DeviceTypeDef device, const T *a, const T *b, T *c,
                size_t m, size_t k, size_t n,
                bool trans_a, bool trans_b) {
        switch (device.type) {
            case EDevice::CPU:
                matmul_cpu(a, b, c, m, k, n, trans_a, trans_b);
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(device.id);
                matmul_cuda(a, b, c, m, k, n, trans_a, trans_b);
                break;
#endif
            default:
                throw std::runtime_error("unsupported device type");
        }
    }

    /* +-+-+-+-+-+-+-+-+-+- unary scalar function -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    void transform(DeviceTypeDef device, const function<T(const T &)> &f,
                   const T *x, T *y, size_t size) {
        switch (device.type) {
            case EDevice::CPU:
                while (size--)
                    *y++ = f(*x++);
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                throw std::runtime_error("insert code here");
#endif
            default:
                throw std::runtime_error("unsupported device type");
        }
    }

    /* +-+-+-+-+-+-+-+-+-+- basic operators -+-+-+-+-+-+-+-+-+-+ */

    /// assign_op begin

    template<class T>
    void assign_op<T>::forward(DeviceTypeDef device,
                               const tensor_ptr *ins, size_type in_num,
                               tensor_ptr *outs, size_type out_num,
                               const T *params) {
        auto &in = *ins[0];
        auto &out = *outs[0];
        assign(device, out.pdata, in.pdata, in.numel(device));
    }

    template<class T>
    void assign_op<T>::backward(DeviceTypeDef device,
                                const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                                const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                                const T *params) {
        if (in_grads == nullptr || in_grads[0] == nullptr)
            return;
        auto &out_grad = *out_grads[0];
        auto &in_grad = *in_grads[0];
        assign(device, in_grad.pdata, out_grad.pdata, out_grad.numel(device));
    }

    /// assign_op end

    /// neg_op begin

    template<class T>
    void neg_op<T>::forward(DeviceTypeDef device,
                            const tensor_ptr *ins, size_type in_num,
                            tensor_ptr *outs, size_type out_num,
                            const T *params) {
        auto &in = *ins[0];
        auto &out = *outs[0];
        neg(device, out.pdata, in.pdata, in.numel(device));
    }

    template<class T>
    void neg_op<T>::backward(DeviceTypeDef device,
                             const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                             const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                             const T *params) {
        if (in_grads == nullptr || in_grads[0] == nullptr)
            return;
        auto &out_grad = *out_grads[0];
        auto &in_grad = *in_grads[0];
        neg(device, in_grad.pdata, out_grad.pdata, out_grad.numel(device));
    }

    /// neg_op end

    /// add_op begin

    template<class T>
    void add_op<T>::forward(DeviceTypeDef device,
                            const tensor_ptr *ins, size_type in_num,
                            tensor_ptr *outs, size_type out_num,
                            const T *params) {
        // assert in_num >= 2
        if (in_num < 2)
            throw std::invalid_argument("assert failed: in_num >= 2");
        auto &out = *outs[0];
        auto n = ins[0]->numel(device);
        add(device, ins[0]->pdata, ins[1]->pdata, out.pdata, n);
        for (size_type i = 2; i < in_num; ++i)
            add(device, out.pdata, ins[i]->pdata, out.pdata, n);
    }

    template<class T>
    void add_op<T>::backward(DeviceTypeDef device,
                             const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                             const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                             const T *params) {
        // assert in_num >= 2
        if (in_num < 2)
            throw std::invalid_argument("assert failed: in_num >= 2");
        if (in_grads == nullptr)
            return;
        auto &gs = *out_grads[0];
        auto n = ins[0]->numel(device);
        for (size_type i = 0; i < in_num; ++i)
            if (in_grads[i] != nullptr)
                assign(device, in_grads[i]->pdata, gs.pdata, n);
    }

    /// add_op end

    /// sub_op begin

    template<class T>
    void sub_op<T>::forward(DeviceTypeDef device,
                            const tensor_ptr *ins, size_type in_num,
                            tensor_ptr *outs, size_type out_num,
                            const T *params) {
        // c = a - b
        auto &a = *ins[0];
        auto &b = *ins[1];
        auto &c = *outs[0];
        sub(device, a.pdata, b.pdata, c.pdata, a.numel(device));
    }

    template<class T>
    void sub_op<T>::backward(DeviceTypeDef device,
                             const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                             const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                             const T *params) {
        if (in_grads == nullptr)
            return;
        // c = a - b
        auto &gc = *out_grads[0];
        auto n = gc.numel(device);
        if (in_grads[0] != nullptr) {
            auto &ga = *in_grads[0];
            assign(device, ga.pdata, gc.pdata, n);
        }
        if (in_grads[1] != nullptr) {
            auto &gb = *in_grads[1];
            neg(device, gb.pdata, gc.pdata, n);
        }
    }

    /// sub_op end

    /// matmul_op begin

    template<class T>
    void matmul_op<T>::forward(DeviceTypeDef device,
                               const tensor_ptr *ins, size_type in_num,
                               tensor_ptr *outs, size_type out_num,
                               const T *params) {
        // c = a * b
        auto &a = *ins[0];
        auto &b = *ins[1];
        auto &c = *outs[0];
        size_type m, k, n;
        if (a.dim(device) == 1) {
            m = 1;
            k = a.size(device, 0);
        } else {
            m = a.size(device, 0);
            k = a.size(device, 1);
        }
        n = b.dim(device) == 1 ? b.size(device, 0) : b.size(device, 1);
        matmul(device, a.pdata, b.pdata, c.pdata, m, k, n, false, false);
    }

    template<class T>
    void matmul_op<T>::backward(DeviceTypeDef device,
                                const tensor_ptr *outs, const tensor_ptr *out_grads, size_type out_num,
                                const tensor_ptr *ins, tensor_ptr *in_grads, size_type in_num,
                                const T *params) {
        if (in_grads == nullptr)
            return;
        // c = a * b
        auto &a = *ins[0];
        auto &b = *ins[1];
        auto &c = *outs[0];
        auto &ga = *in_grads[0];
        auto &gb = *in_grads[1];
        auto &gc = *out_grads[0];
        size_type m, k, n;
        if (a.dim(device) == 1) {
            m = 1;
            k = a.size(device, 0);
        } else {
            m = a.size(device, 0);
            k = a.size(device, 1);
        }
        n = b.dim(device) == 1 ? b.size(device, 0) : b.size(device, 1);
        if (in_grads[0] != nullptr)
            matmul(device, gc.pdata, b.pdata, ga.pdata, m, n, k, false, true);
        if (in_grads[1] != nullptr)
            matmul(device, a.pdata, gc.pdata, gb.pdata, k, m, n, true, false);
    }

    /// matmul_op end
} // namespace ocean::nndl
