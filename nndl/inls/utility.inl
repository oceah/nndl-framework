#pragma once

#include "../utility.h"

namespace ocean::nndl {
    /// DeviceTypeDef begin

    inline std::ostream &operator<<(std::ostream &os, const DeviceTypeDef &d) {
        switch (d.type) {
            case EDevice::CPU:
                os << "cpu";
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                os << "cuda:" << d.id;
                break;
#endif
            default:
                os << "invalid device '" << static_cast<int8_t>(d.type) << '\'';
                break;
        }
        return os;
    }

    /// DeviceTypeDef end

    /// tensor_kernel begin

    /* +-+-+-+-+-+-+-+-+-+- constructors -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    __ocean__host__device__
    tensor_kernel<T>::tensor_kernel() {
        pdim = nullptr;
        pnumel = nullptr;
        pshape = nullptr;
        pdata = nullptr;
    }

    template<class T>
    __ocean__device__
    tensor_kernel<T>::tensor_kernel(shape_type _pshape, T *_pdata) {
        if (_pshape == nullptr) {
            pdim = nullptr;
            pnumel = nullptr;
            pshape = nullptr;
            pdata = nullptr;
            return;
        }
        // assert _pshape is not null
        pdim = _pshape++;
        pnumel = _pshape++;
        pshape = _pshape;
        pdata = _pdata;
    }

    /* +-+-+-+-+-+-+-+-+-+- getters -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    __ocean__host__
    tensor_kernel<T>::size_type
    tensor_kernel<T>::dim(const DeviceTypeDef &device) const {
        if (pdim == nullptr)
            return 0;
        // assert pdim is not null
        size_type n;
        switch (device.type) {
            case EDevice::CPU:
                return *pdim;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(device.id);
                cudaMemcpy(&n, pdim, sizeof(size_type), cudaMemcpyDeviceToHost);
                return n;
#endif
            default:
                throw std::invalid_argument("unsupported device");
        }
    }

    template<class T>
    __ocean__host__
    tensor_kernel<T>::size_type
    tensor_kernel<T>::numel(const DeviceTypeDef &device) const {
        if (pnumel == nullptr)
            return 0;
        // assert pnumel is not null
        size_type n;
        switch (device.type) {
            case EDevice::CPU:
                return *pnumel;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(device.id);
                cudaMemcpy(&n, pnumel, sizeof(size_type), cudaMemcpyDeviceToHost);
                return n;
#endif
            default:
                throw std::invalid_argument("unsupported device");
        }
    }

    template<class T>
    __ocean__host__
    tensor_kernel<T>::size_type
    tensor_kernel<T>::size(const DeviceTypeDef &device, size_type i) const {
        if (pshape == nullptr)
            return 0;
        // assert pshape is not null
        size_type n;
        switch (device.type) {
            case EDevice::CPU:
                return pshape[i];
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(device.id);
                cudaMemcpy(&n, pshape + i, sizeof(size_type), cudaMemcpyDeviceToHost);
                return n;
#endif
            default:
                throw std::invalid_argument("unsupported device");
        }
    }

    /* +-+-+-+-+-+-+-+-+-+- interfaces -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    __ocean__host__ void tensor_kernel<T>::clear(const DeviceTypeDef &device) {
        if (pdim == nullptr)
            return;
        switch (device.type) {
            case EDevice::CPU:
                delete[] pdim;
                delete[] pdata;
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaFree(pdim);
                cudaFree(pdata);
                break;
#endif
            default:
                throw std::invalid_argument("unsupported device");
        }
        pdim = nullptr;
        pnumel = nullptr;
        pshape = nullptr;
        pdata = nullptr;
    }

    template<class T>
    __ocean__host__ void tensor_kernel<T>::move(const DeviceTypeDef &src, const DeviceTypeDef &dst) {
        if (src == dst)
            return;
        if (pdim == nullptr)
            return;

        void *_pshape, *_pdata;
        size_type dim, numel;

        switch (src.type) {
            case EDevice::CPU:
                dim = *pdim;
                numel = *pnumel;
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(src.id);
                cudaMemcpy(&dim, pdim, sizeof(size_type), cudaMemcpyDeviceToHost);
                cudaMemcpy(&numel, pnumel, sizeof(size_type), cudaMemcpyDeviceToHost);
                break;
#endif
            default:
                throw std::invalid_argument("unsupported device");
        }

        switch (src.type) {
            case EDevice::CPU:
                // from cpu to ...
                switch (dst.type) {
                    case EDevice::CPU:
                        // from cpu:x to cpu:y
                        throw std::invalid_argument("multiple cpus not supported");
                        break;
#if NNDL_USE_CUDA
                    case EDevice::CUDA:
                        // from cpu to cuda
                        cudaSetDevice(dst.id);
                        // copy data
                        cudaMalloc(&_pshape, (dim + 2) * sizeof(size_type));
                        cudaMemcpy(_pshape, pdim, (dim + 2) * sizeof(size_type), cudaMemcpyHostToDevice);
                        cudaMalloc(&_pdata, numel * sizeof(T));
                        cudaMemcpy(_pdata, pdata, numel * sizeof(T), cudaMemcpyHostToDevice);
                        // clear data
                        delete[] pdim;
                        delete[] pdata;
                        // update data
                        pdim = static_cast<size_type *>(_pshape);
                        pnumel = pdim + 1;
                        pshape = pnumel + 1;
                        pdata = static_cast<T *>(_pdata);
                        break;
#endif
                    default:
                        throw std::invalid_argument("unsupported device");
                }
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                // from cuda to ...
                switch (dst.type) {
                    case EDevice::CPU:
                        // from cuda to cpu
                        cudaSetDevice(src.id);
                        // copy data
                        _pshape = new size_type[dim + 2];
                        cudaMemcpy(_pshape, pdim, (dim + 2) * sizeof(size_type), cudaMemcpyDeviceToHost);
                        _pdata = new T[numel];
                        cudaMemcpy(_pdata, pdata, numel * sizeof(T), cudaMemcpyDeviceToHost);
                        // clear data
                        cudaFree(pdim);
                        cudaFree(pdata);
                        // update data
                        pdim = static_cast<size_type *>(_pshape);
                        pnumel = pdim + 1;
                        pshape = pnumel + 1;
                        pdata = static_cast<T *>(_pdata);
                        break;
                    case EDevice::CUDA:
                        // from cuda:x to cuda:y
                        // 1. copy data from cuda:x to cpu
                        cudaSetDevice(src.id);
                        // copy data
                        _pshape = new size_type[dim + 2];
                        cudaMemcpy(_pshape, pdim, (dim + 2) * sizeof(size_type), cudaMemcpyDeviceToHost);
                        _pdata = new T[numel];
                        cudaMemcpy(_pdata, pdata, numel * sizeof(T), cudaMemcpyDeviceToHost);
                        // clear data
                        cudaFree(pdim);
                        cudaFree(pdata);
                        // 2. copy data from cpu to cuda:y
                        cudaSetDevice(dst.id);
                        // copy data
                        cudaMalloc(reinterpret_cast<void **>(&pdim), (dim + 2) * sizeof(size_type));
                        cudaMemcpy(pdim, _pshape, (dim + 2) * sizeof(size_type), cudaMemcpyHostToDevice);
                        cudaMalloc(reinterpret_cast<void **>(&pdata), numel * sizeof(T));
                        cudaMemcpy(pdata, _pdata, numel * sizeof(T), cudaMemcpyHostToDevice);
                        // clear data
                        delete[] static_cast<size_type *>(_pshape);
                        delete[] static_cast<T *>(_pdata);
                        // update data
                        pnumel = pdim + 1;
                        pshape = pnumel + 1;
                        break;
                    default:
                        throw std::runtime_error("unsupported device type");
                }
                break;
#endif
            default:
                throw std::runtime_error("unsupported device type");
        }
    }

    template<class T>
    __ocean__host__
    tensor_kernel<T>
    tensor_kernel<T>::copy(const DeviceTypeDef &src, const DeviceTypeDef &dst) const {
        tensor_kernel<T> t;
        if (pdim == nullptr)
            return t;

        void *_pshape, *_pdata;
        size_type dim, numel;

        switch (src.type) {
            case EDevice::CPU:
                dim = *pdim;
                numel = *pnumel;
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(src.id);
                cudaMemcpy(&dim, pdim, sizeof(size_type), cudaMemcpyDeviceToHost);
                cudaMemcpy(&numel, pnumel, sizeof(size_type), cudaMemcpyDeviceToHost);
                break;
#endif
            default:
                throw std::runtime_error("unsupported device type");
        }

        switch (src.type) {
            case EDevice::CPU:
                // from cpu to ...
                switch (dst.type) {
                    case EDevice::CPU:
                        // from cpu to cpu
                        // copy data
                        _pshape = new size_type[dim + 2];
                        memcpy(_pshape, pdim, (dim + 2) * sizeof(size_type));
                        _pdata = new T[numel];
                        memcpy(_pdata, pdata, numel * sizeof(T));
                        // update data
                        t.pdim = static_cast<size_type *>(_pshape);
                        t.pnumel = t.pdim + 1;
                        t.pshape = t.pnumel + 1;
                        t.pdata = static_cast<T *>(_pdata);
                        break;
#if NNDL_USE_CUDA
                    case EDevice::CUDA:
                        // from cpu to cuda
                        cudaSetDevice(dst.id);
                        // copy data
                        cudaMalloc(&_pshape, (dim + 2) * sizeof(size_type));
                        cudaMemcpy(_pshape, pdim, (dim + 2) * sizeof(size_type), cudaMemcpyHostToDevice);
                        cudaMalloc(&_pdata, numel * sizeof(T));
                        cudaMemcpy(_pdata, pdata, numel * sizeof(T), cudaMemcpyHostToDevice);
                        // update data
                        t.pdim = static_cast<size_type *>(_pshape);
                        t.pnumel = t.pdim + 1;
                        t.pshape = t.pnumel + 1;
                        t.pdata = static_cast<T *>(_pdata);
                        break;
#endif
                    default:
                        throw std::runtime_error("unsupported device type");
                }
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                // from cuda to ...
                switch (dst.type) {
                    case EDevice::CPU:
                        // from cuda to cpu
                        // copy data
                        _pshape = new size_type[dim + 2];
                        cudaMemcpy(_pshape, pdim, (dim + 2) * sizeof(size_type), cudaMemcpyDeviceToHost);
                        _pdata = new T[numel];
                        cudaMemcpy(_pdata, pdata, numel * sizeof(T), cudaMemcpyDeviceToHost);
                        // update data
                        t.pdim = static_cast<size_type *>(_pshape);
                        t.pnumel = t.pdim + 1;
                        t.pshape = t.pnumel + 1;
                        t.pdata = static_cast<T *>(_pdata);
                        break;
                    case EDevice::CUDA:
                        // from cuda:x to cuda:y
                        // 1. copy data from cuda:x to cpu
                        cudaSetDevice(src.id);
                        // copy data
                        _pshape = new size_type[dim + 2];
                        cudaMemcpy(_pshape, pdim, (dim + 2) * sizeof(size_type), cudaMemcpyDeviceToHost);
                        _pdata = new T[numel];
                        cudaMemcpy(_pdata, pdata, numel * sizeof(T), cudaMemcpyDeviceToHost);
                        // 2. copy data from cpu to cuda:y
                        cudaSetDevice(dst.id);
                        // copy data
                        cudaMalloc(reinterpret_cast<void **>(&t.pdim), (dim + 2) * sizeof(size_type));
                        cudaMemcpy(t.pdim, _pshape, (dim + 2) * sizeof(size_type), cudaMemcpyHostToDevice);
                        cudaMalloc(reinterpret_cast<void **>(&t.pdata), numel * sizeof(T));
                        cudaMemcpy(t.pdata, _pdata, numel * sizeof(T), cudaMemcpyHostToDevice);
                        // clear data
                        delete[] static_cast<size_type *>(_pshape);
                        delete[] static_cast<T *>(_pdata);
                        // update data
                        t.pnumel = t.pdim + 1;
                        t.pshape = t.pnumel + 1;
                        break;
                    default:
                        throw std::runtime_error("unsupported device type");
                }
                break;
#endif
            default:
                throw std::runtime_error("unsupported device type");
        }

        return t;
    }

    /* +-+-+-+-+-+-+-+-+-+- copy ctrls -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    __ocean__host__device__
    tensor_kernel<T>::tensor_kernel(tensor_kernel &&other) noexcept {
        pdim = other.pdim;
        pnumel = other.pnumel;
        pshape = other.pshape;
        pdata = other.pdata;
        other.pdim = nullptr;
        other.pnumel = nullptr;
        other.pshape = nullptr;
        other.pdata = nullptr;
    }

    template<class T>
    __ocean__host__device__
    tensor_kernel<T> &
    tensor_kernel<T>::operator=(tensor_kernel &&other) noexcept {
        if (this == &other)
            return *this;
        // assert different objects
        // assert already cleared
        pdim = other.pdim;
        pnumel = other.pnumel;
        pshape = other.pshape;
        pdata = other.pdata;
        other.pdim = nullptr;
        other.pnumel = nullptr;
        other.pshape = nullptr;
        other.pdata = nullptr;
        return *this;
    }

    /// tensor_kernel end
} // namespace ocean::nndl
