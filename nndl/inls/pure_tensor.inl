#pragma once

#include "../pure_tensor.h"

namespace ocean {
    /// pure_tensor begin

    /* +-+-+-+-+-+-+-+-+-+- constructors -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    pure_tensor<T>::pure_tensor(const shape_type &size, const DeviceTypeDef &device)
        : _device(device) {
        resize(size);
    }

    template<class T>
    template<class U>
    pure_tensor<T>::pure_tensor(const std::initializer_list<std::initializer_list<U> > &list) {
        auto first = list.begin();
        auto last = list.end();
        if (first == last) {
            clear();
            return;
        }
        // assert list is not empty
        auto r = std::distance(first, last);
        auto c = std::distance(first->begin(), first->end());
        for (auto &row: list)
            if (std::distance(row.begin(), row.end()) != c)
                throw std::invalid_argument("inconsistent row size");
        // assert all rows have the same size
        vector<T> list_copy;
        list_copy.reserve(r * c);
        for (auto &row: list)
#pragma warning(push)
#pragma warning(disable : 4244)
            list_copy.insert(list_copy.end(), row.begin(), row.end());
#pragma warning(pop)
        assign(list_copy.begin(), list_copy.end());
        reshape(r, c);
    }

    /* +-+-+-+-+-+-+-+-+-+- access -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    template<class... Args>
        requires(std::conjunction_v<std::is_integral<Args>...>)
    T &pure_tensor<T>::operator()(Args... args) {
        size_t idx = 0;
        size_t stride = numel();
        size_t i = 0;
        ((stride /= size(i++), idx += stride * args), ...);
        return data()[idx];
    }

    /* +-+-+-+-+-+-+-+-+-+- getters -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    pure_tensor<T>::size_type pure_tensor<T>::dim() const noexcept {
        if (_kernel.pdim == nullptr)
            return 0;
        switch (_device.type) {
            case EDevice::ANY:
            case EDevice::CPU:
                return *_kernel.pdim;
#if NNDL_USE_CUDA
            case EDevice::CUDA: {
                size_type n;
                cudaSetDevice(_device.id);
                cudaMemcpy(&n, _kernel.pdim, sizeof(size_type), cudaMemcpyDeviceToHost);
                return n;
            }
#endif
            default:
                return 0;
        }
    }

    template<class T>
    pure_tensor<T>::shape_type pure_tensor<T>::size() const noexcept {
        shape_type shape;
        if (_kernel.pdim == nullptr)
            return shape;
        size_type dim;
        switch (_device.type) {
            case EDevice::ANY:
            case EDevice::CPU:
                dim = *_kernel.pdim;
                shape.resize(dim + 2);
                shape[0] = dim;
                shape[1] = *_kernel.pnumel;
                for (size_type i = 0; i < dim; ++i)
                    shape[i + 2] = _kernel.pshape[i];
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(_device.id);
                cudaMemcpy(&dim, _kernel.pdim, sizeof(size_type), cudaMemcpyDeviceToHost);
                shape.resize(dim + 2);
                shape[0] = dim;
                cudaMemcpy(&shape[1], _kernel.pnumel, (dim + 1) * sizeof(size_type), cudaMemcpyDeviceToHost);
                break;
#endif
        }
        return shape;
    }

    template<class T>
    pure_tensor<T>::size_type pure_tensor<T>::size(size_type i) const {
        if (_kernel.pshape == nullptr)
            return 0;
        switch (_device.type) {
            case EDevice::ANY:
            case EDevice::CPU:
                return _kernel.pshape[i];
#if NNDL_USE_CUDA
            case EDevice::CUDA: {
                size_type n;
                cudaSetDevice(_device.id);
                cudaMemcpy(&n, _kernel.pshape + i, sizeof(size_type), cudaMemcpyDeviceToHost);
                return n;
            }
#endif
            default:
                return 0;
        }
    }

    template<class T>
    pure_tensor<T>::size_type pure_tensor<T>::numel() const {
        if (_kernel.pnumel == nullptr)
            return 0;
        size_type n;
        switch (_device.type) {
            case EDevice::CPU:
                return _kernel.pnumel ? *_kernel.pnumel : 0;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(_device.id);
                cudaMemcpy(&n, _kernel.pnumel, sizeof(size_type), cudaMemcpyDeviceToHost);
                return n;
#endif
            default:
                throw std::runtime_error("unknown device type");
        }
    }

    /* +-+-+-+-+-+-+-+-+-+- size & shape setters -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    void pure_tensor<T>::resize(const shape_type &shape) & {
        if (shape.empty()) {
            clear();
            return;
        }
        size_type cur_numel = numel();
        switch (_device.type) {
            case EDevice::CPU:
                delete[] _kernel.pdim;
                _kernel.pdim = new size_type[shape.size()];
                _kernel.pnumel = _kernel.pdim + 1;
                _kernel.pshape = _kernel.pnumel + 1;
                for (size_type i = 0; i < shape.size(); ++i)
                    _kernel.pdim[i] = shape[i];
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(_device.id);
                cudaFree(_kernel.pdim);
                cudaMalloc(&_kernel.pdim, shape.size() * sizeof(size_type));
                _kernel.pnumel = _kernel.pdim + 1;
                _kernel.pshape = _kernel.pnumel + 1;
                cudaMemcpy(_kernel.pdim, shape.data(), shape.size() * sizeof(size_type), cudaMemcpyHostToDevice);
                break;
#endif
            default:
                throw std::runtime_error("unknown device type");
        }
        if (cur_numel == shape[1])
            return;
        switch (_device.type) {
            case EDevice::CPU:
                delete[] _kernel.pdata;
                _kernel.pdata = new T[shape[1]];
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(_device.id);
                cudaFree(_kernel.pdata);
                cudaMalloc(&_kernel.pdata, shape[1] * sizeof(T));
                break;
#endif
            default:
                throw std::runtime_error("unknown device type");
        }
    }

    template<class T>
    template<class... Args>
        requires(std::conjunction_v<std::is_integral<Args>...>)
    void pure_tensor<T>::resize(Args... args) & {
        if (sizeof...(args) == 0)
            throw std::invalid_argument("at least one argument is required");
        vector<size_type> shape;
        shape.resize(sizeof...(args) + 2);
        shape[0] = sizeof...(args);
        shape[1] = 1;
        size_type i = 2;
        ((shape[i++] = args, shape[1] *= args), ...);
        resize(shape);
    }

    template<class T>
    void pure_tensor<T>::reshape(const shape_type &shape) & {
        if (shape.empty()) {
            clear();
            return;
        }
        // assert sizeof(data) > 0
        if (shape[1] != numel())
            throw std::invalid_argument("reshape: numel mismatch");
        // assert sizeof(data) == numel
        switch (_device.type) {
            case EDevice::CPU:
                delete[] _kernel.pdim;
                _kernel.pdim = new size_type[shape.size()];
                _kernel.pnumel = _kernel.pdim + 1;
                _kernel.pshape = _kernel.pnumel + 1;
                for (size_type i = 0; i < shape.size(); ++i)
                    _kernel.pdim[i] = shape[i];
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(_device.id);
                cudaFree(_kernel.pdim);
                cudaMalloc(&_kernel.pdim, shape.size() * sizeof(size_type));
                _kernel.pnumel = _kernel.pdim + 1;
                _kernel.pshape = _kernel.pnumel + 1;
                cudaMemcpy(_kernel.pdim, shape.data(), shape.size() * sizeof(size_type), cudaMemcpyHostToDevice);
                break;
#endif
            default:
                throw std::runtime_error("unknown device type");
        }
    }

    template<class T>
    template<class... Args>
        requires(std::conjunction_v<std::is_integral<Args>...>)
    void pure_tensor<T>::reshape(Args... args) & {
        if (sizeof...(args) == 0)
            throw std::invalid_argument("at least one argument is required");
        vector<size_type> shape;
        shape.resize(sizeof...(args) + 2);
        shape[0] = sizeof...(args);
        shape[1] = 1;
        size_type i = 2;
        ((shape[i++] = args, shape[1] *= args), ...);
        reshape(shape);
    }

    template<class T>
    void pure_tensor<T>::clear() & {
        if (empty())
            return;
        _kernel.clear(_device);
    }

    template<class T>
    template<class Iter>
    void pure_tensor<T>::assign(Iter first, Iter last) & {
        if (first == last) {
            clear();
            return;
        }
        // assert list is not empty
        resize(std::distance(first, last));
        switch (_device.type) {
            case EDevice::CPU:
#pragma warning(push)
#pragma warning(disable : 4244)
                std::copy(first, last, data());
#pragma warning(pop)
                break;
#if NNDL_USE_CUDA
            case EDevice::CUDA:
                cudaSetDevice(_device.id);
                cudaMemcpy(data(), &(*first), numel() * sizeof(T), cudaMemcpyHostToDevice);
                break;
#endif
            default:
                throw std::runtime_error("unknown device type");
        }
    }

    /* +-+-+-+-+-+-+-+-+-+- value setters -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    void pure_tensor<T>::fill(const T &value) & {
        nndl::assign(_device, data(), value, numel());
    }

    /* +-+-+-+-+-+-+-+-+-+- device interface -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    void pure_tensor<T>::to(const DeviceTypeDef &device) & {
        _kernel.move(_device, device);
        _device = device;
    }

    template<class T>
    void pure_tensor<T>::to(const string &device) & {
        if (device == "cpu")
            to(DeviceTypeDef{EDevice::CPU, 0});
        else if (device.starts_with("cuda")) {
            // case 1. "cuda"
            if (device.size() == 4)
                to(DeviceTypeDef{EDevice::CUDA, 0});
                // case 2. "cuda:x"
            else if (device.size() > 5) {
                size_t device_id;
                try {
                    device_id = std::stoul(device.substr(5));
                } catch (std::invalid_argument &) {
                    throw std::invalid_argument("invalid device id");
                }
                to(DeviceTypeDef{EDevice::CUDA, device_id});
            } else
                throw std::invalid_argument("invalid device name");
        }
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::cpu() && {
        DeviceTypeDef device{EDevice::CPU, 0};
        _kernel.move(_device, device);
        _device = device;
        return std::move(*this);
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::cpu() const & {
        pure_tensor t;
        t._device = {EDevice::CPU, 0};
        t._kernel = _kernel.copy(_device, t._device);
        return t;
    }

#if NNDL_USE_CUDA

    template<class T>
    pure_tensor<T> pure_tensor<T>::cuda(size_type device_id) && {
        DeviceTypeDef device{EDevice::CUDA, device_id};
        _kernel.move(_device, device);
        _device = device;
        return std::move(*this);
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::cuda(size_type device_id) const & {
        pure_tensor t;
        t._device = {EDevice::CUDA, device_id};
        t._kernel = _kernel.copy(t._device);
        return t;
    }

#endif

    /* +-+-+-+-+-+-+-+-+-+- copy ctrls -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    pure_tensor<T>::pure_tensor(const pure_tensor &other) {
        _device = other._device;
        _kernel = other._kernel.copy(_device, _device);
    }

    template<class T>
    pure_tensor<T>::pure_tensor(pure_tensor &&other) noexcept {
        _device = other._device;
        _kernel = std::move(other._kernel);
    }

    template<class T>
    pure_tensor<T> &pure_tensor<T>::operator=(const pure_tensor &other) {
        clear();
        _device = other._device;
        _kernel = other._kernel.copy(_device, _device);
        return *this;
    }

    template<class T>
    pure_tensor<T> &pure_tensor<T>::operator=(pure_tensor &&other) noexcept {
        clear();
        _device = other._device;
        _kernel = std::move(other._kernel);
        return *this;
    }

    /* +-+-+-+-+-+-+-+-+-+- tensor operators -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator+(const pure_tensor &t) && {
        if (device() != t.device())
            throw std::invalid_argument("tensor device mismatch");
        // assert same device
        if (empty() != t.empty())
            throw std::invalid_argument("tensor shape mismatch");
        if (empty())
            return pure_tensor();
        // assert both tensors are not empty
        if (size() != t.size())
            throw std::invalid_argument("tensor shape mismatch");
        // assert same shape
        add(_device, data(), t.data(), data(), numel());
        return std::move(*this);
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator-(const pure_tensor &t) && {
        if (device() != t.device())
            throw std::invalid_argument("tensor device mismatch");
        // assert same device
        if (empty() != t.empty())
            throw std::invalid_argument("tensor shape mismatch");
        if (empty())
            return pure_tensor();
        // assert both tensors are not empty
        if (size() != t.size())
            throw std::invalid_argument("tensor shape mismatch");
        // assert same shape
        sub(_device, data(), t.data(), data(), numel());
        return std::move(*this);
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator-(pure_tensor &&t) const & {
        if (device() != t.device())
            throw std::invalid_argument("tensor device mismatch");
        // assert same device
        if (empty() != t.empty())
            throw std::invalid_argument("tensor shape mismatch");
        if (empty())
            return pure_tensor();
        // assert both tensors are not empty
        if (size() != t.size())
            throw std::invalid_argument("tensor shape mismatch");
        // assert same shape
        sub(_device, t.data(), data(), t.data(), numel());
        return std::move(t);
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator*(const pure_tensor &t) const & {
        if (device() != t.device())
            throw std::invalid_argument("tensor device mismatch");
        // assert same device
        if (empty() || t.empty())
            throw std::invalid_argument("matmul with empty tensor");
        // assert both tensors are not empty
        size_type ra, rb, ca, cb;
        if (dim() == 1) {
            ra = 1;
            ca = size(0);
        } else if (dim() == 2) {
            ra = size(0);
            ca = size(1);
        } else
            throw std::invalid_argument("matrix dimension mast be 1 or 2");
        if (t.dim() == 1) {
            rb = 1;
            cb = t.size(0);
        } else if (t.dim() == 2) {
            rb = t.size(0);
            cb = t.size(1);
        } else {
            throw std::invalid_argument("matrix dimension mast be 1 or 2");
        }
        if (ca != rb) {
            throw std::invalid_argument("matmul shape mismatch");
        }
        // assert matmul shape match
        pure_tensor result;
        result.to(_device);
        result.resize(ra, cb);
        matmul(_device, data(), t.data(), result.data(), ra, ca, cb, false, false);
        // debug: vecotr * matrix = vector
        if (dim() == 1)
            result.reshape(cb);
        return result;
    }

    /* +-+-+-+-+-+-+-+-+-+- scalar operators -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator-() && {
        neg(_device, data(), data(), numel());
        return std::move(*this);
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator-() const & {
        pure_tensor t(this->size());
        neg(_device, t.data(), data(), numel());
        return t;
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator+=(const T &s) && {
        add(_device, data(), s, data(), numel());
        return std::move(*this);
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator+=(const T &s) const & {
        pure_tensor t(*this);
        add(_device, t.data(), s, t.data(), numel());
        return t;
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator+(const T &s) && {
        return std::move(*this) += s;
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator+(const T &s) const & {
        return *this += s;
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator*=(const T &s) && {
        scalar_mul(_device, data(), data(), s, numel());
        return std::move(*this);
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator*=(const T &s) const & {
        pure_tensor t(*this);
        scalar_mul(_device, t.data(), t.data(), s, numel());
        return t;
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator*(const T &s) && {
        return std::move(*this) *= s;
    }

    template<class T>
    pure_tensor<T> pure_tensor<T>::operator*(const T &s) const & {
        return pure_tensor(*this) *= s;
    }

    /* +-+-+-+-+-+-+-+-+-+- friends -+-+-+-+-+-+-+-+-+-+ */

    template<class U>
    ostream &operator<<(ostream &os, const pure_tensor<U> &t) {
        if (t._device.type != EDevice::CPU) {
            os << t.cpu();
            return os;
        }
        // assert device.type is CPU
        using size_type = pure_tensor<U>::size_type;

        auto print_vec = [&](const U *p, size_type n) {
            os << p[0];
            for (size_type i = 1; i < n; ++i)
                os << ' ' << p[i];
        };

        auto print_mat = [&](const U *p, size_type r, size_type c) {
            print_vec(p, c);
            for (size_type i = 1; i < r; ++i) {
                os << std::endl;
                p += c;
                print_vec(p, c);
            }
        };

        switch (t.dim()) {
            case 0: // empty
                os << "[]";
                break;
            case 1: // vector
                print_vec(t.data(), t.size(0));
                break;
            case 2: // matrix
                print_mat(t.data(), t.size(0), t.size(1));
                break;
            default: // tensor with more than 2 dimensions
                /// TODO: insert code here
                break;
        }

        return os;
    }

    /// pure_tensor end

    /* +-+-+-+-+-+-+-+-+-+- logic -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    bool equal(const pure_tensor<T> &a, const pure_tensor<T> &b) {
        if (a.empty() != b.empty())
            return false;
        if (a.empty())
            return true;
        // assert both a and b are not empty
        if (a.dim() != b.dim())
            return false;
        // assert same dimension
        if (a.size() != b.size())
            return false;
        // assert same shape
        auto n = a.numel();
        for (decltype(n) i = 0; i < n; ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }
} // namespace ocean
