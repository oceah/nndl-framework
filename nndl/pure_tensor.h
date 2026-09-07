/**
******************************************************************************
* @file             pure_tensor.h
* @brief            tensor without autograd
* @author           OceanH
* @create           2024-11-04
* @latestupdate     2025-02-19
******************************************************************************
*/

#ifndef PURE_TENSOR_H
#define PURE_TENSOR_H

#include "utility.h"
#include "operator.h"

using namespace ocean::nndl;

#include <string>
#include <vector>

using std::ostream;
using std::string;
using std::vector;

namespace ocean {
    /// @brief tensor without autograd
    template<class T>
    class pure_tensor {
    public:
        using size_type = tensor_kernel<T>::size_type;
        using shape_type = vector<size_type>;

        /* +-+-+-+-+-+-+-+-+-+- constructors -+-+-+-+-+-+-+-+-+-+ */

        pure_tensor() = default;

        /// @brief create a tensor with specific shape
        template<class... Args>
            requires(std::conjunction_v<std::is_integral<Args>...>)
        explicit pure_tensor(Args... args) {
            resize(args...);
        }

        explicit pure_tensor(const shape_type &size, const DeviceTypeDef &device = {EDevice::CPU, 0});

        /// @brief create a vector with specific shape and fill with specific value
        template<class U>
        pure_tensor(const std::initializer_list<U> &list) { assign(list.begin(), list.end()); }

        /// @brief create a matrix with specific shape and fill with specific value
        template<class U>
        pure_tensor(const std::initializer_list<std::initializer_list<U> > &list);

        /* +-+-+-+-+-+-+-+-+-+- access -+-+-+-+-+-+-+-+-+-+ */

        /// @brief access element by page, row and column
        template<class... Args>
            requires(std::conjunction_v<std::is_integral<Args>...>)
        T &operator()(Args... args);

        template<typename... Args>
            requires(std::conjunction_v<std::is_integral<Args>...>)
        const T &operator()(Args... args) const {
            return const_cast<pure_tensor *>(this)->operator()(std::forward<Args>(args)...);
        }

        /// @brief access element by index
        T &operator[](size_type i) { return _kernel.pdata[i]; }

        const T &operator[](size_type i) const { return const_cast<pure_tensor *>(this)->operator[](i); }

        /* +-+-+-+-+-+-+-+-+-+- getters -+-+-+-+-+-+-+-+-+-+ */

        /// @brief kernel getter
        [[nodiscard]] tensor_kernel<T> &kernel() noexcept { return _kernel; }

        [[nodiscard]] const tensor_kernel<T> &kernel() const noexcept {
            return const_cast<pure_tensor<T> *>(this)->kernel();
        }

        /// @brief data pointer
        [[nodiscard]] T *data() noexcept { return _kernel.pdata; }

        [[nodiscard]] const T *data() const noexcept {
            return const_cast<pure_tensor<T> *>(this)->data();
        }

        /// @brief order of tensor
        [[nodiscard]] size_type dim() const noexcept;

        /// @brief shape of tensor
        [[nodiscard]] shape_type size() const noexcept;

        /// @brief specific size of tensor
        [[nodiscard]] size_type size(size_type i) const;

        /// @brief number of elements
        [[nodiscard]] size_type numel() const;

        /// @brief if tensor is empty
        [[nodiscard]] bool empty() const noexcept { return numel() == 0; }

        /// @brief where tensor is stored
        [[nodiscard]] DeviceTypeDef device() const noexcept { return _device; }

        /* +-+-+-+-+-+-+-+-+-+- size & shape setters -+-+-+-+-+-+-+-+-+-+ */

        /// @brief modify shape of tensor
        /// @note accessing tensor after resize is undefined
        /// @note if you want to reshape with equal capacity, please use reshape
        void resize(const shape_type &shape) &;

        template<class... Args>
            requires(std::conjunction_v<std::is_integral<Args>...>)
        void resize(Args... args) &;

        /// @brief reshape with equal capacity
        /// @note keep the total number of elements and the order of elements in row-major order unchanged
        /// @note if the total number of elements changes, an exception will be thrown
        void reshape(const shape_type &shape) &;

        template<class... Args>
            requires(std::conjunction_v<std::is_integral<Args>...>)
        void reshape(Args... args) &;

        /// @brief clear tensor
        /// @note usually not necessary manually
        void clear() &;

        /// @brief assign data to vector
        template<class Iter>
        void assign(Iter first, Iter last) &;

        /* +-+-+-+-+-+-+-+-+-+- value setters -+-+-+-+-+-+-+-+-+-+ */

        /// @brief fill tensor with specific value
        void fill(const T &value) &;

        /* +-+-+-+-+-+-+-+-+-+- device interfaces -+-+-+-+-+-+-+-+-+-+ */

        /// @brief move tensor to specific device
        void to(const DeviceTypeDef &device) &;

        void to(const string &device) &;

        /// @brief create a copy of tensor on CPU
        pure_tensor cpu() &&;

        pure_tensor cpu() const &;

#if NNDL_USE_CUDA

        /// @brief create a copy of tensor on CUDA
        pure_tensor cuda(size_type device_id = 0) &&;

        pure_tensor cuda(size_type device_id = 0) const &;

#endif

        /* +-+-+-+-+-+-+-+-+-+- copy ctrls -+-+-+-+-+-+-+-+-+-+ */

        pure_tensor(const pure_tensor &other);

        pure_tensor(pure_tensor &&other) noexcept;

        pure_tensor &operator=(const pure_tensor &other);

        pure_tensor &operator=(pure_tensor &&other) noexcept;

        ~pure_tensor() { clear(); }

        /* +-+-+-+-+-+-+-+-+-+- tensor operators -+-+-+-+-+-+-+-+-+-+ */

        pure_tensor operator+(const pure_tensor &t) &&;

        pure_tensor operator+(pure_tensor &&t) && { return std::move(*this) + t; }

        pure_tensor operator+(pure_tensor &&t) const & { return std::move(t) + *this; }

        pure_tensor operator+(const pure_tensor &t) const & { return pure_tensor(*this) + t; }

        pure_tensor &operator+=(const pure_tensor &t) { return *this = std::move(*this) + t; }

        pure_tensor operator-(const pure_tensor &t) &&;

        pure_tensor operator-(pure_tensor &&t) && { return std::move(*this) - t; }

        pure_tensor operator-(pure_tensor &&t) const &;

        pure_tensor operator-(const pure_tensor &t) const & { return pure_tensor(*this) - t; }

        pure_tensor &operator-=(const pure_tensor &t) { return *this = std::move(*this) - t; }

        // pure_tensor operator*(const pure_tensor &t) &&;

        pure_tensor operator*(const pure_tensor &t) const &;

        // pure_tensor &operator*=(const pure_tensor &t) &&;

        // pure_tensor &operator*=(const pure_tensor &t) const &;

        /* +-+-+-+-+-+-+-+-+-+- scalar operators -+-+-+-+-+-+-+-+-+-+ */

        pure_tensor operator-() &&;

        pure_tensor operator-() const &;

        pure_tensor operator+=(const T &s) &&;

        pure_tensor operator+=(const T &s) const &;

        pure_tensor operator+(const T &s) &&;

        pure_tensor operator+(const T &s) const &;

        pure_tensor operator*(const T &s) const &;

        pure_tensor operator*(const T &s) &&;

        pure_tensor operator*=(const T &s) &&;

        pure_tensor operator*=(const T &s) const &;

        /* +-+-+-+-+-+-+-+-+-+- iterator -+-+-+-+-+-+-+-+-+-+ */

        [[nodiscard]] T *begin() noexcept { return _kernel.pdata; }

        [[nodiscard]] const T *begin() const noexcept { return const_cast<pure_tensor<T> *>(this)->begin(); }

        [[nodiscard]] T *end() noexcept { return _kernel.pdata + numel(); }

        [[nodiscard]] const T *end() const { return const_cast<pure_tensor<T> *>(this)->end(); }

        /* +-+-+-+-+-+-+-+-+-+- friends -+-+-+-+-+-+-+-+-+-+ */

        /// @brief stream output
        template<class U>
        friend ostream &operator<<(ostream &os, const pure_tensor<U> &t);

    private:
        /// @below store in cpu
        DeviceTypeDef _device = {EDevice::CPU, 0}; // device type and id

        /// @below store in cpu/gpu
        tensor_kernel<T> _kernel;
    }; // class pure_tensor

    /* +-+-+-+-+-+-+-+-+-+- logic -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    bool equal(const pure_tensor<T> &a, const pure_tensor<T> &b);
} // namespace ocean

#include "inls/pure_tensor.inl"

#endif // PURE_TENSOR_H
