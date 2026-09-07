/**
******************************************************************************
* @file             tensor.h
* @brief            tensor data structure[autograd]
* @author           OceanH
* @create           2024-11-01
* @latestupdate     2025-01-22
******************************************************************************
*/

#ifndef TENSOR_H
#define TENSOR_H

#include "pure_tensor.h"
#include "operator.h"

#include <memory>

using std::shared_ptr;
using std::unique_ptr;

namespace ocean
{
    /// @brief node of computational graph
    /// @design
    ///     1. ctrl the data flow, not responsible for memory allocation
    ///     2. memory management is done by tensor
    template <class T>
    struct ComputationalGraphNode
    {
        using tensor_ptr = shared_ptr<pure_tensor<T>>;       // tensor pointer type
        using node_ptr = shared_ptr<ComputationalGraphNode>; // node pointer type
        using op_type = tensor_op<T>;                        // operator type
        using operand_type = vector<tensor_ptr>;             // operand type
        using size_type = pure_tensor<T>::size_type;

        /// @brief input edge info
        vector<tensor_ptr> in_pt;  // input tensor data
        vector<tensor_ptr> in_pg;  // input gradient data
        vector<node_ptr> in_gen;   // input generator node
        vector<bool> in_need_grad; // input need gradient

        /// @brief output edge info
        vector<tensor_ptr> out_pt; // output tensor data
        vector<tensor_ptr> out_pg; // output gradient data

        /// @brief node info
        unique_ptr<op_type> op; // operator
        vector<T> params;       // operator parameters
        size_type bp_cnt = 0;   // back propagation count

        /// @brief data forward flow
        void forward();

        /// @brief gradient backward flow
        void backward();
    }; // struct ComputationalGraphNode

    /// @brief tensor with autograd
    template <class T>
    class tensor
    {
    public:
        using size_type = pure_tensor<T>::size_type;
        using shape_type = pure_tensor<T>::shape_type;
        using tensor_ptr = shared_ptr<pure_tensor<T>>;
        using op_type = tensor_op<T>;
        using node_ptr = ComputationalGraphNode<T>::node_ptr;

        /* +-+-+-+-+-+-+-+-+-+- constructors -+-+-+-+-+-+-+-+-+-+ */

        template <class... Args>
        tensor(Args... args)
            : pt(std::make_shared<pure_tensor<T>>(std::forward<Args>(args)...))
        {
        }

        /// @brief specify shape
        template <class... Args>
            requires(std::is_integral_v<Args> && ...)

        explicit tensor(Args... args)
            : pt(std::make_shared<pure_tensor<T>>(std::forward<Args>(args)...))
        {
        }

        explicit tensor(const shape_type &size)
            : pt(std::make_shared<pure_tensor<T>>(size))
        {
        }

        /// @brief construct a vector
        template <class U>
        tensor(const std::initializer_list<U> &list)
            : pt(std::make_shared<pure_tensor<T>>(
                  std::forward<const std::initializer_list<U>>(list)))
        {
        }

        /// @brief construct a matrix
        template <typename U>
        tensor(const std::initializer_list<std::initializer_list<U>> &list)
            : pt(std::make_shared<pure_tensor<T>>(
                  std::forward<const std::initializer_list<std::initializer_list<U>>>(list)))
        {
        }

        /* +-+-+-+-+-+-+-+-+-+- nndl -+-+-+-+-+-+-+-+-+-+ */

        /// @brief get gradient
        [[nodiscard]] pure_tensor<T> &grad();

        [[nodiscard]] const pure_tensor<T> &grad() const
        {
            return const_cast<tensor *>(this)->grad();
        }

        /// @brief get tensor pointer
        [[nodiscard]] tensor_ptr tensor_pointer() const
        {
            return pt;
        }

        /// @brief get gradient pointer
        [[nodiscard]] tensor_ptr grad_pointer() const
        {
            return pg;
        }

        /// @brief get computational graph node that generates this tensor
        [[nodiscard]] node_ptr generator() const
        {
            return gen;
        }

        /// @brief set computational graph node that generates this tensor
        void generator(node_ptr &&node)
        {
            gen = std::move(node);
        }

        void generator(const node_ptr &node)
        {
            gen = node;
        }

        /// @brief set computational graph node that generates this tensor
        void requires_grad(bool val) &;

        /// @brief get if this tensor requires gradient
        [[nodiscard]] bool requires_grad() const
        {
            return pg.get() != nullptr;
        }

        /// @brief back propagation
        void backward();

        /* +-+-+-+-+-+-+-+-+-+- access -+-+-+-+-+-+-+-+-+-+ */

        /// @brief access element by page, row and column
        template <class... Args>
            requires(std::conjunction_v<std::is_integral<Args>...>)
        T &operator()(Args... args)
        {
            return pt->operator()(std::forward<Args>(args)...);
        }

        template <typename... Args>
            requires(std::conjunction_v<std::is_integral<Args>...>)
        const T &operator()(Args... args) const
        {
            return const_cast<tensor *>(this)->operator()(std::forward<Args>(args)...);
        }

        /// @brief access element by index
        T &operator[](size_type i)
        {
            return pt->operator[](i);
        }

        const T &operator[](size_type i) const
        {
            return const_cast<tensor *>(this)->operator[](i);
        }

        /* +-+-+-+-+-+-+-+-+-+- getters -+-+-+-+-+-+-+-+-+-+ */

        /// @brief data pointer
        [[nodiscard]] T *data() noexcept
        {
            return pt->data();
        }

        [[nodiscard]] const T *data() const noexcept
        {
            return pt->data();
        }

        /// @brief dimension of tensor
        [[nodiscard]] size_type dim() const noexcept
        {
            return pt->dim();
        }

        /// @brief shape of tensor
        [[nodiscard]] shape_type size() const noexcept
        {
            return pt->size();
        }

        /// @brief specific size of tensor
        [[nodiscard]] size_type size(size_type i) const
        {
            return pt->size(i);
        }

        /// @brief number of elements
        [[nodiscard]] size_type numel() const noexcept
        {
            return pt->numel();
        }

        /// @brief if tensor is empty
        [[nodiscard]] bool empty() const noexcept
        {
            return pt->empty();
        }

        /// @brief where tensor is stored
        [[nodiscard]] DeviceTypeDef device() const noexcept
        {
            return pt->device();
        }

        /* +-+-+-+-+-+-+-+-+-+- size & shape setters -+-+-+-+-+-+-+-+-+-+ */

        /// @brief modify shape of tensor
        /// @note accessing tensor after resize is undefined
        /// @note if you want to reshape with equal capacity, please use reshape
        template <class... Args>
        void resize(Args... args) &;

        /// @brief reshape with equal capacity
        /// @note keep the total number of tensor elements and the row-major order of elements unchanged
        /// @note if the total number of elements changes, an exception will be thrown
        template <class... Args>
        void reshape(Args... args) &;

        /// @brief clear tensor data
        /// @note usually not necessary manually
        void clear() &;

        /// @brief assign iterator data to vector
        template <class Iter>
        void assign(Iter first, Iter last) &;

        /* +-+-+-+-+-+-+-+-+-+- value setters -+-+-+-+-+-+-+-+-+-+ */

        /// @brief fill tensor with specific value
        void fill(const T &val) & { pt->fill(val); }

        /* +-+-+-+-+-+-+-+-+-+- device interface -+-+-+-+-+-+-+-+-+-+ */

        /// @brief move tensor to specific device
        template <class... Args>
        void to(Args... args) &;

        /// @brief create a copy of tensor on CPU
        tensor cpu() &&;

        tensor cpu() const &;

#if NNDL_USE_CUDA

        /// @brief create a copy of tensor on CUDA
        tensor cuda() &&;

        tensor cuda() const &;

#endif

        /* +-+-+-+-+-+-+-+-+-+- copy ctrls -+-+-+-+-+-+-+-+-+-+ */

        tensor(const tensor &t) { *this = t; }

        tensor(tensor &&t) noexcept { *this = std::move(t); }

        tensor &operator=(const tensor &t);

        tensor &operator=(tensor &&t) noexcept;

        /* +-+-+-+-+-+-+-+-+-+- operators -+-+-+-+-+-+-+-+-+-+ */

        tensor operator+(const tensor &t) &&;

        tensor operator+(tensor &&t) &&;

        tensor operator+(tensor &&t) const & { return std::move(t) + *this; }

        tensor operator+(const tensor &t) const &;

        tensor operator-(const tensor &t) &&;

        // tensor operator-(tensor &&t) &&;

        // tensor operator-(tensor &&t) const &;

        tensor operator-(const tensor &t) const &;

        tensor operator*(const tensor &t) const;

        /* +-+-+-+-+-+-+-+-+-+- scalar operators -+-+-+-+-+-+-+-+-+-+ */

        tensor operator+(const T &s) &&;

        tensor operator+(const T &s) const &;

        // tensor operator*(const T &s) &&;

        // tensor operator*(const T &s) const &;

        tensor operator-() &&;

        tensor operator-() const &;

        /* +-+-+-+-+-+-+-+-+-+- iterator -+-+-+-+-+-+-+-+-+-+ */

        [[nodiscard]] auto begin() noexcept { return pt->begin(); }

        [[nodiscard]] auto begin() const noexcept { return pt->begin(); }

        [[nodiscard]] auto end() noexcept { return pt->end(); }

        [[nodiscard]] auto end() const noexcept { return pt->end(); }

        /* +-+-+-+-+-+-+-+-+-+- friends -+-+-+-+-+-+-+-+-+-+ */

        /// @brief stream out
        template <class U>
        friend std::ostream &operator<<(std::ostream &os, const tensor<U> &t);

    private:
        tensor_ptr pt; // tensor data
        tensor_ptr pg; // gradient data
        node_ptr gen;  // generator node
    }; // class tensor
} // namespace ocean

#include "inls/tensor.inl"

#endif // TENSOR_H
