#pragma once

#include "../tensor.h"

namespace ocean {
    /* +-+-+-+-+-+-+-+-+-+- ComputationalGraphNode -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    void ComputationalGraphNode<T>::forward() {
        vector<tensor_kernel<T> *> ins, outs;
        auto in_deg = in_pt.size();
        auto out_deg = out_pt.size();
        ins.resize(in_deg);
        for (size_type i = 0; i < in_deg; ++i)
            ins[i] = &in_pt[i]->kernel();
        outs.resize(out_deg);
        for (size_type i = 0; i < out_deg; ++i)
            outs[i] = &out_pt[i]->kernel();
        op->forward(out_pt[0]->device(), ins.data(), in_deg, outs.data(), out_deg, params.data());
    }

    template<class T>
    void ComputationalGraphNode<T>::backward() {
        if (op == nullptr)
            throw std::runtime_error("graph node has no operator");
        // assert this node has an operator
        vector<tensor_kernel<T> *> ins, in_grads, outs, out_grads;
        auto in_deg = in_pt.size();
        auto out_deg = out_pt.size();
        ins.resize(in_deg);
        in_grads.resize(in_deg);
        for (size_type i = 0; i < in_deg; ++i) {
            ins[i] = &in_pt[i]->kernel();
            // debug: check if in_grads[i] is null
            // set in_grads[i] to nullptr if in_need_grad[i] is false
            in_grads[i] = in_pg[i] == nullptr ? nullptr : &in_pg[i]->kernel();
        }
        outs.resize(out_deg);
        out_grads.resize(out_deg);
        for (size_type i = 0; i < out_deg; ++i) {
            outs[i] = &out_pt[i]->kernel();
            out_grads[i] = &out_pg[i]->kernel();
        }
        op->backward(out_pt[0]->device(), outs.data(), out_grads.data(), out_deg,
                     ins.data(), in_grads.data(), in_deg, params.data());
        // check all the in edges
        for (size_type i = 0; i < in_gen.size(); ++i) {
            if (!in_need_grad[i] || !in_gen[i])
                continue;
            // assert this in edge need grad and has a valid node
            ++in_gen[i]->bp_cnt;
            if (in_gen[i]->bp_cnt < in_gen[i]->out_pt.size())
                continue;
            // assert all out edges have been back propagated
            in_gen[i]->bp_cnt = 0;
            in_gen[i]->backward();
        }
    }

    /* +-+-+-+-+-+-+-+-+-+- nndl -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    pure_tensor<T> &tensor<T>::grad() {
        if (!pg)
            throw std::runtime_error("this tensor has no gradient");
        // assert pg is not null
        return *pg;
    }

    template<class T>
    void tensor<T>::requires_grad(bool val) & {
        if (val == (pg.get() != nullptr)) {
            return;
        }
        if (val) {
            pg = std::make_shared<pure_tensor<T> >(pt->size(), pt->device());
        } else {
            pg.reset();
        }
    }

    template<class T>
    void tensor<T>::backward() {
        if (!requires_grad())
            return;
        // assert requires grad
        // backward from the current node
        // check if it is a leaf node
        if (!gen)
            return;
        // assert not leaf node
        auto &graph = *gen;
        ++graph.bp_cnt;
        if (graph.bp_cnt < graph.out_pt.size())
            return;
        // assert all out edges have been back propagated
        graph.bp_cnt = 0;
        graph.backward();
    }

    /* +-+-+-+-+-+-+-+-+-+- size & shape setters -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    template<class... Args>
    void tensor<T>::resize(Args... args) & {
        pt->resize(std::forward<Args>(args)...);
        if (pg)
            pg->resize(pt->size());
    }

    template<class T>
    template<class... Args>
    void tensor<T>::reshape(Args... args) & {
        pt->reshape(std::forward<Args>(args)...);
        if (pg)
            pg->reshape(pt->size());
    }

    template<class T>
    void tensor<T>::clear() & {
        if (pt) {
            pt->clear();
            pt.reset();
            if (pg) {
                pg->clear();
                pg.reset();
            }
            gen.reset();
        }
    }

    template<class T>
    template<class Iter>
    void tensor<T>::assign(Iter first, Iter last) & {
        pt->assign(first, last);
        if (pg)
            pg->assign(first, last);
    }

    /* +-+-+-+-+-+-+-+-+-+- device interface -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    template<class... Args>
    void tensor<T>::to(Args... args) & {
        pt->to(std::forward<Args>(args)...);
        if (pg)
            pg->to(pt->device());
    }

    template<class T>
    tensor<T> tensor<T>::cpu() && {
        pt->to("cpu");
        if (pg)
            pg->to("cpu");
        return std::move(*this);
    }

    template<class T>
    tensor<T> tensor<T>::cpu() const & {
        tensor res;
        *res.pt = pt->cpu();
        if (pg)
            res.pg = std::make_shared<pure_tensor<T> >(pg->cpu());
        return res;
    }

#if NNDL_USE_CUDA

    template<class T>
    tensor<T> tensor<T>::cuda() && {
        pt->to("cuda");
        if (pg)
            pg->to("cuda");
        return std::move(*this);
    }

    template<class T>
    tensor<T> tensor<T>::cuda() const & {
        tensor res;
        *res.pt = pt->cuda();
        if (pg)
            res.pg = std::make_shared<pure_tensor<T> >(pg->cuda());
        return res;
    }

#endif

    /* +-+-+-+-+-+-+-+-+-+- copy ctrls -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    tensor<T> &tensor<T>::operator=(const tensor &t) {
        clear();
        // deep copy
        pt = std::make_shared<pure_tensor<T> >(*t.pt);
        // automatic differentiation
        if (t.requires_grad()) {
            // if tensor t requires grad
            this->requires_grad(true);
            // create a new graph node
            // in: t
            // out: this
            auto graph = std::make_shared<ComputationalGraphNode<T> >();

            graph->in_pt = {t.pt};
            graph->in_pg = {t.pg};
            graph->in_gen = {t.gen};
            graph->in_need_grad = {true};

            graph->out_pt = {this->pt};
            graph->out_pg = {this->pg};

            graph->op = std::make_unique<assign_op<T> >();

            this->gen = graph;
        }
        return *this;
    }

    template<class T>
    tensor<T> &tensor<T>::operator=(tensor &&t) noexcept {
        clear();
        pt = std::move(t.pt);
        pg = std::move(t.pg);
        gen = std::move(t.gen);
        return *this;
    }

    /* +-+-+-+-+-+-+-+-+-+- operators -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    tensor<T> tensor<T>::operator+(const tensor &t) && {
        if (!this->requires_grad()) {
            // case 1: none requires grad
            if (!t.requires_grad()) {
                *pt += *t.pt;
                return std::move(*this);
            }
            // case 2: only t requires grad
            this->requires_grad(true);
            // create a new graph node
            // in: t
            // out: this
            auto graph = std::make_shared<ComputationalGraphNode<T> >();

            graph->in_pt = {t.pt};
            graph->in_pg = {t.pg};
            graph->in_gen = {t.gen};
            graph->in_need_grad = {true};

            graph->out_pt = {pt};
            graph->out_pg = {pg};
            // use assign_op to replace add_op for optimization
            graph->op = std::make_unique<assign_op<T> >();

            this->gen = graph;

            return std::move(*this);
        }
        // assert this requires grad
        // right value that requires grad is treated as left value
        return *this + t;
    }

    template<class T>
    tensor<T> tensor<T>::operator+(tensor &&t) && {
        if (!this->requires_grad())
            return std::move(*this) + t;
        // assert this requires grad
        return std::move(t) + this;
    }

    template<class T>
    tensor<T> tensor<T>::operator+(const tensor &t) const & {
        tensor res;
        *res.pt = *pt + *t.pt;
        // automatic differentiation
        if (this->requires_grad() || t.requires_grad()) {
            res.requires_grad(true);
            // create a new graph node
            // in: this, t
            // out: res
            auto graph = std::make_shared<ComputationalGraphNode<T> >();

            graph->in_pt = {this->pt, t.pt};
            graph->in_pg = {this->pg, t.pg};
            graph->in_gen = {this->gen, t.gen};
            graph->in_need_grad = {this->requires_grad(), t.requires_grad()};

            graph->out_pt = {res.pt};
            graph->out_pg = {res.pg};

            graph->op = std::make_unique<add_op<T> >();

            res.gen = graph;
        }
        return res;
    }

    template<class T>
    tensor<T> tensor<T>::operator-(const tensor &t) && {
        if (!this->requires_grad()) {
            // case 1: none requires grad
            if (!t.requires_grad()) {
                *pt -= *t.pt;
                return std::move(*this);
            }
            // case 2: only t requires grad
            this->requires_grad(true);
            // create a new graph node
            // in: t
            // out: this
            auto graph = std::make_shared<ComputationalGraphNode<T> >();

            graph->in_pt = {t.pt};
            graph->in_pg = {t.pg};
            graph->in_gen = {t.gen};
            graph->in_need_grad = {true};

            graph->out_pt = {pt};
            graph->out_pg = {pg};
            // use neg_op to replace sub_op for optimization
            graph->op = std::make_unique<neg_op<T> >();

            this->gen = graph;

            return std::move(*this);
        }
        // assert this requires grad
        // right value that requires grad is treated as left value
        return *this - t;
    }

    template<class T>
    tensor<T> tensor<T>::operator-(const tensor &t) const & {
        tensor res;
        *res.pt = *pt - *t.pt;
        // automatic differentiation
        if (this->requires_grad() || t.requires_grad()) {
            res.requires_grad(true);
            // create a new graph node
            // in: this, t
            // out: res
            auto graph = std::make_shared<ComputationalGraphNode<T> >();

            graph->in_pt = {this->pt, t.pt};
            graph->in_pg = {this->pg, t.pg};
            graph->in_gen = {this->gen, t.gen};
            graph->in_need_grad = {this->requires_grad(), t.requires_grad()};

            graph->out_pt = {res.pt};
            graph->out_pg = {res.pg};

            graph->op = std::make_unique<sub_op<T> >();

            res.gen = graph;
        }
        return res;
    }

    template<class T>
    tensor<T> tensor<T>::operator*(const tensor &t) const {
        // calculate result
        tensor res;
        *res.pt = *pt * *t.pt;

        // automatic differentiation
        if (this->requires_grad() || t.requires_grad()) {
            res.requires_grad(true);
            // create a new graph node
            // in: this, t
            // out: res
            auto graph = std::make_shared<ComputationalGraphNode<T> >();

            graph->in_pt = {this->pt, t.pt};
            graph->in_pg = {this->pg, t.pg};
            graph->in_gen = {this->gen, t.gen};
            graph->in_need_grad = {this->requires_grad(), t.requires_grad()};

            graph->out_pt = {res.pt};
            graph->out_pg = {res.pg};

            graph->op = std::make_unique<matmul_op<T> >();

            res.gen = graph;
        }
        return res;
    }

    /* +-+-+-+-+-+-+-+-+-+- scalar operators -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    tensor<T> tensor<T>::operator+(const T &s) && {
        if (!this->requires_grad()) {
            *pt += s;
            return std::move(*this);
        }
        // assert this requires grad
        // right value that requires grad is treated as left value
        return *this + s;
    }

    template<class T>
    tensor<T> tensor<T>::operator+(const T &s) const & {
        tensor res;
        *res.pt = *pt + s;
        // automatic differentiation
        if (this->requires_grad()) {
            res.requires_grad(true);
            // create a new graph node
            // in: this
            // out: res
            auto graph = std::make_shared<ComputationalGraphNode<T> >();

            graph->in_pt = {this->pt};
            graph->in_pg = {this->pg};
            graph->in_gen = {this->gen};
            graph->in_need_grad = {true};

            graph->out_pt = {res.pt};
            graph->out_pg = {res.pg};

            // use assign_op for optimization
            graph->op = std::make_unique<assign_op<T> >();

            res.gen = graph;
        }
        return res;
    }

    template<class T>
    tensor<T> tensor<T>::operator-() && {
        if (!this->requires_grad()) {
            *pt = -*pt;
            return std::move(*this);
        }
        // assert this requires grad
        // right value that requires grad is treated as left value
        return -*this;
    }

    template<class T>
    tensor<T> tensor<T>::operator-() const & {
        tensor res;
        *res.pt = -*pt;
        // automatic differentiation
        if (this->requires_grad()) {
            res.requires_grad(true);
            // create a new graph node
            // in: this
            // out: res
            auto graph = std::make_shared<ComputationalGraphNode<T> >();

            graph->in_pt = {this->pt};
            graph->in_pg = {this->pg};
            graph->in_gen = {this->gen};
            graph->in_need_grad = {true};

            graph->out_pt = {res.pt};
            graph->out_pg = {res.pg};

            graph->op = std::make_unique<neg_op<T> >();

            res.gen = graph;
        }
        return res;
    }

    /* +-+-+-+-+-+-+-+-+-+- friends -+-+-+-+-+-+-+-+-+-+ */

    template<class U>
    std::ostream &operator<<(std::ostream &os, const tensor<U> &t) {
        os << *t.pt;
        return os;
    }
} // namespace ocean
