#pragma once

#include "nndl/nn.h"
#include <random>

namespace ocean::nn {
    /// Linear begin

    template<class T>
    Linear<T>::Linear(size_type in_features,
                      size_type out_features,
                      bool bias,
                      DeviceTypeDef device) : _in_features(in_features),
                                              _out_features(out_features),
                                              _device(device) {
        w = tensor<T>(in_features, out_features);
        // He initialization
        T r = std::sqrt(6.0 / in_features);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(-r, r);
        for (auto &v: w)
            v = dis(gen);
        if (bias) {
            b = tensor<T>(out_features);
            b.fill(0);
            if (_device.type != EDevice::ANY)
                b.to(_device);
        }
        w.requires_grad(true);
        b.requires_grad(true);
    }

    template<class T>
    tensor<T> Linear<T>::operator()(const tensor<T> &x) const {
        auto y = x * w;
        if (!b.empty()) {
            tensor<T> ones(x.size(0), 1);
            ones.fill(static_cast<T>(1));
            ones.to(x.device());
            auto yb = y + ones * b;
            return yb;
        }
        return y;
    }

    /* +-+-+-+-+-+-+-+-+-+- setters -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    template<class... Args>
    void Linear<T>::to(Args &&... args) & {
        w.to(std::forward<Args>(args)...);
        b.to(std::forward<Args>(args)...);
    }

    template<class T>
    void Linear<T>::update(T lr) {
        auto helper = [&](tensor<T> &t) {
            auto pt = t.tensor_pointer();
            auto pg = t.grad_pointer();
            *pt -= (*pg) * lr;
        };
        helper(w);
        if (!b.empty())
            helper(b);
    }

    /* +-+-+-+-+-+-+-+-+-+- friendss -+-+-+-+-+-+-+-+-+-+ */

    template<class T>
    std::ostream &operator<<(std::ostream &os, const Linear<T> &linear) {
        // format: Linear(in_features=3, out_features=2, bias=True)
        os << "Linear(in_features=" << linear._in_features
                << ", out_features=" << linear._out_features
                << ", bias=" << (linear.b.empty() ? "False" : "True") << ")";
        return os;
    }

    /// Linear end
} // namespace ocean::nn
