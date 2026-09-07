/**
******************************************************************************
* @file         neural_net.h
* @brief        神经网络
* @author       透小犹
* @create       2024-9-9
* @lastupdate   2024-10-29
******************************************************************************
* @update
* 2024-10-25    修复了参数管理器中的构造函数覆盖拷贝成员导致的错误
* 2024-10-29    为线性变换单元提供 CUDA 加速
* 2024-10-30    为 Softmax 单元提供 CUDA 加速
******************************************************************************
*/

#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "tensor.h"

#include <map>
#include <string>
#include <vector>
#include <utility>
#include <functional>

using std::pair;
using std::vector;
using std::string;
using std::function;

/// @brief 调试开关
#define ONN_DBG 0

/// @brief CUDA 加速开关
#define ONN_USE_CUDA 1

// TODO 为 NNParam 添加范围管理

namespace ocean {

    /* +-+-+-+-+-+-+-+-+-+- 神经元基类 -+-+-+-+-+-+-+-+-+-+ */

    /// @brief 神经元基类
    /// @override input, output, backprop, update, connect, accept_connect
    template<class Float = double>
    class Neuron {
    public:
        using size_type = std::size_t;
        using tensor_type = tensor<Float>;

        virtual ~Neuron() = default;

        /// @brief 输入
        /// @param x 输入数据
        /// @param p 前驱神经元
        /// @param inertial 是否使用惯性参数
        ///     @value false 计算模型输出
        ///     @value true 计算惯性输出(用于NAG)
        /// @attention 保证输入数据的生命周期
        virtual void input(const tensor_type &x, Neuron *p = nullptr, bool inertial = false) = 0;

        /// @brief 输出
        /// @param p 后继神经元
        virtual inline const tensor_type &output(Neuron *p = nullptr) const = 0;

        /// @brief 反向传播误差
        /// @param dl 输出误差
        /// @param p 后继神经元
        /// @attention 不更新神经元参数时
        virtual void backprop(const tensor_type &dl, Neuron *p = nullptr) = 0;

        /// @brief 更新神经元参数
        /// @param alpha 学习率
        virtual void update(const Float &alpha) = 0;

        /// @brief 连接后继神经元
        /// @param p 后继神经元
        virtual inline void connect(Neuron *p) = 0;

    protected:
        /// @brief 连接前驱神经元
        /// @param p 前驱神经元
        virtual inline void accept_connect(Neuron *p) = 0;

        /// @brief 辅助连接神经元(派生类调用)
        /// @param p 后继神经元
        inline void aux_accept_connect(Neuron *p) {
            p->accept_connect(this);
        }
    }; // class Neuron

    /* +-+-+-+-+-+-+-+-+-+- 前馈神经网络 -+-+-+-+-+-+-+-+-+-+ */

    /// @input RixCi 矩阵
    /// @output RoxCo 矩阵
    template<class Float = double>
    class FNN {
    public:
        using size_type = Neuron<Float>::size_type;
        using tensor_type = Neuron<Float>::tensor_type;

        /// @param input_layer: 输入层
        /// @param output_layer: 输出层
        /// @param L: 损失函数 (default: 平方损失) L(y, yh) > 0
        /// @param dL: 损失函数的导数 (default: yh - y)
        FNN(Neuron<Float> *input_layer, Neuron<Float> *output_layer,
            const function<tensor_type(const tensor_type &, const tensor_type &)> &L = [](const tensor_type &y,
                                                                                          const tensor_type &yh) {
                auto d = yh - y;
                return d.scalar_mul(d) / (Float) 2;
            },
            const function<tensor_type(const tensor_type &, const tensor_type &)> &dL = [](const tensor_type &y,
                                                                                           const tensor_type &yh) {
                return yh - y;
            }) : input_layer(input_layer), output_layer(output_layer), L(L), dL(dL) {}

        /// @brief 单例训练|小批量训练
        /// @param x: 输入 NxD|NxKxD
        /// @param y: 输出 NxC|NxKxC
        /// @param alpha: 学习率
        void train(const tensor_type &x, const tensor_type &y, Float alpha);

        /// @brief 小批量训练
        /// @param X: 输入NxKxD
        /// @param Y: 输出NxKxC
        /// @param N: 数据集大小
        /// @param K: 小批量大小
        /// @param alpha: 学习率
        void train(const auto &X, const auto &Y, size_type N, size_type K, const Float &alpha) {
            using rand_gen_type = std::conditional_t<sizeof(void *) == 4, std::mt19937, std::mt19937_64>;

            std::vector<size_type> table(N);
            for (size_type i = 0; i < N; i++)
                table[i] = i;
            std::shuffle(table.begin(), table.end(), rand_gen_type(std::random_device{}()));

            auto &yh = output();
            size_type k = 0;
            for (size_type i = 0; i < N; i++) {
                size_type j = table[i];

                input(X[j], true);
                auto dl = dL(Y[j], yh);
                backprop(dl);

                ++k;
                if (k == K) {
                    k = 0;
                    update(alpha);
                }
            }
            if (k != 0)
                update(alpha);
        }

        /// @brief 模型在测试集上的损失
        tensor<Float> loss(const tensor<Float> &X, const tensor<Float> &Y) {
            input(X);
            auto &yh = output();
            tensor<Float> dl;
            switch (X.dim()) {
                case 1: // 单样本
                case 2: // 批量样本
                    dl = L(Y, yh);
                    break;
            }
            if (X.dim() > 1)
                dl = dl.mean();
            return dl;
        }

        /// @brief 模型输出
        [[nodiscard]] const tensor<Float> &operator()(const tensor_type &x) const {
            input(x);
            return output();
        }

    private:
        Neuron<Float> *input_layer;  // 输入层
        Neuron<Float> *output_layer; // 输出层

        // 损失函数(实际输出, 模型输出)
        function<tensor_type(const tensor_type &, const tensor_type &)> L;
        // 损失函数的导数(实际输出, 模型输出)
        function<tensor_type(const tensor_type &, const tensor_type &)> dL;

        /// @brief 输入
        /// @param inertial: 是否使用惯性参数
        ///     @value false: 计算模型输出
        ///     @value true: 计算惯性输出(用于NAG)
        inline void input(const tensor_type &X, bool inertial = false) const {
            input_layer->input(X, nullptr, inertial);
        }

        /// @brief 输出
        [[nodiscard]] inline const tensor_type &output() const { return output_layer->output(); }

        /// @brief 反向传播误差
        /// @param D: 输出误差
        inline void backprop(const tensor_type &D) { output_layer->backprop(D); }

        /// @brief 更新模型
        /// @param alpha: 学习率
        inline void update(const Float &alpha) { output_layer->update(alpha); }
    };

    /* +-+-+-+-+-+-+-+-+-+- 参数管理器 -+-+-+-+-+-+-+-+-+-+ */

    template<class Float = double>
    class NNParam {
    public:
        using tensor_type = tensor<Float>;
        using size_type = tensor_type::size_type;

        NNParam() noexcept = default;

        /// @brief 指定参数尺寸
        template<class... Args> requires (std::is_integral_v<std::remove_cvref_t<Args>> && ...)

        NNParam(Args &&...args);

        /// @brief 超参数设置
        /// @note 不调用则使用默认值
        /// @param beta: 梯度二阶矩衰减率
        /// @param rho: 动量因子
        /// @param clip: 梯度截断阈值
        /// @param lambda: 正则化参数
        /// @param eps: 防止除零
        void set(Float _beta, Float _rho, Float _clip, Float _lambda = 0, Float _eps = 1e-10);

        /// @brief 超参数设置
        /// @param opt: 超参数名
        ///     @value "beta": 梯度二阶矩衰减率
        ///     @value "rho": 动量因子
        ///     @value "clip": 梯度截断阈值
        ///     @value "lambda": 正则化参数
        ///     @value "eps": 防止除零
        /// @param val: 超参数值
        void set(const string &opt, Float val);

        /* +-+-+-+-+-+-+-+-+-+- 参数寻访 -+-+-+-+-+-+-+-+-+-+ */

        template<typename... Args>
        inline Float &operator()(Args... args) { return _param(std::forward<Args>(args)...); }

        template<typename... Args>
        inline const Float &operator()(Args... args) const { return _param(std::forward<Args>(args)...); }

        inline Float &operator[](size_type i) { return _param[i]; }

        inline const Float &operator[](size_type i) const { return _param[i]; }

        /// @brief 获取参数
        [[nodiscard]]  tensor_type &param() { return _param; }

        [[nodiscard]]  const tensor_type &param() const { return _param; }

        /// @brief 获取惯性参数
        [[nodiscard]]  tensor_type &inertia();

        /* +-+-+-+-+-+-+-+-+-+- 梯度管理 -+-+-+-+-+-+-+-+-+-+ */

        /// @brief 清空梯度
        /// @note 一般不需要手动调用
        inline void clear_grad();

        /// @brief 更新小批量梯度
        inline void push_grad(const tensor_type &g);

        /// @brief 更新参数
        void update(Float alpha);

        /* +-+-+-+-+-+-+-+-+-+- 访问器 -+-+-+-+-+-+-+-+-+-+ */

        /// @brief 参数尺寸
        [[nodiscard]]inline auto size() const { return _param.size(); }

        [[nodiscard]]inline auto size(size_type i) const { return _param.size(i); }

    private:
        tensor_type _param; // 参数
        tensor_type gparam; // 梯度
        Float K;            // 小批量大小

        /* +-+-+-+-+- 学习率调整 -+-+-+-+-+ */
        Float beta = 0.9;   // [超参数]梯度二阶矩衰减率
        Float eps = 1e-10;  // [超参数]防止除零
        tensor_type Gt;     // 梯度二阶矩
        tensor_type Dt;     // 参数更新差值

        /* +-+-+-+-+- 梯度估计修正 -+-+-+-+-+ */
        Float rho = 0.9;    // [超参数]动量因子
        tensor_type Mt;     // 动量
        tensor_type paramh; // 惯性参数 paramh = param + rho * Mt

        /* +-+-+-+-+- 梯度截断 -+-+-+-+-+ */
        Float clip = 1.0;   // [超参数]梯度阈值

        /* +-+-+-+-+- 正则化 -+-+-+-+-+ */
        Float lambda = 0;   // 正则化参数
    };

    template<class T>
    std::ostream &operator<<(std::ostream &os, const NNParam<T> &pm);

    /* +-+-+-+-+-+-+-+-+-+- SISO神经元 -+-+-+-+-+-+-+-+-+-+ */

    /// @brief SISO神经元
    /// @override cal_out, cal_inerr, cal_grad
    template<class Float = double>
    class SISONeuron : public Neuron<Float> {
    public:
        using size_type = Neuron<Float>::size_type;
        using tensor_type = Neuron<Float>::tensor_type;
        using param_type = NNParam<Float>;

        virtual ~SISONeuron() = default;

        virtual void input(const tensor_type &x, Neuron<Float> *p = nullptr, bool inertial = false) override;

        [[nodiscard]] virtual inline const tensor_type &output(Neuron<Float> *p = nullptr) const override {
            return out;
        }

        virtual void backprop(const tensor_type &dl, Neuron<Float> *p = nullptr) override;

        virtual void update(const Float &alpha) override;

        virtual inline void connect(Neuron<Float> *p) override;

    protected:
        using param_getter = function<
                tensor_type &(size_type
                              idx)>;

        Neuron<Float> *prev = nullptr;  // 前驱神经元
        Neuron<Float> *next = nullptr;  // 后继神经元

        const tensor_type *pin; // 输入数据指针
        const tensor_type *pdl; // 输出误差指针
        tensor_type out;        // 输出
        tensor_type inerr;      // 输入误差

        vector<param_type> param;  // 参数

        /// @brief 计算输出
        /// @param x 输入
        /// @param y 输出
        /// @param pg 参数获取器
        virtual void cal_out(const tensor_type &x, const param_getter &pg, tensor_type &y) = 0;

        /// @brief 计算输入误差
        /// @param x 输入
        /// @param y 输出
        /// @param dl 输出误差
        /// @param pg 参数获取器
        /// @param dx 输入误差
        virtual void cal_inerr(const tensor_type &x, const tensor_type &y,
                               const tensor_type &dl, const param_getter &pg,
                               tensor_type &dx) = 0;

        /// @brief 计算梯度
        /// @param x 输入
        /// @param y 输出
        /// @param dl 输出误差
        /// @param p 参数
        virtual vector<tensor_type> cal_grad(const tensor_type &x, const tensor_type &y,
                                             const tensor_type &dl, const param_getter &pg) = 0;

        virtual inline void accept_connect(Neuron<Float> *p) override { prev = p; }
    }; // class SISONeuron

    /* +-+-+-+-+-+-+-+-+-+- 神经分路器 -+-+-+-+-+-+-+-+-+-+ */

    /// @brief 神经分路器
    /// @note 将输入数据分发到多个后继神经元
    /// @note 不会改变输入数据
    template<class Float = double>
    class NNBrancher : public Neuron<Float> {
    public:
        using size_type = Neuron<Float>::size_type;
        using tensor_type = Neuron<Float>::tensor_type;

        virtual ~NNBrancher() = default;

        /// @brief 输入
        /// @param x: 输入数据
        /// @param p: 前驱神经元
        /// @param inertial: 是否使用惯性参数
        ///     @value false: 计算模型输出
        ///     @value true: 计算惯性输出(用于NAG)
        virtual void input(const tensor_type &x, Neuron<Float> *p = nullptr, bool inertial = false) override;

        /// @brief 输出
        /// @param p: 后继神经元
        [[nodiscard]] virtual inline const tensor_type &output(Neuron<Float> *p = nullptr) const override {
            return *pin;
        }

        /// @brief 反向传播误差
        /// @param dl: 输出误差
        /// @param p: 后继神经元
        virtual void backprop(const tensor_type &dl, Neuron<Float> *p = nullptr) override;

        /// @brief 更新神经元参数
        /// @param alpha: 学习率
        virtual void update(const Float &alpha) override;

        /// @brief 连接后继神经元
        /// @param p: 后继神经元
        virtual inline void connect(Neuron<Float> *p) override;

    private:
        Neuron<Float> *prev = nullptr; // 前驱神经元
        const tensor_type *pin;        // 输入数据指针
        tensor_type inerr;             // 输入误差

        std::map<void *, const tensor_type *> pdls; // 输出误差指针
        size_type ready = 0;                        // 就绪分支数

        /// @brief 连接前驱神经元
        /// @param p: 前驱神经元
        virtual inline void accept_connect(Neuron<Float> *p) override { prev = p; }
    };

    /* +-+-+-+-+-+-+-+-+-+- 神经并路器 -+-+-+-+-+-+-+-+-+-+ */

    /// @brief 神经并路器
    /// @note 将多个输入数据合并到一个后继神经元
    /// @example x1(NxD1), x2(NxD2) => y(Nx(D1+D2))
    template<class Float = double>
    class NNMerger : public Neuron<Float> {
    public:
        using size_type = Neuron<Float>::size_type;
        using tensor_type = Neuron<Float>::tensor_type;

        virtual ~NNMerger() = default;

        /// @brief 输入
        /// @param x: 输入数据
        /// @param p: 前驱神经元
        /// @param inertial: 是否使用惯性参数
        ///     @value false: 计算模型输出
        ///     @value true: 计算惯性输出(用于NAG)
        virtual void input(const tensor_type &x, Neuron<Float> *p = nullptr, bool inertial = false) override;

        /// @brief 输出
        /// @param p: 后继神经元
        [[nodiscard]] virtual inline const tensor_type &output(Neuron<Float> *p = nullptr) const override {
            return out;
        }

        /// @brief 反向传播误差
        /// @param dl: 输出误差
        /// @param p: 后继神经元
        virtual void backprop(const tensor_type &dl, Neuron<Float> *p = nullptr) override;

        /// @brief 更新神经元参数
        /// @param alpha: 学习率
        virtual void update(const Float &alpha) override;

        /// @brief 连接后继神经元
        /// @param p: 后继神经元
        virtual inline void connect(Neuron<Float> *p) override;

    private:
        struct InputInfo {
            const tensor_type *pin; // 输入数据指针
            tensor_type inerr;      // 输入误差
        };
        std::map<void *, InputInfo> prevs; // 前驱神经元
        size_type ready = 0;               // 就绪分支数

        Neuron<Float> *next = nullptr; // 后继神经元
        const tensor_type *pdl;        // 输出误差指针
        tensor_type out;               // 输出

        /// @brief 连接前驱神经元
        /// @param p: 前驱神经元
        virtual inline void accept_connect(Neuron<Float> *p) override;
    };

    /* +-+-+-+-+-+-+-+-+-+- 数据管理器 -+-+-+-+-+-+-+-+-+-+ */

    /// @design 方便读取excel/csv
    /// 1. 数据一定是行向量
    /// 2. 数据集一定是矩阵

    /// @brief 标签 => one-hot 行向量
    /// @param Y 标签 Nx1
    /// @param C 类别数
    /// @note 类别标签从 0 开始
    template<class Float = double>
    tensor<Float> to_one_hat(const tensor<Float> &Y, size_t C);

    /// @brief one-hot 行向量 => 标签
    /// @param Y one-hot 行向量 Nx1
    /// @note 类别标签从 0 开始
    template<class Float = double>
    tensor<Float> argmax(const tensor<Float> &Y);

    /// @brief 数据归一化器
    /// @note 数据以行向量形式存储
    /// @override normalize, denormalize
    template<class Float = double>
    class DataNormalizer : public SISONeuron<Float> {
    public:
        using tensor_type = tensor<Float>;
        using size_type = tensor_type::size_type;

        virtual ~DataNormalizer() = default;

        virtual void update(const Float &alpha) override {}

    protected:
        using param_getter = SISONeuron<Float>::param_getter;

        virtual void cal_out(const tensor_type &x, const param_getter &pg, tensor_type &y) override;

        virtual void cal_inerr(const tensor_type &x, const tensor_type &y,
                               const tensor_type &dl, const param_getter &pg,
                               tensor_type &dx) override;

        virtual vector<tensor_type> cal_grad(const tensor_type &x, const tensor_type &y,
                                             const tensor_type &dl, const param_getter &pg) override {
            return {};
        }

        /// @brief 归一化
        /// @param x: 数据
        virtual void normalize(tensor_type &x) const = 0;

        /// @brief 反归一化
        /// @param x: 数据
        /// @return 反归一化后的数据
        virtual void denormalize(tensor_type &x) const = 0;
    };

    /// @brief 最小最大值归一化
    template<class Float = double>
    class MinMaxNormalizer : public DataNormalizer<Float> {
    public:
        using tensor_type = DataNormalizer<Float>::tensor_type;
        using size_type = DataNormalizer<Float>::size_type;

        MinMaxNormalizer() noexcept = default;

        MinMaxNormalizer(const tensor_type &X, Float minval = 0, Float maxval = 1) {
            set(X, minval, maxval);
        }

        /// @brief 设置数据集
        /// @param X: 数据集
        /// @reflect 最小值 => minval
        /// @reflect 最大值 => maxval
        void set(const tensor<Float> &X, Float minval = 1, Float maxval = 1);

        template<class T>
        friend std::ostream &operator<<(std::ostream &os, const MinMaxNormalizer<T> &n);

    protected:
        /// @brief 归一化
        /// @param x: 数据
        virtual void normalize(tensor_type &x) const override;

        /// @brief 反归一化
        /// @param x: 数据
        virtual void denormalize(tensor_type &x) const override;

    private:
        tensor_type min;
        tensor_type range;
    };

    /* +-+-+-+-+-+-+-+-+-+- 激活函数 -+-+-+-+-+-+-+-+-+-+ */

    /* +-+-+-+-+- Softmax函数 -+-+-+-+-+ */

    /// @brief Softmax函数
    template<class Float = double>
    class Softmax : public SISONeuron<Float> {
    public:
        using size_type = SISONeuron<Float>::size_type;
        using tensor_type = SISONeuron<Float>::tensor_type;

        template<class T>
        friend std::ostream &operator<<(std::ostream &os, const Softmax<T> &n);

    private:
        using param_getter = SISONeuron<Float>::param_getter;

        virtual void cal_out(const tensor_type &x, const param_getter &pg, tensor_type &y) override;

        virtual void cal_inerr(const tensor_type &x, const tensor_type &y,
                               const tensor_type &dl, const param_getter &pg,
                               tensor_type &dx) override;

        virtual vector<tensor_type> cal_grad(const tensor_type &x, const tensor_type &y,
                                             const tensor_type &dl, const param_getter &pg) override {
            return {};
        }
    };

    /* +-+-+-+-+- ReLU函数 -+-+-+-+-+ */

    /// @brief 带参数的ReLU
    template<class Float = double>
    class PReLU : public SISONeuron<Float> {
    public:
        using size_type = SISONeuron<Float>::size_type;
        using tensor_type = SISONeuron<Float>::tensor_type;

        PReLU();

        template<class T>
        friend std::ostream &operator<<(std::ostream &os, const PReLU<T> &n);

    private:
        using param_getter = SISONeuron<Float>::param_getter;

        virtual void cal_out(const tensor_type &x, const param_getter &pg, tensor_type &y) override;

        virtual void cal_inerr(const tensor_type &x, const tensor_type &y,
                               const tensor_type &dl, const param_getter &pg,
                               tensor_type &dx) override;

        virtual vector<tensor_type> cal_grad(const tensor_type &x, const tensor_type &y,
                                             const tensor_type &dl, const param_getter &pg) override;
    };

    /* +-+-+-+-+-+-+-+-+-+- 线性变换单元 -+-+-+-+-+-+-+-+-+-+ */

    /// @input NxD 向量
    /// @output NxC 向量
    template<class Float = double>
    class LinearTransUnit : public SISONeuron<Float> {
    public:
        using size_type = SISONeuron<Float>::size_type;
        using tensor_type = SISONeuron<Float>::tensor_type;

        /// @pramm D: 输入维度
        /// @pramm C: 输出维度
        LinearTransUnit(size_type D, size_type C);

        /* +-+-+-+-+- 友元 -+-+-+-+-+ */

        template<class T>
        friend std::ostream &operator<<(std::ostream &os, const LinearTransUnit<T> &n);

    private:
        using param_getter = SISONeuron<Float>::param_getter;

        virtual void cal_out(const tensor_type &x, const param_getter &pg, tensor_type &y) override;

        virtual void cal_inerr(const tensor_type &x, const tensor_type &y,
                               const tensor_type &dl, const param_getter &pg,
                               tensor_type &dx) override;

        virtual vector<tensor_type> cal_grad(const tensor_type &x, const tensor_type &y,
                                             const tensor_type &dl, const param_getter &pg) override;
    };

    /* +-+-+-+-+-+-+-+-+-+- 一维卷积单元 -+-+-+-+-+-+-+-+-+-+ */

    /// @input NxD 向量
    /// @output NxC 向量
    template<class Float = double>
    class ConvUnit : public SISONeuron<Float> {
    public:
        using size_type = SISONeuron<Float>::size_type;
        using tensor_type = SISONeuron<Float>::tensor_type;

        /// @pramm D: 输入维度
        /// @pramm K: 卷积核大小
        ConvUnit(size_type D, size_type K);

        /// @brief 输出尺寸
        [[nodiscard]] inline size_type out_size() const { return this->param[1].size(0); }

        /* +-+-+-+-+- 友元 -+-+-+-+-+ */

        template<class T>
        friend std::ostream &operator<<(std::ostream &os, const ConvUnit<T> &n);

    private:
        using param_getter = SISONeuron<Float>::param_getter;

        virtual void cal_out(const tensor_type &x, const param_getter &pg, tensor_type &y) override;

        virtual void cal_inerr(const tensor_type &x, const tensor_type &y,
                               const tensor_type &dl, const param_getter &pg,
                               tensor_type &dx) override;

        virtual vector<tensor_type> cal_grad(const tensor_type &x, const tensor_type &y,
                                             const tensor_type &dl, const param_getter &pg) override;
    };

    /* +-+-+-+-+-+-+-+-+-+- 一维平均汇聚单元 -+-+-+-+-+-+-+-+-+-+ */

    /// @input NxD 向量
    /// @output NxD/K 向量
    template<class Float = double>
    class AverPoolUnit : public SISONeuron<Float> {
    public:
        using size_type = SISONeuron<Float>::size_type;
        using tensor_type = SISONeuron<Float>::tensor_type;

        /// @pramm D: 输入维度
        /// @pramm K: 卷积核大小
        explicit AverPoolUnit(size_type D, size_type K) : D(D), K(K) {}

        /// @brief 输出尺寸
        [[nodiscard]] inline size_type out_size() const { return (D + K - 1) / K; }

        /* +-+-+-+-+- 友元 -+-+-+-+-+ */

        template<class T>
        friend std::ostream &operator<<(std::ostream &os, const AverPoolUnit<T> &n);

    private:
        size_type D; // 输入维度
        size_type K; // 汇聚核大小

        using param_getter = SISONeuron<Float>::param_getter;

        virtual void cal_out(const tensor_type &x, const param_getter &pg, tensor_type &y) override;

        virtual void cal_inerr(const tensor_type &x, const tensor_type &y,
                               const tensor_type &dl, const param_getter &pg,
                               tensor_type &dx) override;

        virtual vector<tensor_type> cal_grad(const tensor_type &x, const tensor_type &y,
                                             const tensor_type &dl, const param_getter &pg) override {
            return {};
        }
    };

} // namespace ocean

#include "inls/neural_net.inl"

#endif // NEURAL_NET_H
