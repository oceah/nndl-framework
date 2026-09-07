#include "nndl/tensor.h"
#include "nndl/nn.h"
#include "nndl/math.h"

#include <iostream>

static void test1()
{
    using tensor = ocean::tensor<float>;

    tensor a = {{1, 2}, {4, 5}};
    tensor b = {{2, 7}, {5, 8}};
    a.to("cuda");
    b.to("cuda");

    a.requires_grad(true);
    b.requires_grad(true);
    auto c = a * b;

    tensor cx = {{4, 5}, {3, 2}};
    cx.to("cuda");

    auto d = cx - c;
    d.grad().fill(1);
    d.backward();

    std::cout << "a = \n"
              << a << std::endl;
    std::cout << "da = \n"
              << a.grad() << std::endl;
    std::cout << "b = \n"
              << b << std::endl;
    std::cout << "db = \n"
              << b.grad() << std::endl;
    std::cout << "c = \n"
              << c << std::endl;
    std::cout << "dc = \n"
              << c.grad() << std::endl;
    std::cout << "d = \n"
              << d << std::endl;
    std::cout << "dd = \n"
              << d.grad() << std::endl;
}

static void test2()
{
    using tensor = ocean::tensor<float>;
    using Linear = ocean::nn::Linear<float>;

    constexpr int N = 32;        // 训练样本数
    constexpr int EPOCHS = 1000; // 训练轮次
    constexpr float LR = 0.01f;  // 学习率

    // 真实参数
    const float w_true[3] = {1.0f, -2.0f, 0.5f};
    const float b_true = 0.3f;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    tensor x(N, 3); // (N, 3)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < 3; ++j)
            x(i, j) = dis(gen);

    tensor y_true(N, 1); // (N, 1)
    for (int i = 0; i < N; ++i)
    {
        float yi = b_true;
        for (int j = 0; j < 3; ++j)
            yi += w_true[j] * x(i, j);
        y_true(i, 0) = yi;
    }

    x.to("cuda");
    y_true.to("cuda");

    Linear fc(3, 1, true);
    fc.to("cuda");

    tensor ones(N, 1);
    ones.fill(1.0f);
    ones.to("cuda");

    /* ---------- 训练 ---------- */
    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        auto y_pred = fc(x);
        auto d = y_pred - y_true;
        d.reshape(1, N);
        auto sq = ocean::nndl::power<float, 2>(d);
        auto loss = sq * ones;

        loss.grad().fill(1.0f);
        loss.backward();

        fc.update(LR);
        fc.weight().grad().fill(0.0f);
        fc.bias().grad().fill(0.0f);

        if (epoch % 100 == 0 || epoch == EPOCHS - 1)
            std::cout << "epoch " << epoch << "  loss = " << loss << std::endl;
    }

    /* ---------- 输出学习到的参数 ---------- */
    std::cout << "\nlearned w = \n"
              << fc.weight() << std::endl;
    std::cout << "learned b = " << fc.bias() << std::endl;
    std::cout << "expected w = 1 -2 0.5" << std::endl;
    std::cout << "expected b = 0.3" << std::endl;
}

int main()
{
    std::cout << "[Test 1]" << std::endl;
    test1();
    std::cout << "[Test 2]" << std::endl;
    test2();
}
