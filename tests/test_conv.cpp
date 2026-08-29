// Conv2D / MaxPool2D 测试（移植自上游，fork assert 风格）
#include <neuralnet/nn/nn.h>
#include <neuralnet/layer.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

/**
 * @brief Compares two floating-point values within a specified tolerance.
 *
 * @param a First value.
 * @param b Second value.
 * @param tol Maximum allowed absolute difference.
 * @return `true` if the absolute difference between the values is less than `tol`, `false` otherwise.
 */
static bool approx(double a, double b, double tol = 1e-6)
{
    return std::fabs(a - b) < tol;
}

/**
 * @brief Verifies hand-computed Conv2D forward results with and without bias.
 */

static void test_conv2d_forward_hand()
{
    std::puts("  [Conv2D] forward (hand-computed) ...");

    nn::Conv2D conv(1, 1, 2, 3, 3); // C_in=1, C_out=1, k=2, 3x3, stride=1, pad=0
    auto params = conv.parameters();
    nn::Matrix &W = params[0].get();
    nn::Matrix &bias = params[1].get();
    W.set_value_unchecked(0, 0, 1.0);
    W.set_value_unchecked(0, 1, 2.0);
    W.set_value_unchecked(0, 2, 3.0);
    W.set_value_unchecked(0, 3, 4.0);

    // x = [[1,2,3],[4,5,6],[7,8,9]]，行主序平铺
    nn::Matrix x(9, 1);
    for (std::size_t i = 0; i < 9; ++i)
        x.set_value_unchecked(i, 0, static_cast<double>(i + 1));

    nn::Matrix out = conv.forward(x);
    assert(out.rows() == 4 && out.cols() == 1); // 2x2 输出平铺

    // 窗口 (oh,ow)：out = 1*x00 + 2*x01 + 3*x10 + 4*x11（对应窗口元素）
    assert(approx(out.at(0, 0), 1 * 1 + 2 * 2 + 3 * 4 + 4 * 5));   // 37
    assert(approx(out.at(1, 0), 1 * 2 + 2 * 3 + 3 * 5 + 4 * 6));   // 47
    assert(approx(out.at(2, 0), 1 * 4 + 2 * 5 + 3 * 7 + 4 * 8));   // 67
    assert(approx(out.at(3, 0), 1 * 5 + 2 * 6 + 3 * 8 + 4 * 9));   // 77

    // 偏置广播
    bias.set_value_unchecked(0, 0, 0.5);
    out = conv.forward(x);
    assert(approx(out.at(0, 0), 37.5));
    assert(approx(out.at(3, 0), 77.5));

    std::puts("  [Conv2D] forward PASSED");
}

/**
 * @brief Verifies padded Conv2D im2col basis extraction against expected input elements.
 */
static void test_conv2d_im2col_basis()
{
    std::puts("  [Conv2D] im2col basis extraction (with padding) ...");

    // C_in=1, 4x4, k=3, stride=1, pad=1 → 4x4 输出。
    // W 行向量 = e_q 时，输出恰为 im2col 第 q 行的重排 —— 逐 q 验证。
    const std::size_t H = 4, Wd = 4, k = 3, OH = 4, OW = 4;
    nn::Conv2D conv(1, 1, k, H, Wd, 1, 1);

    nn::Matrix x(H * Wd, 1);
    for (std::size_t i = 0; i < H; ++i)
        for (std::size_t j = 0; j < Wd; ++j)
            x.set_value_unchecked(i * Wd + j, 0, static_cast<double>(i * 10 + j));

    auto params = conv.parameters();
    nn::Matrix &W = params[0].get();

    for (std::size_t q = 0; q < k * k; ++q)
    {
        for (std::size_t c = 0; c < k * k; ++c)
            W.set_value_unchecked(0, c, c == q ? 1.0 : 0.0);

        nn::Matrix out = conv.forward(x);
        const std::size_t kh = q / k, kw = q % k;
        for (std::size_t oh = 0; oh < OH; ++oh)
            for (std::size_t ow = 0; ow < OW; ++ow)
            {
                const long ih = static_cast<long>(oh + kh) - 1; // stride=1, pad=1
                const long iw = static_cast<long>(ow + kw) - 1;
                double expect = 0.0; // 越界 → 零填充
                if (ih >= 0 && iw >= 0 && ih < static_cast<long>(H) && iw < static_cast<long>(Wd))
                    expect = static_cast<double>(ih * 10 + iw);
                assert(approx(out.at(oh * OW + ow, 0), expect));
            }
    }

    std::puts("  [Conv2D] im2col basis extraction PASSED");
}

/**
 * @brief Verifies Conv2D input and parameter gradients using central differences.
 */
static void test_conv2d_gradient_check()
{
    std::puts("  [Conv2D] gradient check (central difference) ...");

    const std::size_t C_in = 2, H = 4, Wd = 4, k = 3, C_out = 3, B = 2;
    nn::Conv2D conv(C_in, C_out, k, H, Wd, /*stride=*/1, /*pad=*/1);

    std::mt19937_64 rng(101);
    std::normal_distribution<double> dist(0.0, 1.0);
    nn::Matrix in(C_in * H * Wd, B);
    for (auto &v : in.data())
        v = dist(rng);
    nn::Matrix R(C_out * H * Wd, B); // OH=OW=4
    for (auto &v : R.data())
        v = dist(rng);

    auto loss_of = [&](const nn::Matrix &inp)
    {
        nn::Matrix o = conv.forward(inp);
        double s = 0.0;
        for (std::size_t t = 0; t < o.size(); ++t)
            s += o.data()[t] * R.data()[t];
        return s;
    };

    (void)conv.forward(in);
    nn::Matrix grad_in = conv.backward(R);
    assert(grad_in.rows() == C_in * H * Wd && grad_in.cols() == B);

    const double eps = 1e-6;
    // 1) 输入梯度
    for (std::size_t i = 0; i < grad_in.rows(); ++i)
        for (std::size_t j = 0; j < B; ++j)
        {
            const double orig = in.at_unchecked(i, j);
            in.set_value_unchecked(i, j, orig + eps);
            const double lp = loss_of(in);
            in.set_value_unchecked(i, j, orig - eps);
            const double lm = loss_of(in);
            in.set_value_unchecked(i, j, orig);
            assert(approx((lp - lm) / (2 * eps), grad_in.at_unchecked(i, j), 1e-4));
        }

    // 2) 参数梯度（W 与 b）
    auto param_refs = conv.parameters();
    auto grad_refs = conv.param_gradients();
    nn::Matrix &W = param_refs[0].get();
    nn::Matrix &bias = param_refs[1].get();
    const nn::Matrix &dW = grad_refs[0].get();
    const nn::Matrix &db = grad_refs[1].get();

    for (std::size_t r = 0; r < W.rows(); ++r)
        for (std::size_t c = 0; c < W.cols(); ++c)
        {
            const double orig = W.at_unchecked(r, c);
            W.set_value_unchecked(r, c, orig + eps);
            const double lp = loss_of(in);
            W.set_value_unchecked(r, c, orig - eps);
            const double lm = loss_of(in);
            W.set_value_unchecked(r, c, orig);
            assert(approx((lp - lm) / (2 * eps), dW.at_unchecked(r, c), 1e-4));
        }
    for (std::size_t r = 0; r < bias.rows(); ++r)
    {
        const double orig = bias.at_unchecked(r, 0);
        bias.set_value_unchecked(r, 0, orig + eps);
        const double lp = loss_of(in);
        bias.set_value_unchecked(r, 0, orig - eps);
        const double lm = loss_of(in);
        bias.set_value_unchecked(r, 0, orig);
        assert(approx((lp - lm) / (2 * eps), db.at_unchecked(r, 0), 1e-4));
    }

    std::puts("  [Conv2D] gradient check PASSED");
}

/**
 * @brief Verifies MaxPool2D forward propagation for non-overlapping and overlapping windows.
 */

static void test_maxpool2d_forward()
{
    std::puts("  [MaxPool2D] forward ...");

    // 4x4 输入 x(i,j) = i*10 + j（各不相同）
    nn::Matrix x(16, 1);
    for (std::size_t i = 0; i < 16; ++i)
        x.set_value_unchecked(i, 0, static_cast<double>(i));

    // pool=2, stride=2 → 2x2
    {
        nn::MaxPool2D pool(1, 4, 4, 2, 2);
        nn::Matrix out = pool.forward(x);
        assert(out.rows() == 4 && out.cols() == 1);
        assert(approx(out.at(0, 0), 5.0));   // max{0,1,4,5}
        assert(approx(out.at(1, 0), 7.0));   // max{2,3,6,7}
        assert(approx(out.at(2, 0), 13.0));  // max{8,9,12,13}
        assert(approx(out.at(3, 0), 15.0));  // max{10,11,14,15}
    }

    // pool=2, stride=1 → 3x3 重叠窗口
    {
        nn::MaxPool2D pool(1, 4, 4, 2, 1);
        nn::Matrix out = pool.forward(x);
        assert(out.rows() == 9 && out.cols() == 1);
        const double expect[9] = {5, 6, 7, 9, 10, 11, 13, 14, 15};
        for (std::size_t i = 0; i < 9; ++i)
            assert(approx(out.at(i, 0), expect[i]));
    }

    std::puts("  [MaxPool2D] forward PASSED");
}

/**
 * @brief Verifies MaxPool2D backward gradient scattering for non-overlapping and overlapping windows.
 */
static void test_maxpool2d_backward()
{
    std::puts("  [MaxPool2D] backward scatter ...");

    // 非重叠：梯度散射到各窗口 argmax，其余为 0
    {
        nn::MaxPool2D pool(1, 4, 4, 2, 2);
        nn::Matrix x(16, 1);
        for (std::size_t i = 0; i < 16; ++i)
            x.set_value_unchecked(i, 0, static_cast<double>(i));
        (void)pool.forward(x);

        nn::Matrix g(4, 1);
        for (std::size_t i = 0; i < 4; ++i)
            g.set_value_unchecked(i, 0, 1.0);
        nn::Matrix gin = pool.backward(g);
        assert(gin.rows() == 16);
        // argmax 行索引：窗口 0→5, 1→7, 2→13, 3→15
        const std::size_t argmax_rows[4] = {5, 7, 13, 15};
        double total = 0.0;
        for (std::size_t i = 0; i < 16; ++i)
        {
            double expect = 0.0;
            for (std::size_t a : argmax_rows)
                if (i == a)
                    expect = 1.0;
            assert(approx(gin.at(i, 0), expect));
            total += gin.at(i, 0);
        }
        assert(approx(total, 4.0));
    }

    // 重叠（stride=1）：同一格是两个窗口的 argmax → 梯度累加
    {
        nn::MaxPool2D pool(1, 4, 4, 2, 1);
        nn::Matrix x(16, 1);
        for (std::size_t i = 0; i < 16; ++i)
            x.set_value_unchecked(i, 0, static_cast<double>(i));
        x.set_value_unchecked(2 * 4 + 2, 0, 100.0); // x22 = 全局最大
        (void)pool.forward(x);

        nn::Matrix g(9, 1);
        for (std::size_t i = 0; i < 9; ++i)
            g.set_value_unchecked(i, 0, 1.0);
        nn::Matrix gin = pool.backward(g);

        // x22（行 10）是包含它的 4 个重叠窗口 (1,1)(1,2)(2,1)(2,2) 的最大 → gin[10] = 4
        assert(approx(gin.at(10, 0), 4.0));
        double total = 0.0;
        for (std::size_t i = 0; i < 16; ++i)
            total += gin.at(i, 0);
        assert(approx(total, 9.0)); // 9 个窗口各散射 1
    }

    std::puts("  [MaxPool2D] backward scatter PASSED");
}

/**
 * @brief Verifies output and gradient shapes through a Conv2D, ReLU, and MaxPool2D pipeline.
 */

static void test_conv_relu_pool_composite()
{
    std::puts("  [Conv2D→ReLU→MaxPool2D] composite shapes ...");

    const std::size_t B = 2;
    nn::Conv2D conv(1, 2, 3, 4, 4, 1, 1); // → (2*4*4=32, B)
    nn::ReLU relu;
    nn::MaxPool2D pool(2, 4, 4, 2, 2);    // → (2*2*2=8, B)

    nn::Matrix x(16, B);
    std::mt19937_64 rng(77);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto &v : x.data())
        v = dist(rng);

    nn::Matrix h1 = conv.forward(x);
    assert(h1.rows() == 32 && h1.cols() == B);
    nn::Matrix h2 = relu.forward(h1);
    assert(h2.rows() == 32 && h2.cols() == B);
    nn::Matrix out = pool.forward(h2);
    assert(out.rows() == 8 && out.cols() == B);

    nn::Matrix g(out.rows(), B);
    for (auto &v : g.data())
        v = dist(rng);
    nn::Matrix g2 = pool.backward(g);
    assert(g2.rows() == 32 && g2.cols() == B);
    nn::Matrix g3 = relu.backward(g2);
    assert(g3.rows() == 32 && g3.cols() == B);
    nn::Matrix g4 = conv.backward(g3);
    assert(g4.rows() == 16 && g4.cols() == B);

    std::puts("  [Conv2D→ReLU→MaxPool2D] composite shapes PASSED");
}

#include "test_runner.h"

/**
 * @brief Runs the Conv2D, MaxPool2D, and composite pipeline tests.
 *
 * @return int Test runner status code.
 */
int main(int argc, char *argv[])
{
    const TestEntry tests[] = {
        {"conv2d_forward_hand",       test_conv2d_forward_hand},
        {"conv2d_im2col_basis",       test_conv2d_im2col_basis},
        {"conv2d_gradient_check",     test_conv2d_gradient_check},
        {"maxpool2d_forward",         test_maxpool2d_forward},
        {"maxpool2d_backward",        test_maxpool2d_backward},
        {"conv_relu_pool_composite",  test_conv_relu_pool_composite},
    };
    return run_tests(argc, argv, tests, sizeof(tests) / sizeof(tests[0]));
}
