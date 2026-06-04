#include <neuralnet/nn/nn.h>
#include <neuralnet/loss.h>
#include <neuralnet/optimizer.h>
#include <neuralnet/lr_scheduler.h>
#include <neuralnet/model/model.h>
#include <neuralnet/model/io.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

// ── 辅助：浮点近似比较 ────────────────────────────────────────────────────────
static bool approx(double a, double b, double tol = 1e-6)
{
    return std::fabs(a - b) < tol;
}

// ══════════════════════════════════════════════════════════════════════════════
// counting_iterator 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_counting_iterator()
{
    std::puts("  [counting_iterator] basic traversal ...");

    nn::counting_iterator<std::size_t> it(5);
    assert(*it == 5);
    ++it;
    assert(*it == 6);
    it += 3;
    assert(*it == 9);
    assert(it - nn::counting_iterator<std::size_t>(5) == 4);

    // 用于 std::for_each 求和
    std::size_t sum = 0;
    std::for_each(nn::counting_iterator<std::size_t>(1),
                  nn::counting_iterator<std::size_t>(6),
                  [&](std::size_t v) { sum += v; });
    assert(sum == 15); // 1+2+3+4+5

    // 非成员 operator+
    auto it2 = 3 + nn::counting_iterator<std::size_t>(10);
    assert(*it2 == 13);

    // operator[]
    assert(it2[2] == 15);

    std::puts("  [counting_iterator] PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// Matrix 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_matrix_construction()
{
    std::puts("  [Matrix] construction & accessors ...");

    nn::Matrix m(3, 4);
    assert(m.rows() == 3);
    assert(m.cols() == 4);
    assert(m.size() == 12);
    assert(!m.empty());

    // 默认零初始化
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < m.cols(); ++j)
            assert(approx(m.at(i, j), 0.0));

    // 标量初始化
    nn::Matrix m2(2, 3, 7.0);
    for (std::size_t i = 0; i < m2.rows(); ++i)
        for (std::size_t j = 0; j < m2.cols(); ++j)
            assert(approx(m2.at(i, j), 7.0));

    // vector 构造
    std::vector<double> v = {1, 2, 3, 4, 5, 6};
    nn::Matrix m3(std::vector<double>(v), 2, 3);
    assert(approx(m3.at(0, 0), 1.0));
    assert(approx(m3.at(1, 2), 6.0));

    // set_value / get_data
    m.set_value(1, 2, 42.0);
    assert(approx(m.at(1, 2), 42.0));
    auto data2d = m.get_data();
    assert(approx(data2d[1][2], 42.0));

    std::puts("  [Matrix] construction & accessors PASSED");
}

static void test_matrix_arithmetic()
{
    std::puts("  [Matrix] arithmetic ...");

    nn::Matrix a(std::vector<double>{1, 2, 3, 4}, 2, 2);
    nn::Matrix b(std::vector<double>{5, 6, 7, 8}, 2, 2);

    // 加法
    auto c = a + b;
    assert(approx(c.at(0, 0), 6.0));
    assert(approx(c.at(1, 1), 12.0));

    // 减法
    auto d = b - a;
    assert(approx(d.at(0, 0), 4.0));
    assert(approx(d.at(1, 1), 4.0));

    // 标量乘法
    auto e = a * 3.0;
    assert(approx(e.at(0, 1), 6.0));
    auto f = 2.0 * a;
    assert(approx(f.at(1, 0), 6.0));

    // 矩阵乘法: 2x3 * 3x2 = 2x2
    nn::Matrix m1(std::vector<double>{1, 2, 3, 4, 5, 6}, 2, 3);
    nn::Matrix m2(std::vector<double>{7, 8, 9, 10, 11, 12}, 3, 2);
    auto m3 = m1 * m2;
    assert(m3.rows() == 2);
    assert(m3.cols() == 2);
    // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    assert(approx(m3.at(0, 0), 58.0));
    assert(approx(m3.at(0, 1), 64.0));
    assert(approx(m3.at(1, 0), 139.0));
    assert(approx(m3.at(1, 1), 154.0));

    // inplace 操作
    nn::Matrix g(std::vector<double>{1, 2, 3, 4}, 2, 2);
    nn::Matrix h(std::vector<double>{10, 20, 30, 40}, 2, 2);
    g.add_inplace(h);
    assert(approx(g.at(0, 0), 11.0));
    g.subtract_inplace(h);
    assert(approx(g.at(0, 0), 1.0));
    g.scale_inplace(2.0);
    assert(approx(g.at(0, 0), 2.0));
    g.zero();
    assert(approx(g.at(1, 1), 0.0));

    std::puts("  [Matrix] arithmetic PASSED");
}

static void test_matrix_transpose()
{
    std::puts("  [Matrix] transpose ...");

    nn::Matrix m(std::vector<double>{1, 2, 3, 4, 5, 6}, 2, 3);
    auto t = m.transpose();
    assert(t.rows() == 3);
    assert(t.cols() == 2);
    assert(approx(t.at(0, 0), 1.0));
    assert(approx(t.at(2, 1), 6.0));
    assert(approx(t.at(1, 0), 2.0));

    // 方阵转置
    nn::Matrix sq(std::vector<double>{1, 2, 3, 4}, 2, 2);
    auto sq_t = sq.transpose();
    assert(approx(sq_t.at(0, 1), 3.0));
    assert(approx(sq_t.at(1, 0), 2.0));

    std::puts("  [Matrix] transpose PASSED");
}

static void test_matrix_multiplication_large()
{
    std::puts("  [Matrix] block multiplication (64x64) ...");

    // 测试 BLOCK_SIZE 边界：两个 64x64 矩阵乘法
    const std::size_t N = 64;
    nn::Matrix a(N, N, 1.0);
    nn::Matrix b(N, N, 2.0);
    auto c = a * b;
    // 1.0 * 2.0 * 64 = 128.0
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            assert(approx(c.at(i, j), 128.0));

    // 非 BLOCK_SIZE 整除: 100x100
    const std::size_t M = 100;
    nn::Matrix d(M, M, 0.5);
    nn::Matrix e(M, M, 3.0);
    auto f = d * e;
    // 0.5 * 3.0 * 100 = 150.0
    assert(approx(f.at(0, 0), 150.0));
    assert(approx(f.at(99, 99), 150.0));

    std::puts("  [Matrix] block multiplication PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// Layer 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_relu()
{
    std::puts("  [ReLU] forward & backward ...");

    nn::ReLU relu;
    nn::Matrix input(std::vector<double>{-2, 0, 3, -1, 4, -5}, 2, 3);
    auto out = relu.forward(input);
    assert(approx(out.at(0, 0), 0.0)); // -2 -> 0
    assert(approx(out.at(0, 1), 0.0)); // 0 -> 0
    assert(approx(out.at(0, 2), 3.0)); // 3 -> 3
    assert(approx(out.at(1, 2), 0.0)); // -5 -> 0
    assert(approx(out.at(1, 1), 4.0)); // 4 -> 4

    // backward: 梯度只通过正数
    nn::Matrix grad(std::vector<double>{1, 2, 3, 4, 5, 6}, 2, 3);
    auto grad_out = relu.backward(grad);
    assert(approx(grad_out.at(0, 0), 0.0)); // input -2, grad blocked
    assert(approx(grad_out.at(0, 2), 3.0)); // input 3, grad passes
    assert(approx(grad_out.at(1, 1), 5.0)); // input 4, grad passes
    assert(approx(grad_out.at(1, 2), 0.0)); // input -5, grad blocked

    std::puts("  [ReLU] forward & backward PASSED");
}

static void test_linear_forward_backward()
{
    std::puts("  [Linear] forward & backward shape ...");

    nn::Linear lin(3, 2); // 3 -> 2
    auto params = lin.parameters();
    assert(params.size() == 2); // W and b

    // forward: input (3,1) -> output (2,1)
    nn::Matrix x(std::vector<double>{1.0, 2.0, 3.0}, 3, 1);
    auto y = lin.forward(x);
    assert(y.rows() == 2);
    assert(y.cols() == 1);

    // backward: grad_output (2,1) -> grad_input (3,1)
    nn::Matrix grad_y(std::vector<double>{1.0, 1.0}, 2, 1);
    auto grad_x = lin.backward(grad_y);
    assert(grad_x.rows() == 3);
    assert(grad_x.cols() == 1);

    // 参数梯度应该非零
    auto param_grads = lin.param_gradients();
    assert(param_grads.size() == 2);
    bool grad_w_has_nonzero = false;
    for (std::size_t i = 0; i < param_grads[0].get().rows(); ++i)
        for (std::size_t j = 0; j < param_grads[0].get().cols(); ++j)
            if (!approx(param_grads[0].get().at(i, j), 0.0))
                grad_w_has_nonzero = true;
    assert(grad_w_has_nonzero);

    std::puts("  [Linear] forward & backward shape PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// Loss 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_mse_loss()
{
    std::puts("  [MSELoss] forward & backward ...");

    nn::MSELoss loss;
    nn::Matrix pred(std::vector<double>{1.0, 2.0, 3.0}, 3, 1);
    nn::Matrix target(std::vector<double>{1.5, 2.5, 3.5}, 3, 1);

    double val = loss.forward(pred, target);
    // MSE = mean((0.5)^2 + (0.5)^2 + (0.5)^2) = 0.25
    assert(approx(val, 0.25));

    auto &grad = loss.backward();
    assert(grad.rows() == 3);
    assert(grad.cols() == 1);
    // grad = 2*(pred-target)/N = 2*(-0.5)/3 = -1/3
    assert(approx(grad.at(0, 0), -1.0 / 3.0, 1e-5));

    std::puts("  [MSELoss] forward & backward PASSED");
}

static void test_cross_entropy_loss()
{
    std::puts("  [CrossEntropyLoss] forward & backward ...");

    nn::CrossEntropyLoss loss;
    // 3 类，batch=1，真实标签 = 类 2
    nn::Matrix logits(std::vector<double>{0.1, 0.2, 0.7}, 3, 1);
    std::vector<std::size_t> labels = {2};
    nn::Matrix target_onehot = nn::one_hot(labels, 3);

    double val = loss.forward(logits, target_onehot);
    assert(val > 0.0); // loss 为正

    auto &grad = loss.backward();
    assert(grad.rows() == 3);
    assert(grad.cols() == 1);
    // 梯度 = softmax(logits) - target_onehot
    // softmax(0.7) ≈ 0.475, 所以 grad[2] ≈ 0.475 - 1.0 = -0.525
    assert(grad.at(2, 0) < 0.0); // 对正确类别梯度应为负
    assert(grad.at(0, 0) > 0.0); // 对错误类别梯度应为正

    std::puts("  [CrossEntropyLoss] forward & backward PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// Optimizer 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_sgd()
{
    std::puts("  [SGD] step & zero_grad ...");

    nn::Matrix w(std::vector<double>{1.0, 2.0, 3.0, 4.0}, 2, 2);
    nn::Matrix g(std::vector<double>{0.1, 0.2, 0.3, 0.4}, 2, 2);

    nn::SGD sgd({std::ref(w)}, {std::ref(g)}, 0.1);
    sgd.step();

    // w_new = w - lr * g
    assert(approx(w.at(0, 0), 1.0 - 0.1 * 0.1));
    assert(approx(w.at(1, 1), 4.0 - 0.1 * 0.4));

    sgd.zero_grad();
    for (std::size_t i = 0; i < g.rows(); ++i)
        for (std::size_t j = 0; j < g.cols(); ++j)
            assert(approx(g.at(i, j), 0.0));

    std::puts("  [SGD] step & zero_grad PASSED");
}

static void test_sgd_momentum()
{
    std::puts("  [SGD_w_Momentum] step ...");

    nn::Matrix w(std::vector<double>{1.0, 2.0}, 2, 1);
    nn::Matrix g(std::vector<double>{1.0, 1.0}, 2, 1);

    nn::SGD_w_Momentum opt({std::ref(w)}, {std::ref(g)}, 1.0, 0.9);

    // 第1步: v = 0.9*0 + 0.1*g = [0.1, 0.1], w = w - 1.0*v = [0.9, 1.9]
    opt.step();
    assert(approx(w.at(0, 0), 0.9, 1e-5));
    assert(approx(w.at(1, 0), 1.9, 1e-5));

    // 第2步: v = 0.9*0.1 + 0.1*1.0 = 0.19, w = 0.9 - 0.19 = 0.71
    opt.step();
    assert(approx(w.at(0, 0), 0.71, 1e-5));

    std::puts("  [SGD_w_Momentum] step PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// Adam 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_adam()
{
    std::puts("  [Adam] step ...");

    nn::Matrix w(std::vector<double>{1.0, 2.0}, 2, 1);
    nn::Matrix g(std::vector<double>{1.0, 1.0}, 2, 1);

    nn::Adam opt({std::ref(w)}, {std::ref(g)}, 0.1); // lr=0.1 方便数值观察

    // 第1步: m1 = 0.1*1 = 0.1, v1 = 0.999*1 = 0.999
    //         m_hat = 0.1/0.1 = 1.0, v_hat = 0.999/0.001 = 999
    //         w = 1.0 - 0.1 * 1.0 / (sqrt(999) + 1e-8)
    opt.step();
    // 只验证 w 下降（梯度为正，参数应减小）
    assert(w.at(0, 0) < 1.0);
    assert(w.at(1, 0) < 2.0);

    opt.zero_grad();
    assert(approx(g.at(0, 0), 0.0));
    assert(approx(g.at(1, 0), 0.0));

    std::puts("  [Adam] step PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// LeakyReLU 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_leaky_relu()
{
    std::puts("  [LeakyReLU] forward & backward ...");

    nn::LeakyReLU relu(0.01);
    nn::Matrix x(std::vector<double>{-2.0, -1.0, 0.0, 1.0, 2.0}, 5, 1);
    auto out = relu.forward(x);

    assert(approx(out.at(0, 0), -0.02)); // -2 * 0.01
    assert(approx(out.at(1, 0), -0.01)); // -1 * 0.01
    assert(approx(out.at(2, 0), 0.0));
    assert(approx(out.at(3, 0), 1.0));
    assert(approx(out.at(4, 0), 2.0));

    // backward: grad=1 对所有元素
    nn::Matrix grad(std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0}, 5, 1);
    auto grad_in = relu.backward(grad);
    assert(approx(grad_in.at(0, 0), 0.01)); // 负区间
    assert(approx(grad_in.at(1, 0), 0.01));
    assert(approx(grad_in.at(2, 0), 0.0));  // x=0 处按 <=0 处理
    assert(approx(grad_in.at(3, 0), 1.0));  // 正区间
    assert(approx(grad_in.at(4, 0), 1.0));

    std::puts("  [LeakyReLU] forward & backward PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// Sigmoid 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_sigmoid()
{
    std::puts("  [Sigmoid] forward & backward ...");

    nn::Sigmoid sig;
    nn::Matrix x(std::vector<double>{0.0, 1.0, -1.0}, 3, 1);
    auto out = sig.forward(x);

    assert(approx(out.at(0, 0), 0.5));
    assert(approx(out.at(1, 0), 1.0 / (1.0 + std::exp(-1.0))));
    assert(approx(out.at(2, 0), 1.0 / (1.0 + std::exp(1.0))));

    // backward at x=0: sigmoid'(0) = 0.5 * 0.5 = 0.25
    nn::Matrix grad(std::vector<double>{1.0, 1.0, 1.0}, 3, 1);
    auto grad_in = sig.backward(grad);
    assert(approx(grad_in.at(0, 0), 0.25));

    std::puts("  [Sigmoid] forward & backward PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// Tanh 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_tanh()
{
    std::puts("  [Tanh] forward & backward ...");

    nn::Tanh t;
    nn::Matrix x(std::vector<double>{0.0, 1.0, -1.0}, 3, 1);
    auto out = t.forward(x);

    assert(approx(out.at(0, 0), 0.0));
    assert(approx(out.at(1, 0), std::tanh(1.0)));
    assert(approx(out.at(2, 0), std::tanh(-1.0)));

    // backward at x=0: tanh'(0) = 1 - 0 = 1
    nn::Matrix grad(std::vector<double>{1.0, 1.0, 1.0}, 3, 1);
    auto grad_in = t.backward(grad);
    assert(approx(grad_in.at(0, 0), 1.0));
    // at x=1: tanh(1) ≈ 0.7616, grad = 1 - 0.7616^2 ≈ 0.42
    double expected = 1.0 - std::tanh(1.0) * std::tanh(1.0);
    assert(approx(grad_in.at(1, 0), expected));

    std::puts("  [Tanh] forward & backward PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// GELU 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_gelu()
{
    std::puts("  [GELU] forward & backward ...");

    nn::GELU gelu;
    nn::Matrix x(std::vector<double>{0.0, 1.0, -1.0, 5.0, -5.0}, 5, 1);
    auto out = gelu.forward(x);

    // GELU(0) ≈ 0
    assert(approx(out.at(0, 0), 0.0, 1e-4));
    // GELU(5) ≈ 5 (大正数趋近自身)
    assert(approx(out.at(3, 0), 5.0, 1e-3));
    // GELU(-5) ≈ 0 (大负数趋近 0)
    assert(approx(out.at(4, 0), 0.0, 1e-3));
    // GELU(1) > 0 且 < 1
    assert(out.at(1, 0) > 0.0 && out.at(1, 0) < 1.0);
    // GELU(-1) < 0 且 > -1
    assert(out.at(2, 0) < 0.0 && out.at(2, 0) > -1.0);

    // backward: GELU'(0) ≈ 0.5
    nn::Matrix grad(std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0}, 5, 1);
    auto grad_in = gelu.backward(grad);
    assert(approx(grad_in.at(0, 0), 0.5, 1e-3));
    // GELU'(5) ≈ 1
    assert(approx(grad_in.at(3, 0), 1.0, 1e-2));
    // GELU'(-5) ≈ 0
    assert(approx(grad_in.at(4, 0), 0.0, 1e-2));

    std::puts("  [GELU] forward & backward PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// Dropout 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_dropout()
{
    std::puts("  [Dropout] forward & backward ...");

    // 推理模式：不丢弃
    nn::Dropout eval_drop(0.5, false);
    nn::Matrix x(std::vector<double>{1.0, 2.0, 3.0, 4.0}, 4, 1);
    auto out_eval = eval_drop.forward(x);
    for (std::size_t i = 0; i < 4; ++i)
        assert(approx(out_eval.at(i, 0), x.at(i, 0)));

    // 训练模式：p=0（不丢弃）
    nn::Dropout zero_drop(0.0, true);
    auto out_zero = zero_drop.forward(x);
    for (std::size_t i = 0; i < 4; ++i)
        assert(approx(out_zero.at(i, 0), x.at(i, 0)));

    // 训练模式：p=0.5，输出均值接近原始（inverted dropout 缩放）
    nn::Dropout train_drop(0.5, true);
    nn::Matrix big_x(1000, 1);
    for (std::size_t i = 0; i < 1000; ++i)
        big_x.set_value_unchecked(i, 0, 1.0);

    auto out_train = train_drop.forward(big_x);
    double sum = 0.0;
    int zeros = 0;
    for (std::size_t i = 0; i < 1000; ++i)
    {
        sum += out_train.at(i, 0);
        if (out_train.at(i, 0) == 0.0)
            ++zeros;
    }
    // 大约 50% 被置零
    assert(zeros > 300 && zeros < 700);
    // 均值应接近 1.0（因为 inverted scaling）
    double mean = sum / 1000.0;
    assert(mean > 0.7 && mean < 1.3);

    std::puts("  [Dropout] forward & backward PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// LR Scheduler 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_step_lr()
{
    std::puts("  [StepLR] step ...");

    nn::StepLR sched(0.1, 3, 0.1); // lr=0.1, step_size=3, gamma=0.1
    assert(approx(sched.get_lr(), 0.1));
    sched.step(); // epoch 1
    assert(approx(sched.get_lr(), 0.1));
    sched.step(); // epoch 2
    assert(approx(sched.get_lr(), 0.1));
    sched.step(); // epoch 3 → 衰减
    assert(approx(sched.get_lr(), 0.01));
    sched.step(); // epoch 4
    assert(approx(sched.get_lr(), 0.01));
    sched.step(); // epoch 5
    assert(approx(sched.get_lr(), 0.01));
    sched.step(); // epoch 6 → 第二次衰减
    assert(approx(sched.get_lr(), 0.001));

    std::puts("  [StepLR] step PASSED");
}

static void test_cosine_lr()
{
    std::puts("  [CosineAnnealingLR] step ...");

    nn::CosineAnnealingLR sched(1.0, 4, 0.0); // lr_max=1.0, T_max=4, lr_min=0
    assert(approx(sched.get_lr(), 1.0)); // epoch 0: cos(0) = 1 → 1.0

    sched.step(); // epoch 1: cos(π/4) ≈ 0.707 → 0.854
    double lr1 = sched.get_lr();
    assert(lr1 < 1.0 && lr1 > 0.0);

    sched.step(); // epoch 2: cos(π/2) = 0 → 0.5
    double lr2 = sched.get_lr();
    assert(approx(lr2, 0.5, 1e-6));

    sched.step(); // epoch 3: cos(3π/4) ≈ -0.707 → 0.146
    double lr3 = sched.get_lr();
    assert(lr3 > 0.0 && lr3 < lr2);

    sched.step(); // epoch 4: cos(π) = -1 → 0.0
    double lr4 = sched.get_lr();
    assert(approx(lr4, 0.0, 1e-6));

    std::puts("  [CosineAnnealingLR] step PASSED");
}

static void test_exp_lr()
{
    std::puts("  [ExponentialLR] step ...");

    nn::ExponentialLR sched(1.0, 0.5); // lr=1.0, gamma=0.5
    assert(approx(sched.get_lr(), 1.0));

    sched.step(); // epoch 1: 1.0 * 0.5^1 = 0.5
    assert(approx(sched.get_lr(), 0.5));

    sched.step(); // epoch 2: 1.0 * 0.5^2 = 0.25
    assert(approx(sched.get_lr(), 0.25));

    sched.step(); // epoch 3: 1.0 * 0.5^3 = 0.125
    assert(approx(sched.get_lr(), 0.125));

    std::puts("  [ExponentialLR] step PASSED");
}
// ══════════════════════════════════════════════════════════════════════════════
// Model 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_model_forward()
{
    std::puts("  [Model] forward ...");

    nn::Model model;
    model.add<nn::Linear>(4, 3)
         .add<nn::ReLU>()
         .add<nn::Linear>(3, 2);

    nn::Matrix x(4, 1, 1.0);
    auto y = model.forward(x);
    assert(y.rows() == 2);
    assert(y.cols() == 1);

    // backward 应该能跑通
    nn::Matrix grad_y(2, 1, 1.0);
    model.backward(grad_y);

    std::puts("  [Model] forward PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// Model IO 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_model_io()
{
    std::puts("  [Model IO] save & load round-trip ...");

    // 构造一个小网络并保存
    nn::Linear l1(4, 3);
    nn::Linear l2(3, 2);
    nn::Linear l3(2, 2);
    nn::Linear l4(2, 1);

    const std::string path = "/tmp/nn_test_model.bin";

    // 记录保存前的参数
    auto w1_before = l1.parameters()[0].get().data();

    nn::save_model(path, l1, l2, l3, l4);

    // 手动修改参数以验证加载确实生效
    l1.parameters()[0].get().set_value(0, 0, 999.0);
    assert(approx(l1.parameters()[0].get().at(0, 0), 999.0));

    // 加载
    nn::load_model(path, l1, l2, l3, l4);

    // 验证加载后参数恢复
    assert(approx(l1.parameters()[0].get().at(0, 0), w1_before[0]));

    // Model 版本的 save/load
    nn::Model model;
    model.add<nn::Linear>(3, 2)
         .add<nn::ReLU>()
         .add<nn::Linear>(2, 1);

    const std::string path2 = "/tmp/nn_test_model2.bin";
    nn::save_model(path2, model);

    // 修改参数
    model.parameters()[0].get().set_value(0, 0, -1234.0);

    nn::load_model(path2, model);
    // 加载后应恢复（非 -1234）
    assert(!approx(model.parameters()[0].get().at(0, 0), -1234.0));

    // 清理临时文件
    std::remove(path.c_str());
    std::remove(path2.c_str());

    std::puts("  [Model IO] save & load round-trip PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// one_hot 测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_one_hot()
{
    std::puts("  [one_hot] ...");

    std::vector<std::size_t> labels = {0, 2, 1};
    auto m = nn::one_hot(labels, 3);
    assert(m.rows() == 3);
    assert(m.cols() == 3);
    assert(approx(m.at(0, 0), 1.0));
    assert(approx(m.at(2, 1), 1.0));
    assert(approx(m.at(1, 2), 1.0));
    // 其他位置为零
    assert(approx(m.at(1, 0), 0.0));
    assert(approx(m.at(0, 1), 0.0));

    std::puts("  [one_hot] PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// end-to-end 微型训练测试
// ══════════════════════════════════════════════════════════════════════════════
static void test_end_to_end_train()
{
    std::puts("  [e2e] tiny training loop ...");

    // 简单问题：2个输入，2个输出，1个样本
    nn::Model model;
    model.add<nn::Linear>(2, 4)
         .add<nn::ReLU>()
         .add<nn::Linear>(4, 2);

    nn::MSELoss loss_fn;
    nn::SGD optimizer(model.parameters(), model.param_gradients(), 0.01);

    nn::Matrix x(std::vector<double>{1.0, 0.0}, 2, 1);
    nn::Matrix target(std::vector<double>{0.0, 1.0}, 2, 1);

    double loss_before = loss_fn.forward(model.forward(x), target);

    // 训练 100 步
    for (int i = 0; i < 100; ++i)
    {
        auto pred = model.forward(x);
        (void)loss_fn.forward(pred, target);
        model.backward(loss_fn.backward());
        optimizer.step();
        optimizer.zero_grad();
    }

    double loss_after = loss_fn.forward(model.forward(x), target);
    // loss 应该下降
    assert(loss_after < loss_before);

    std::puts("  [e2e] tiny training loop PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// Matrix::set_data 测试
// ══════════════════════════════════════════════════════════════════════════════

// 场景 1：尺寸匹配 → 静默填充（不应抛、不应 resize）
static void test_set_data_match()
{
    std::puts("  [set_data] match: dimensions match, silent copy ...");

    nn::Matrix m(2, 3);
    m.set_data({{1, 2, 3},
                {4, 5, 6}});

    assert(m.rows() == 2);
    assert(m.cols() == 3);
    assert(approx(m.at(0, 0), 1.0));
    assert(approx(m.at(0, 2), 3.0));
    assert(approx(m.at(1, 0), 4.0));
    assert(approx(m.at(1, 2), 6.0));

    std::puts("  [set_data] match PASSED");
}

// 场景 2：尺寸不匹配（smaller matrix ← larger data）→ 警告 + 自动 resize
static void test_set_data_resize_grow()
{
    std::puts("  [set_data] resize grow: 2x3 -> 3x4, warn + auto-resize ...");

    nn::Matrix m(2, 3);
    m.set_data({{1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12}});

    assert(m.rows() == 3);
    assert(m.cols() == 4);
    assert(approx(m.at(0, 0), 1.0));
    assert(approx(m.at(0, 3), 4.0));
    assert(approx(m.at(2, 0), 9.0));
    assert(approx(m.at(2, 3), 12.0));

    std::puts("  [set_data] resize grow PASSED");
}

// 场景 3：尺寸不匹配（larger matrix ← smaller data）→ 警告 + 自动 resize
static void test_set_data_resize_shrink()
{
    std::puts("  [set_data] resize shrink: 4x4 -> 2x2, warn + auto-resize ...");

    nn::Matrix m(4, 4, 7.0);  // 初始填 7
    m.set_data({{10, 20},
                {30, 40}});

    assert(m.rows() == 2);
    assert(m.cols() == 2);
    assert(approx(m.at(0, 0), 10.0));
    assert(approx(m.at(0, 1), 20.0));
    assert(approx(m.at(1, 0), 30.0));
    assert(approx(m.at(1, 1), 40.0));

    std::puts("  [set_data] resize shrink PASSED");
}

// 场景 4：默认构造的 0x0 矩阵 → 警告 + 自动 resize 到非零
static void test_set_data_from_empty()
{
    std::puts("  [set_data] from empty: 0x0 default -> 2x3, warn + auto-resize ...");

    nn::Matrix m;  // 默认 0x0
    assert(m.rows() == 0);
    assert(m.cols() == 0);
    assert(m.empty());

    m.set_data({{1, 2, 3},
                {4, 5, 6}});

    assert(m.rows() == 2);
    assert(m.cols() == 3);
    assert(!m.empty());
    assert(approx(m.at(0, 0), 1.0));
    assert(approx(m.at(1, 2), 6.0));

    std::puts("  [set_data] from empty PASSED");
}

// 场景 5：空 new_data → 抛 std::invalid_argument
static void test_set_data_empty_throws()
{
    std::puts("  [set_data] empty: empty new_data throws invalid_argument ...");

    nn::Matrix m(2, 2);
    bool threw = false;
    try
    {
        m.set_data({});
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    catch (...)
    {
        // 不应该走到这里
    }
    assert(threw);

    std::puts("  [set_data] empty throws PASSED");
}

// 场景 6：行宽不一致 → 抛 std::invalid_argument
static void test_set_data_inconsistent_throws()
{
    std::puts("  [set_data] inconsistent: row widths differ throws invalid_argument ...");

    nn::Matrix m(2, 2);
    bool threw = false;
    try
    {
        m.set_data({{1, 2, 3},
                    {4, 5}});
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    catch (...)
    {
    }
    assert(threw);

    std::puts("  [set_data] inconsistent throws PASSED");
}

// 场景 7：异常安全 — 抛错后矩阵状态完全保持
static void test_set_data_exception_safety()
{
    std::puts("  [set_data] exception safety: state preserved after throw ...");

    nn::Matrix m(2, 3);
    m.set_data({{10, 20, 30},
                {40, 50, 60}});

    // 故意触发异常
    try
    {
        m.set_data({});
    }
    catch (...)
    {
    }

    // shape 必须不变
    assert(m.rows() == 2);
    assert(m.cols() == 3);

    // 数据必须不变
    assert(approx(m.at(0, 0), 10.0));
    assert(approx(m.at(0, 1), 20.0));
    assert(approx(m.at(0, 2), 30.0));
    assert(approx(m.at(1, 0), 40.0));
    assert(approx(m.at(1, 1), 50.0));
    assert(approx(m.at(1, 2), 60.0));

    // 再触发一次行宽不一致的异常
    try
    {
        m.set_data({{1, 2}, {3, 4, 5}});
    }
    catch (...)
    {
    }

    // 状态仍然不变
    assert(m.rows() == 2);
    assert(m.cols() == 3);
    assert(approx(m.at(0, 0), 10.0));
    assert(approx(m.at(1, 2), 60.0));

    std::puts("  [set_data] exception safety PASSED");
}

// ── 测试注册表 ──────────────────────────────────────────────────────────────
struct test_entry
{
    const char *name;
    void (*fn)();
};

static const test_entry all_tests[] = {
    {"counting_iterator",       test_counting_iterator},
    {"matrix_construction",     test_matrix_construction},
    {"matrix_arithmetic",       test_matrix_arithmetic},
    {"matrix_transpose",        test_matrix_transpose},
    {"matrix_multiplication",   test_matrix_multiplication_large},
    {"relu",                    test_relu},
    {"leaky_relu",              test_leaky_relu},
    {"sigmoid",                 test_sigmoid},
    {"tanh",                    test_tanh},
    {"gelu",                    test_gelu},
    {"dropout",                 test_dropout},
    {"linear_forward_backward", test_linear_forward_backward},
    {"mse_loss",                test_mse_loss},
    {"cross_entropy_loss",      test_cross_entropy_loss},
    {"sgd",                     test_sgd},
    {"sgd_momentum",            test_sgd_momentum},
    {"adam",                    test_adam},
    {"step_lr",                 test_step_lr},
    {"cosine_lr",               test_cosine_lr},
    {"exp_lr",                  test_exp_lr},
    {"model_forward",           test_model_forward},
    {"model_io",                test_model_io},
    {"one_hot",                 test_one_hot},
    {"e2e_train",               test_end_to_end_train},
    {"set_data_match",          test_set_data_match},
    {"set_data_resize_grow",    test_set_data_resize_grow},
    {"set_data_resize_shrink",  test_set_data_resize_shrink},
    {"set_data_from_empty",     test_set_data_from_empty},
    {"set_data_empty_throws",   test_set_data_empty_throws},
    {"set_data_inconsistent_throws", test_set_data_inconsistent_throws},
    {"set_data_exception_safety",    test_set_data_exception_safety},
};

static constexpr std::size_t num_tests = sizeof(all_tests) / sizeof(all_tests[0]);

// ══════════════════════════════════════════════════════════════════════════════
// main
// ══════════════════════════════════════════════════════════════════════════════
int main(int argc, char *argv[])
{
    // 用法：
    //   test_nn            → 运行全部测试
    //   test_nn <name>     → 只运行指定测试
    //   test_nn --list     → 列出所有测试名
    if (argc >= 2 && std::string(argv[1]) == "--list")
    {
        for (std::size_t i = 0; i < num_tests; ++i)
            std::printf("%s\n", all_tests[i].name);
        return 0;
    }

    if (argc >= 2)
    {
        // 运行指定测试
        bool found = false;
        for (std::size_t i = 0; i < num_tests; ++i)
        {
            if (std::string(argv[1]) == all_tests[i].name)
            {
                all_tests[i].fn();
                found = true;
                break;
            }
        }
        if (!found)
        {
            std::fprintf(stderr, "Unknown test: %s\n", argv[1]);
            return 1;
        }
        return 0;
    }

    // 运行全部测试
    std::puts("=== nn test suite ===\n");
    for (std::size_t i = 0; i < num_tests; ++i)
        all_tests[i].fn();

    std::puts("\n=== ALL TESTS PASSED ===");
    return 0;
}
