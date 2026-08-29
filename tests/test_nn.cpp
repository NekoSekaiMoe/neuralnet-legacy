#include <neuralnet/nn/nn.h>
#include <neuralnet/loss.h>
#include <neuralnet/optimizer.h>
#include <neuralnet/lr_scheduler.h>
#include <neuralnet/model/model.h>
#include <neuralnet/model/io.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// ── 辅助：浮点近似比较 ────────────────────────────────────────────────────────
static bool approx(double a, double b, double tol = 1e-6)
{
    return std::fabs(a - b) < tol;
}

// ── 辅助：有限差分梯度（中心差分） ─────────────────────────────────────────────
// 用于 BatchNorm 数值梯度校验：对 layer.parameters()[param_idx] 的
// (row, col) 元素施加 ±eps 扰动，forward 后比较 Σ out^2。
// 解析路径调用方传入 grad_output = 2*out（对应 L = Σ out^2，dL/dout = 2*out），
// 所以 FD 必须用同一个 L = Σ out^2，否则 analytical 与 fd 比较的不是同一个量。
static double finite_diff_grad(nn::Layer &layer, const nn::Matrix &input,
                               std::size_t param_idx, std::size_t row,
                               std::size_t col, double eps = 1e-5)
{
    auto params = layer.parameters();
    nn::Matrix &param = params[param_idx].get();
    const double original = param.at_unchecked(row, col);

    auto sum_sq = [](const nn::Matrix &m) noexcept
    {
        double s = 0.0;
        for (std::size_t k = 0; k < m.size(); ++k)
            s += m.data()[k] * m.data()[k];
        return s;
    };

    param.set_value_unchecked(row, col, original + eps);
    const double loss_plus = sum_sq(layer.forward(input));

    param.set_value_unchecked(row, col, original - eps);
    const double loss_minus = sum_sq(layer.forward(input));

    param.set_value_unchecked(row, col, original); // 还原
    return (loss_plus - loss_minus) / (2.0 * eps);
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

// matmul_NT / matmul_TN 是 Linear::backward 的快路径：与朴素 (transpose + matmul)
// 必须产生相同结果。覆盖 BLOCK_SIZE 边界（64）+ 非整除尺寸 + 极小矩阵。
static void test_matrix_matmul_nt_tn()
{
    std::puts("  [Matrix] matmul_NT / matmul_TN equivalence ...");

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    auto rand_mat = [&](std::size_t r, std::size_t c)
    {
        nn::Matrix m(r, c);
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < c; ++j)
                m.set_value_unchecked(i, j, dist(rng));
        return m;
    };
    auto max_diff = [](const nn::Matrix &a, const nn::Matrix &b)
    {
        double d = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
            d = std::max(d, std::fabs(a.data()[i] - b.data()[i]));
        return d;
    };

    const std::tuple<std::size_t, std::size_t, std::size_t> shapes[] = {
        {3, 4, 5}, {64, 64, 64}, {65, 130, 33}, {1, 1, 1}, {7, 1, 9}, {128, 200, 50},
    };
    for (const auto &[M, K, N] : shapes)
    {
        const nn::Matrix A_MK = rand_mat(M, K);
        const nn::Matrix B_NK = rand_mat(N, K);   // for A * B^T
        const nn::Matrix A_KM = rand_mat(K, M);   // for A^T * B
        const nn::Matrix B_KN = rand_mat(K, N);

        const nn::Matrix want_nt = A_MK * B_NK.transpose();
        const nn::Matrix got_nt = A_MK.matmul_NT(B_NK);
        assert(max_diff(want_nt, got_nt) < 1e-9);

        const nn::Matrix want_tn = A_KM.transpose() * B_KN;
        const nn::Matrix got_tn = A_KM.matmul_TN(B_KN);
        assert(max_diff(want_tn, got_tn) < 1e-9);
    }

    std::puts("  [Matrix] matmul_NT / matmul_TN equivalence PASSED");
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

static void test_linear_backward_blocked_matmul()
{
    std::puts("  [Linear::backward] blocked matmul — gradient norm check on 3-layer model ...");

    nn::Model model;
    model.add<nn::Linear>(10, 8)
         .add<nn::ReLU>()
         .add<nn::Linear>(8, 4);

    nn::Matrix input(10, 4);
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input.data()[i] = static_cast<double>(i % 5) * 0.1;
    }

    auto out = model.forward(input);
    nn::Matrix grad_out(4, 4);
    for (std::size_t i = 0; i < grad_out.size(); ++i) grad_out.data()[i] = 0.25;
    model.backward(grad_out);

    bool all_finite = true;
    bool any_nonzero = false;
    for (auto& g_ref : model.param_gradients())
    {
        for (double v : g_ref.get().data())
        {
            if (!std::isfinite(v)) { all_finite = false; break; }
            if (v != 0.0) any_nonzero = true;
        }
        if (!all_finite) break;
    }
    assert(all_finite);
    assert(any_nonzero);

    std::puts("  [Linear::backward] blocked matmul PASSED");
}

static void test_linear_backward_grad_b_nonsquare()
{
    std::puts("  [Linear::backward] grad_b correctness: out_feat(8) != batch(3) ...");

    nn::Linear lin(5, 8);
    nn::Matrix input(5, 3);
    for (std::size_t i = 0; i < input.size(); ++i)
        input.data()[i] = static_cast<double>(i) * 0.1;

    auto out = lin.forward(input);
    assert(out.rows() == 8);
    assert(out.cols() == 3);

    nn::Matrix grad_output(8, 3);
    for (std::size_t i = 0; i < grad_output.size(); ++i)
        grad_output.data()[i] = static_cast<double>(i % 7) * 0.2;

    lin.backward(grad_output);

    auto param_grads = lin.param_gradients();
    const nn::Matrix &grad_b = param_grads[1].get();
    assert(grad_b.rows() == 8);
    assert(grad_b.cols() == 1);

    for (std::size_t i = 0; i < 8; ++i)
    {
        double expected = 0.0;
        for (std::size_t j = 0; j < 3; ++j)
            expected += grad_output.at_unchecked(i, j);
        assert(approx(grad_b.at_unchecked(i, 0), expected, 1e-12));
    }

    std::puts("  [Linear::backward] grad_b nonsquare PASSED");
}
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

/**
 * @brief Verifies CrossEntropyLoss forward values, backward gradients, batching semantics, and numerical stability.
 */
static void test_cross_entropy_loss()
{
    std::puts("  [CrossEntropyLoss] forward & backward ...");

    nn::CrossEntropyLoss loss;
    {
        bool threw = false;
        try
        {
            (void)loss.forward(nn::Matrix(3, 0), nn::Matrix(3, 0));
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        assert(threw);
    }

    // 3 类，batch=1，真实标签 = 类 2
    nn::Matrix logits(std::vector<double>{0.1, 0.2, 0.7}, 3, 1);
    std::vector<std::size_t> labels = {2};
    nn::Matrix target_onehot = nn::one_hot(labels, 3);

    double val = loss.forward(logits, target_onehot);
    assert(val > 0.0); // loss 为正
    // 精确数值：-log softmax[2] = -(0.7 - max - log(Σ exp(l-max)))
    {
        const double m = 0.7;
        const double s = std::exp(0.1 - m) + std::exp(0.2 - m) + std::exp(0.7 - m);
        const double expect = -(0.7 - m - std::log(s));
        assert(std::abs(val - expect) < 1e-12);
    }

    auto &grad = loss.backward();
    assert(grad.rows() == 3);
    assert(grad.cols() == 1);
    // 梯度 = (softmax(logits) - target_onehot) / batch
    assert(grad.at(2, 0) < 0.0); // 对正确类别梯度应为负
    assert(grad.at(0, 0) > 0.0); // 对错误类别梯度应为正

    // ── 均值梯度语义：grad == (softmax - target) / batch，与手算 softmax 对齐 ──
    {
        nn::Matrix lg2(std::vector<double>{0.1, 0.2, 0.7,
                                           0.3, -0.4, 1.2}, 3, 2);
        std::vector<std::size_t> lab2 = {2, 0};
        nn::Matrix tgt2 = nn::one_hot(lab2, 3);
        nn::CrossEntropyLoss l2;
        (void)l2.forward(lg2, tgt2);
        const auto &g2 = l2.backward();
        for (std::size_t i = 0; i < 2; ++i)
        {
            double m = lg2.at(0, i);
            for (std::size_t c = 1; c < 3; ++c)
                m = std::max(m, lg2.at(c, i));
            double s = 0.0;
            for (std::size_t c = 0; c < 3; ++c)
                s += std::exp(lg2.at(c, i) - m);
            for (std::size_t c = 0; c < 3; ++c)
            {
                const double sm = std::exp(lg2.at(c, i) - m) / s;
                const double expect = (sm - tgt2.at(c, i)) / 2.0; // batch=2
                assert(std::abs(g2.at(c, i) - expect) < 1e-12);
            }
        }
    }

    // ── 类数 > 128 回归：旧实现 std::array<double,128> 栈越界写 ──
    {
        const std::size_t C = 8796, B = 4; // GPT 例程实际 vocab 规模
        nn::Matrix big(C, B);
        for (std::size_t i = 0; i < B; ++i)
            big.set_value_unchecked(i * 1973 % C, i, 3.0);
        std::vector<std::size_t> lab = {500, 1, 8000, 8795};
        nn::Matrix tgt = nn::one_hot(lab, C);
        nn::CrossEntropyLoss l3;
        const double lv3 = l3.forward(big, tgt);
        assert(std::isfinite(lv3));
        const auto &g3 = l3.backward();
        assert(g3.rows() == C && g3.cols() == B);
        for (std::size_t i = 0; i < B; ++i)
        {
            double sum = 0.0;
            for (std::size_t c = 0; c < C; ++c)
                sum += g3.at(c, i);
            assert(std::abs(sum) < 1e-9); // softmax 梯度每列和为 0
        }
    }

    // ── 极负 logits 稳定性：旧形式 log(softmax)→-inf，与 target=0 相乘得 NaN ──
    {
        nn::Matrix lg(std::vector<double>{-800.0, 0.0}, 2, 1);
        nn::Matrix tgt = nn::one_hot({1}, 2);
        nn::CrossEntropyLoss l4;
        const double lv = l4.forward(lg, tgt);
        assert(std::isfinite(lv));
        const auto &g4 = l4.backward();
        assert(std::isfinite(g4.at(0, 0)));
        assert(std::abs(g4.at(0, 0)) < 1e-100); // softmax[-800] 下溢为 0，梯度 ≈ 0，无 NaN
        assert(std::abs(g4.at(1, 0)) < 1e-12);   // 正确类别 softmax ≈ 1，梯度 ≈ 0
    }

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
/**
 * @brief Verifies Adam parameter updates and gradient clearing.
 */
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

/**
 * @brief Verifies AdamW decoupled weight decay behavior.
 *
 * Tests equivalence to Adam when weight decay is zero, pure decay with zero
 * gradients, and the expected order of weight decay and the Adam update.
 */
static void test_adamw()
{
    std::puts("  [AdamW] decoupled weight decay ...");

    // 1) wd=0 时与 Adam 逐步 bit-exact 一致
    {
        nn::Matrix w1(std::vector<double>{1.0, -2.0, 0.5}, 3, 1);
        nn::Matrix w2 = w1;
        nn::Matrix g(std::vector<double>{0.3, -0.7, 0.0}, 3, 1);

        nn::Adam opt_a({std::ref(w1)}, {std::ref(g)}, 0.05);
        nn::AdamW opt_w({std::ref(w2)}, {std::ref(g)}, 0.05, 0.9, 0.999, 1e-8, /*wd=*/0.0);
        for (int s = 0; s < 10; ++s)
        {
            opt_a.step();
            opt_w.step();
        }
        for (std::size_t r = 0; r < 3; ++r)
            assert(w1.at(r, 0) == w2.at(r, 0));
    }

    // 2) 零梯度纯衰减：p_t = p_0 * (1-lr*wd)^t（衰减与 Adam 增量解耦的直接证据）
    {
        nn::Matrix w(std::vector<double>{2.0, -1.0}, 2, 1);
        nn::Matrix g(2, 1); // 全零梯度
        const double lr = 0.1, wd = 0.5;
        nn::AdamW opt({std::ref(w)}, {std::ref(g)}, lr, 0.9, 0.999, 1e-8, wd);
        for (int s = 0; s < 8; ++s)
            opt.step();
        const double expect = 2.0 * std::pow(1.0 - lr * wd, 8.0);
        assert(std::abs(w.at(0, 0) - expect) < 1e-15);
        assert(std::abs(w.at(1, 0) + 0.5 * expect) < 1e-15);
    }

    // 3) 一步手算：先 p*=(1-lr*wd) 再 Adam 增量（增量只依赖 g，不受 wd 影响）
    {
        nn::Matrix w(std::vector<double>{1.0}, 1, 1);
        nn::Matrix g(std::vector<double>{1.0}, 1, 1);
        nn::AdamW opt({std::ref(w)}, {std::ref(g)}, 0.1, 0.9, 0.999, 1e-8, 0.1);
        opt.step();
        // decay=0.99；m_hat=1, v_hat=1 → p = 0.99 - 0.1/(1+1e-8)
        const double expect = 0.99 - 0.1 / (1.0 + 1e-8);
        assert(std::abs(w.at(0, 0) - expect) < 1e-15);
    }

    std::puts("  [AdamW] decoupled weight decay PASSED");
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
    assert(approx(grad_in.at(2, 0), 0.01)); // x=0 处按 <=0 处理 → 梯度=negative_slope
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
    nn::Model model;
    model.add<nn::Linear>(3, 2)
         .add<nn::ReLU>()
         .add<nn::Linear>(2, 1);

    const std::string path = "/tmp/nn_test_model.bin";

    // 记录保存前的参数
    auto w_before = model.parameters()[0].get().data();

    // save 走成员函数，load 走自由函数：这样一次性验证了
    // 1) member save 写出可读格式  2) 自由函数 load_model 正确委托给 member load
    model.save(path);

    // 手动修改参数以验证加载确实生效
    model.parameters()[0].get().set_value(0, 0, -1234.0);
    assert(approx(model.parameters()[0].get().at(0, 0), -1234.0));

    nn::load_model(path, model);

    assert(approx(model.parameters()[0].get().at(0, 0), w_before[0]));

    // 清理临时文件
    std::remove(path.c_str());

    std::puts("  [Model IO] save & load round-trip PASSED");
}

static void test_model_io_member()
{
    std::puts("  [Model IO] member save/load: large net, file-format compat ...");

    nn::Model model;
    model.add<nn::Linear>(4, 3)
         .add<nn::ReLU>()
         .add<nn::Linear>(3, 3)
         .add<nn::ReLU>()
         .add<nn::Linear>(3, 2);

    const std::string path = "/tmp/nn_test_model_member.bin";

    std::vector<double> snapshot;
    for (auto &p : model.parameters())
    {
        auto data = p.get().data();
        snapshot.insert(snapshot.end(), data.begin(), data.end());
    }

    model.save(path);

    for (auto &p : model.parameters())
    {
        p.get().set_value(0, 0, 7777.0);
    }

    model.load(path);

    std::size_t idx = 0;
    for (auto &p : model.parameters())
    {
        auto data = p.get().data();
        for (std::size_t k = 0; k < data.size(); ++k, ++idx)
        {
            assert(approx(data[k], snapshot[idx]));
        }
    }

    std::remove(path.c_str());

    std::puts("  [Model IO] member save/load PASSED");
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

// ── Matrix::set_data ──
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

static void test_set_data_resize_shrink()
{
    std::puts("  [set_data] resize shrink: 4x4 -> 2x2, warn + auto-resize ...");

    nn::Matrix m(4, 4, 7.0);  // 用非零值填充，便于检测 resize 后旧值是否被丢弃
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

static void test_set_data_from_empty()
{
    std::puts("  [set_data] from empty: 0x0 default -> 2x3, warn + auto-resize ...");

    nn::Matrix m;
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

static void test_set_data_empty_throws()
{
    std::puts("  [set_data] empty: empty new_data throws invalid_argument ...");

    nn::Matrix m(2, 2);
    try
    {
        m.set_data({});
    }
    catch (const std::invalid_argument &)
    {
        std::puts("  [set_data] empty throws PASSED");
        return;
    }
    std::puts("  [set_data] empty: expected throw, none happened ... FAILED");
    std::abort();
}

static void test_set_data_inconsistent_throws()
{
    std::puts("  [set_data] inconsistent: row widths differ throws invalid_argument ...");

    nn::Matrix m(2, 2);
    try
    {
        m.set_data({{1, 2, 3},
                    {4, 5}});
    }
    catch (const std::invalid_argument &)
    {
        std::puts("  [set_data] inconsistent throws PASSED");
        return;
    }
    std::puts("  [set_data] inconsistent: expected throw, none happened ... FAILED");
    std::abort();
}

static void test_set_data_exception_safety()
{
    std::puts("  [set_data] exception safety: state preserved after throw ...");

    nn::Matrix m(2, 3);
    m.set_data({{10, 20, 30},
                {40, 50, 60}});

    try
    {
        m.set_data({});
    }
    catch (...)
    {
    }
    assert(m.rows() == 2);
    assert(m.cols() == 3);
    assert(approx(m.at(0, 0), 10.0));
    assert(approx(m.at(1, 2), 60.0));

    try
    {
        m.set_data({{1, 2}, {3, 4, 5}});
    }
    catch (...)
    {
    }
    assert(m.rows() == 2);
    assert(m.cols() == 3);
    assert(approx(m.at(0, 0), 10.0));
    assert(approx(m.at(1, 2), 60.0));

    std::puts("  [set_data] exception safety PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// Matrix 规约操作测试 (F8)
// ══════════════════════════════════════════════════════════════════════════════
static void test_matrix_norm()
{
    std::puts("  [Matrix] norm ...");

    nn::Matrix v(std::vector<double>{3.0, 4.0}, 1, 2);
    assert(approx(v.norm(), 5.0, 1e-9));

    nn::Matrix z(3, 4);
    assert(approx(z.norm(), 0.0));

    nn::Matrix o(3, 4, 1.0);
    assert(approx(o.norm(), std::sqrt(12.0), 1e-9));

    nn::Matrix m(std::vector<double>{1.0, 2.0, 3.0, 4.0}, 2, 2);
    assert(approx(m.sum(), 10.0));
    assert(approx(m.mean(), 2.5));
    assert(approx(nn::Matrix().sum(), 0.0));
    assert(approx(nn::Matrix().mean(), 0.0));

    std::puts("  [Matrix] norm PASSED");
}

static void test_matrix_colwise_mean()
{
    std::puts("  [Matrix] colwise_mean ...");

    // 3x2 行主序：col 0 = {1,2,3}, col 1 = {4,5,6}  ⟹  data = {1,4, 2,5, 3,6}
    nn::Matrix m(std::vector<double>{1.0, 4.0, 2.0, 5.0, 3.0, 6.0}, 3, 2);
    auto means = m.colwise_mean();
    assert(means.size() == 2);
    assert(approx(means[0], 2.0, 1e-9));
    assert(approx(means[1], 5.0, 1e-9));

    nn::Matrix single(std::vector<double>{7.0}, 1, 1);
    auto single_means = single.colwise_mean();
    assert(single_means.size() == 1);
    assert(approx(single_means[0], 7.0));

    auto sums = m.colwise_sum();
    assert(approx(sums[0], 6.0));
    assert(approx(sums[1], 15.0));

    // col 0 = {1,2,3}，mean=2, var = ((1-2)^2 + 0 + (3-2)^2)/(3-1) = 1
    auto vars = m.colwise_var(means);
    assert(approx(vars[0], 1.0, 1e-9));
    assert(approx(vars[1], 1.0, 1e-9));

    std::puts("  [Matrix] colwise_mean PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// 模型模式切换 + 持久化测试 (F6/F5)
// ══════════════════════════════════════════════════════════════════════════════

namespace
{
    class CountingLayer : public nn::Layer
    {
    public:
        inline static int train_calls = 0;
        inline static int eval_calls = 0;

        nn::Matrix forward(const nn::Matrix &input) override
        {
            return nn::Matrix(input.rows(), input.cols());
        }
        nn::Matrix backward(const nn::Matrix &grad_output) override
        {
            return grad_output;
        }
        std::vector<std::reference_wrapper<nn::Matrix>> parameters() override
        {
            return {};
        }
        std::vector<std::reference_wrapper<nn::Matrix>> param_gradients() override
        {
            return {};
        }

        void on_mode_change(bool training) override
        {
            if (training) ++train_calls;
            else ++eval_calls;
        }
    };

    // 暴露 1 个参数的 Layer：v1 文件无 n_params 段，
    // 加载端 parameters().size() 决定读取数量，故用 1 参数层做最小化测试。
    class SingleParamLayer : public nn::Layer
    {
    public:
        nn::Matrix W;

        explicit SingleParamLayer(std::size_t rows, std::size_t cols) : W(rows, cols) {}

        nn::Matrix forward(const nn::Matrix &input) override
        {
            return input;
        }
        nn::Matrix backward(const nn::Matrix &grad_output) override
        {
            return grad_output;
        }
        std::vector<std::reference_wrapper<nn::Matrix>> parameters() override
        {
            return {std::ref(W)};
        }
        std::vector<std::reference_wrapper<nn::Matrix>> param_gradients() override
        {
            return {};
        }
    };
} // namespace

static void test_model_default_is_training()
{
    std::puts("  [Model] default is_training is true ...");

    nn::Model model;
    assert(model.is_training() == true);

    std::puts("  [Model] default is_training PASSED");
}

static void test_model_train_eval()
{
    std::puts("  [Model] train()/eval() toggle is_training ...");

    nn::Model model;
    model.add<nn::Linear>(2, 2);

    model.train();
    assert(model.is_training() == true);

    model.eval();
    assert(model.is_training() == false);

    model.train();
    assert(model.is_training() == true);

    std::puts("  [Model] train/eval PASSED");
}

static void test_on_mode_change_called()
{
    std::puts("  [Model] on_mode_change propagates to layers ...");

    CountingLayer::train_calls = 0;
    CountingLayer::eval_calls = 0;

    nn::Model model;
    model.add<CountingLayer>();

    model.train();
    assert(CountingLayer::train_calls == 1);
    assert(CountingLayer::eval_calls == 0);

    model.eval();
    assert(CountingLayer::train_calls == 1);
    assert(CountingLayer::eval_calls == 1);

    model.train();
    assert(CountingLayer::train_calls == 2);
    assert(CountingLayer::eval_calls == 1);

    std::puts("  [Model] on_mode_change PASSED");
}

static void test_model_save_load_v1_roundtrip()
{
    std::puts("  [Model] v1 bytes manual roundtrip ...");

    nn::Model model;
    model.add<SingleParamLayer>(1, 2);
    auto &W = model.parameters()[0].get();
    assert(approx(W.at(0, 0), 0.0));
    assert(approx(W.at(0, 1), 0.0));

    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    const uint32_t magic = 0x4E4E4E4E;
    const uint32_t version = 1;
    ss.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    ss.write(reinterpret_cast<const char *>(&version), sizeof(version));

    // v1 无 n_params 段：直接写 1 个 1x2 矩阵
    std::size_t rows = 1, cols = 2;
    const double data[2] = {1.0, 2.0};
    ss.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    ss.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    ss.write(reinterpret_cast<const char *>(data), sizeof(data));

    model.load(ss);

    assert(approx(W.at(0, 0), 1.0));
    assert(approx(W.at(0, 1), 2.0));

    std::puts("  [Model] v1 bytes roundtrip PASSED");
}

static void test_model_save_load_v2_no_batchnorm()
{
    std::puts("  [Model] v2 save/load Linear(2,3) roundtrip ...");

    nn::Model model1;
    model1.add<nn::Linear>(2, 3); // W(3,2) + b(3,1) = 2 params, 9 elements

    std::size_t tag = 0;
    for (auto &p_ref : model1.parameters())
    {
        auto &p = p_ref.get();
        for (std::size_t r = 0; r < p.rows(); ++r)
        {
            for (std::size_t c = 0; c < p.cols(); ++c, ++tag)
            {
                p.set_value_unchecked(r, c, 100.0 + static_cast<double>(tag));
            }
        }
    }

    const std::string path = "/tmp/nn_test_v2_no_bn.bin";
    model1.save(path);

    nn::Model model2;
    model2.add<nn::Linear>(2, 3);
    model2.load(path);

    auto p1 = model1.parameters();
    auto p2 = model2.parameters();
    assert(p1.size() == p2.size());
    assert(p1.size() == 2);

    for (std::size_t i = 0; i < p1.size(); ++i)
    {
        const auto &a = p1[i].get();
        const auto &b = p2[i].get();
        assert(a.rows() == b.rows());
        assert(a.cols() == b.cols());
        for (std::size_t r = 0; r < a.rows(); ++r)
        {
            for (std::size_t c = 0; c < a.cols(); ++c)
            {
                assert(approx(a.at_unchecked(r, c), b.at_unchecked(r, c), 1e-12));
            }
        }
    }

    std::remove(path.c_str());
    std::puts("  [Model] v2 save/load Linear(2,3) PASSED");
}

static void test_model_load_state_count_mismatch()
{
    std::puts("  [Model] v2 load rejects state-layer count mismatch ...");

    nn::Model model1;
    model1.add<nn::BatchNorm1d>(2);

    auto *bn1 = static_cast<nn::BatchNorm1d *>(model1.get_layers()[0].get());
    nn::Matrix train_input(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2, 3);
    for (int i = 0; i < 3; ++i)
        bn1->forward(train_input);

    std::stringstream ss;
    model1.save(ss);

    nn::Model model2;
    model2.add<nn::Linear>(2, 3);

    ss.seekg(0);
    bool threw = false;
    try
    {
        model2.load(ss);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    assert(threw);

    nn::Model model3;
    model3.add<nn::BatchNorm1d>(2);
    ss.seekg(0);
    model3.load(ss);

    std::puts("  [Model] v2 load state count mismatch PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// BatchNorm1d 测试 (F2)
// ══════════════════════════════════════════════════════════════════════════════
static void test_batchnorm1d_forward_shape()
{
    std::puts("  [BatchNorm1d] forward shape ...");

    nn::BatchNorm1d bn(4);
    nn::Matrix input(4, 10, 1.0);
    auto out = bn.forward(input);
    assert(out.rows() == 4);
    assert(out.cols() == 10);

    std::puts("  [BatchNorm1d] forward shape PASSED");
}

static void test_batchnorm1d_forward_normalize()
{
    std::puts("  [BatchNorm1d] forward normalize: per-feature mean≈0, var≈1 ...");

    nn::BatchNorm1d bn(4);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(5.0, std::sqrt(2.0));

    std::vector<double> data(4 * 1000);
    for (auto &v : data)
        v = dist(rng);
    nn::Matrix input(std::move(data), 4, 1000);

    auto out = bn.forward(input);
    for (std::size_t i = 0; i < 4; ++i)
    {
        double sum = 0.0;
        for (std::size_t j = 0; j < 1000; ++j)
            sum += out.at_unchecked(i, j);
        double mean = sum / 1000.0;
        assert(std::fabs(mean) < 1e-5);

        double var = 0.0;
        for (std::size_t j = 0; j < 1000; ++j)
        {
            double d = out.at_unchecked(i, j) - mean;
            var += d * d;
        }
        // BN 内部用总体方差（分母 N，PyTorch 约定），所以归一化后总体方差严格为
        // 1.0（除 eps=1e-5 引入的偏移外）。这里用同样的分母 N 才能匹配。
        var /= 1000.0;
        assert(std::fabs(var - 1.0) < 1e-4);
    }

    std::puts("  [BatchNorm1d] forward normalize PASSED");
}

static void test_batchnorm1d_backward_shape()
{
    std::puts("  [BatchNorm1d] backward shape ...");

    nn::BatchNorm1d bn(4);
    nn::Matrix input(4, 10);
    bn.forward(input);

    nn::Matrix grad_output(4, 10, 0.5);
    auto grad_input = bn.backward(grad_output);
    assert(grad_input.rows() == 4);
    assert(grad_input.cols() == 10);

    std::puts("  [BatchNorm1d] backward shape PASSED");
}

static void test_batchnorm1d_eval_mode()
{
    std::puts("  [BatchNorm1d] eval mode uses running stats ...");

    nn::BatchNorm1d bn(2, 1e-5, 0.1, true);
    bn.on_mode_change(true);

    // 训练若干步以更新 running stats
    nn::Matrix train_input(std::vector<double>{10.0, 20.0, 30.0, 40.0, 50.0, 60.0}, 2, 3);
    for (int i = 0; i < 10; ++i)
        bn.forward(train_input);

    // 切换到 eval 模式
    bn.on_mode_change(false);
    nn::Matrix test_input(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2, 3);
    auto out_eval = bn.forward(test_input);

    // eval 模式的输出应满足 (input - running_mean) / sqrt(running_var + eps) * gamma + beta
    const auto &rm = bn.running_mean();
    const auto &rv = bn.running_var();
    for (std::size_t i = 0; i < 2; ++i)
    {
        const double m = rm.at_unchecked(i, 0);
        const double s = std::sqrt(rv.at_unchecked(i, 0) + 1e-5);
        for (std::size_t j = 0; j < 3; ++j)
        {
            const double expected = (test_input.at_unchecked(i, j) - m) / s;
            assert(approx(out_eval.at_unchecked(i, j), expected, 1e-9));
        }
    }

    // 训练模式与 eval 模式输出应有显著差异
    bn.on_mode_change(true);
    auto out_train = bn.forward(test_input);
    double diff = 0.0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            diff += std::fabs(out_train.at_unchecked(i, j) - out_eval.at_unchecked(i, j));
    assert(diff > 1e-3);

    std::puts("  [BatchNorm1d] eval mode PASSED");
}

static void test_batchnorm1d_running_stats_update()
{
    std::puts("  [BatchNorm1d] running stats update with momentum ...");

    nn::BatchNorm1d bn(2, 1e-5, 0.1, true);
    // 初始: running_mean=0, running_var=1
    nn::Matrix input(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                         11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0},
                      2, 10);

    bn.forward(input);
    // batch_mean[0] = 5.5, batch_mean[1] = 15.5
    // new_running_mean = (1-0.1)*0 + 0.1*batch_mean = 0.55, 1.55
    const auto &rm = bn.running_mean();
    assert(approx(rm.at_unchecked(0, 0), 0.55, 1e-9));
    assert(approx(rm.at_unchecked(1, 0), 1.55, 1e-9));
    assert(bn.num_batches_tracked() == 1);

    std::puts("  [BatchNorm1d] running stats update PASSED");
}

static void test_batchnorm1d_no_affine()
{
    std::puts("  [BatchNorm1d] no affine (parameters empty) ...");

    nn::BatchNorm1d bn(2, 1e-5, 0.1, /*affine=*/false);
    assert(bn.parameters().size() == 0);
    assert(bn.param_gradients().size() == 0);

    nn::Matrix input(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2, 3);
    auto out = bn.forward(input);
    // BN 按行（每个特征）归一化：每行均值应≈0、方差应≈1（总体方差，分母=N）。
    auto means = out.rowwise_mean();
    assert(std::fabs(means[0]) < 1e-9);
    assert(std::fabs(means[1]) < 1e-9);

    std::puts("  [BatchNorm1d] no affine PASSED");
}

static void test_batchnorm1d_save_load_v2()
{
    std::puts("  [BatchNorm1d] save/load v2 roundtrip (running stats + num_batches_tracked) ...");

    nn::Model model1;
    model1.add<nn::BatchNorm1d>(2);
    model1.add<nn::ReLU>();

    auto *bn1 = static_cast<nn::BatchNorm1d *>(model1.get_layers()[0].get());
    nn::Matrix input(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2, 3);
    for (int i = 0; i < 3; ++i)
        bn1->forward(input);

    model1.eval();
    auto out_before = model1.forward(input);
    const int64_t nbt_before = bn1->num_batches_tracked();

    const std::string path = "/tmp/nn_test_bn1d_v2.bin";
    model1.save(path);

    nn::Model model2;
    model2.add<nn::BatchNorm1d>(2);
    model2.add<nn::ReLU>();
    model2.load(path);
    model2.eval();
    auto out_after = model2.forward(input);

    for (std::size_t i = 0; i < out_before.rows(); ++i)
    {
        for (std::size_t j = 0; j < out_before.cols(); ++j)
        {
            assert(approx(out_before.at_unchecked(i, j), out_after.at_unchecked(i, j), 1e-12));
        }
    }

    auto *bn2 = static_cast<nn::BatchNorm1d *>(model2.get_layers()[0].get());
    assert(bn2->num_batches_tracked() == nbt_before);

    std::remove(path.c_str());
    std::puts("  [BatchNorm1d] save/load v2 PASSED");
}

static void test_batchnorm1d_gradient_check()
{
    std::puts("  [BatchNorm1d] gradient check via finite differences ...");

    nn::BatchNorm1d bn(2, 1e-5, 0.1, /*affine=*/true);
    nn::Matrix input(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2, 3);

    // 解析路径：loss = sum(out^2) → grad_out = 2*out（与 input_gradient_check 一致）
    // 此前用 grad_out = 1（loss = sum(out)）时，dgamma = sum(normalized) ≈ 0，
    // FD 也对 gamma 不敏感（sum 是线性仿射不变），所以原测试是 vacuous 的。
    // 改用 2*out 后，dgamma = sum(2*out * normalized) 显著非零，测试真正有效。
    const auto out = bn.forward(input);
    nn::Matrix grad_output(input.rows(), input.cols());
    for (std::size_t i = 0; i < input.rows(); ++i)
        for (std::size_t j = 0; j < input.cols(); ++j)
            grad_output.set_value_unchecked(i, j,
                                            2.0 * out.at_unchecked(i, j));
    bn.backward(grad_output);

    auto param_grads = bn.param_gradients();
    assert(param_grads.size() == 2);

    std::printf("    dgamma_analytical = [%.6e, %.6e]\n",
                param_grads[0].get().at_unchecked(0, 0),
                param_grads[0].get().at_unchecked(1, 0));
    std::printf("    dbeta_analytical  = [%.6e, %.6e]\n",
                param_grads[1].get().at_unchecked(0, 0),
                param_grads[1].get().at_unchecked(1, 0));

    // 检查 dgamma_ 每个元素的解析梯度 vs 有限差分
    for (std::size_t i = 0; i < 2; ++i)
    {
        const double analytical = param_grads[0].get().at_unchecked(i, 0);
        const double fd = finite_diff_grad(bn, input, /*param_idx=*/0, i, 0, 1e-5);
        assert(approx(analytical, fd, 1e-4));
    }

    // 检查 dbeta_ 同上（loss=sum(out²) 下 dbeta=2*sum(out)=0，对称数据上恒为 0；
    // 解析=0，FD=0，approx(0,0,1e-4) 仍通过。dgamma 才是真正捕捉到梯度的检查项。）
    for (std::size_t i = 0; i < 2; ++i)
    {
        const double analytical = param_grads[1].get().at_unchecked(i, 0);
        const double fd = finite_diff_grad(bn, input, /*param_idx=*/1, i, 0, 1e-5);
        assert(approx(analytical, fd, 1e-4));
    }

    std::puts("  [BatchNorm1d] gradient check PASSED");
}

static void test_batchnorm1d_batch_size_one()
{
    std::puts("  [BatchNorm1d] batch_size=1 edge case ...");

    nn::BatchNorm1d bn(2, 1e-5, 0.1, /*affine=*/true);
    nn::Matrix input(std::vector<double>{5.0, 10.0}, 2, 1);
    const auto out = bn.forward(input);

    assert(out.rows() == 2);
    assert(out.cols() == 1);
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        const double v = out.data()[i];
        assert(std::isfinite(v));
        assert(std::fabs(v) < 1e-6);
    }

    nn::Matrix grad_output(2, 1);
    for (std::size_t i = 0; i < out.size(); ++i)
        grad_output.set_value_unchecked(i, 0, 2.0 * out.data()[i]);
    const auto dx = bn.backward(grad_output);

    assert(dx.rows() == 2);
    assert(dx.cols() == 1);
    for (std::size_t i = 0; i < dx.size(); ++i)
    {
        const double v = dx.data()[i];
        assert(std::isfinite(v));
        assert(std::fabs(v) < 1e-6);
    }

    std::puts("  [BatchNorm1d] batch_size=1 edge case PASSED");
}

static void test_batchnorm1d_input_gradient_check()
{
    std::puts("  [BatchNorm1d] input gradient (dx) check via finite differences ...");

    nn::BatchNorm1d bn(2, 1e-5, 0.1, /*affine=*/true);
    nn::Matrix input(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2, 3);
    const std::size_t rows = input.rows();
    const std::size_t cols = input.cols();

    // 解析路径：loss = sum(out^2) → grad_out = 2*out
    const auto out = bn.forward(input);
    nn::Matrix grad_output(rows, cols);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            grad_output.set_value_unchecked(i, j,
                                            2.0 * out.at_unchecked(i, j));
    const auto dx_analytical = bn.backward(grad_output);

    std::printf("    dx_analytical =\n");
    for (std::size_t i = 0; i < rows; ++i)
    {
        std::printf("      [");
        for (std::size_t j = 0; j < cols; ++j)
        {
            std::printf("%s%.6e", j == 0 ? "" : ", ",
                        dx_analytical.at_unchecked(i, j));
        }
        std::printf("]\n");
    }

    const double h = 1e-5;
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            const double original = input.at_unchecked(i, j);

            nn::Matrix input_p = input;
            input_p.set_value_unchecked(i, j, original + h);
            const auto out_p = bn.forward(input_p);
            double loss_p = 0.0;
            for (std::size_t k = 0; k < out_p.size(); ++k)
                loss_p += out_p.data()[k] * out_p.data()[k];

            nn::Matrix input_m = input;
            input_m.set_value_unchecked(i, j, original - h);
            const auto out_m = bn.forward(input_m);
            double loss_m = 0.0;
            for (std::size_t k = 0; k < out_m.size(); ++k)
                loss_m += out_m.data()[k] * out_m.data()[k];

            const double fd = (loss_p - loss_m) / (2.0 * h);
            const double analytical = dx_analytical.at_unchecked(i, j);

            const double abs_tol = 1e-4;
            const double ref = std::fabs(analytical) > 1e-8
                                   ? std::fabs(analytical) : 1.0;
            const double tol = std::max(abs_tol, 1e-4 * ref);
            assert(approx(analytical, fd, tol));
        }
    }

    std::puts("  [BatchNorm1d] input gradient check PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// BatchNorm2d 测试 (F2)
// ══════════════════════════════════════════════════════════════════════════════
static void test_batchnorm2d_forward_shape()
{
    std::puts("  [BatchNorm2d] forward shape (C, N*H*W) -> (C, N*H*W) ...");

    nn::BatchNorm2d bn(3);
    nn::Matrix input(3, 2 * 4 * 5, 1.0); // C=3, N=2, H=4, W=5
    auto out = bn.forward(input);
    assert(out.rows() == 3);
    assert(out.cols() == 40);

    std::puts("  [BatchNorm2d] forward shape PASSED");
}

static void test_batchnorm2d_normalize()
{
    std::puts("  [BatchNorm2d] normalize per-channel ...");

    nn::BatchNorm2d bn(2);
    nn::Matrix input(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0,
                                         6.0, 7.0, 8.0, 9.0, 10.0},
                      2, 5);
    auto out = bn.forward(input);
    // BN2d 按通道（行）归一化：每行均值应≈0。
    auto means = out.rowwise_mean();
    assert(std::fabs(means[0]) < 1e-9);
    assert(std::fabs(means[1]) < 1e-9);

    std::puts("  [BatchNorm2d] normalize PASSED");
}

static void test_batchnorm2d_save_load_v2()
{
    std::puts("  [BatchNorm2d] save/load v2 roundtrip ...");

    nn::Model model1;
    model1.add<nn::BatchNorm2d>(3);

    auto *bn1 = static_cast<nn::BatchNorm2d *>(model1.get_layers()[0].get());
    nn::Matrix input(3, 8, 1.0);
    for (int i = 0; i < 3; ++i)
        bn1->forward(input);

    model1.eval();
    auto out_before = model1.forward(input);

    const std::string path = "/tmp/nn_test_bn2d_v2.bin";
    model1.save(path);

    nn::Model model2;
    model2.add<nn::BatchNorm2d>(3);
    model2.load(path);
    model2.eval();
    auto out_after = model2.forward(input);

    for (std::size_t i = 0; i < out_before.rows(); ++i)
    {
        for (std::size_t j = 0; j < out_before.cols(); ++j)
        {
            assert(approx(out_before.at_unchecked(i, j), out_after.at_unchecked(i, j), 1e-12));
        }
    }

    std::remove(path.c_str());
    std::puts("  [BatchNorm2d] save/load v2 PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// DataLoader 测试 (F1)
// ══════════════════════════════════════════════════════════════════════════════
static void test_dataloader_basic()
{
    std::puts("  [DataLoader] basic: 10 samples, batch=3, no drop -> 4 batches (3,3,3,1) ...");

    nn::Matrix features(std::vector<double>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                                            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0},
                        2, 10);
    nn::Matrix labels(std::vector<double>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, 1, 10);

    nn::TensorDataset ds(std::move(features), std::move(labels));
    nn::DataLoader loader(ds, /*batch_size=*/3, /*shuffle=*/false, /*drop_last=*/false);

    std::vector<std::size_t> batch_cols;
    for (auto [x, y] : loader)
    {
        (void)x;
        batch_cols.push_back(y.cols());
    }
    assert(batch_cols.size() == 4);
    assert(batch_cols[0] == 3);
    assert(batch_cols[1] == 3);
    assert(batch_cols[2] == 3);
    assert(batch_cols[3] == 1);

    std::puts("  [DataLoader] basic PASSED");
}

static void test_dataloader_drop_last()
{
    std::puts("  [DataLoader] drop_last: 10 samples, batch=3 -> 3 batches of 3 ...");

    nn::Matrix features(2, 10, 0.0);
    nn::Matrix labels(1, 10, 0.0);
    for (std::size_t i = 0; i < 10; ++i)
    {
        features.set_value_unchecked(0, i, static_cast<double>(i));
        labels.set_value_unchecked(0, i, static_cast<double>(i));
    }

    nn::TensorDataset ds(std::move(features), std::move(labels));
    nn::DataLoader loader(ds, 3, false, /*drop_last=*/true, 0);

    int count = 0;
    for (auto [x, y] : loader)
    {
        (void)x;
        assert(y.cols() == 3);
        ++count;
    }
    assert(count == 3);

    std::puts("  [DataLoader] drop_last PASSED");
}

static void test_dataloader_shuffle()
{
    std::puts("  [DataLoader] shuffle: different seeds produce different orderings ...");

    nn::Matrix features(2, 100, 0.0);
    nn::Matrix labels(1, 100, 0.0);
    for (std::size_t i = 0; i < 100; ++i)
    {
        features.set_value_unchecked(0, i, static_cast<double>(i));
        labels.set_value_unchecked(0, i, static_cast<double>(i));
    }

    nn::TensorDataset ds(std::move(features), std::move(labels));
    nn::DataLoader loader1(ds, 10, /*shuffle=*/true, false, /*seed=*/1);
    nn::DataLoader loader2(ds, 10, /*shuffle=*/true, false, /*seed=*/2);

    std::vector<double> order1, order2;
    for (auto [x, y] : loader1)
    {
        (void)x;
        for (std::size_t j = 0; j < y.cols(); ++j)
            order1.push_back(y.at_unchecked(0, j));
    }
    for (auto [x, y] : loader2)
    {
        (void)x;
        for (std::size_t j = 0; j < y.cols(); ++j)
            order2.push_back(y.at_unchecked(0, j));
    }

    int diffs = 0;
    for (std::size_t i = 0; i < order1.size(); ++i)
    {
        if (order1[i] != order2[i])
            ++diffs;
    }
    assert(diffs > 5);

    std::puts("  [DataLoader] shuffle PASSED");
}

static void test_dataloader_shuffle_reproducible_with_seed()
{
    std::puts("  [DataLoader] shuffle: same seed -> same order ...");

    nn::Matrix features(2, 100, 0.0);
    nn::Matrix labels(1, 100, 0.0);
    for (std::size_t i = 0; i < 100; ++i)
    {
        features.set_value_unchecked(0, i, static_cast<double>(i));
        labels.set_value_unchecked(0, i, static_cast<double>(i));
    }

    nn::TensorDataset ds(std::move(features), std::move(labels));
    nn::DataLoader loader1(ds, 10, true, false, 42);
    nn::DataLoader loader2(ds, 10, true, false, 42);

    std::vector<double> order1, order2;
    for (auto [x, y] : loader1)
    {
        (void)x;
        for (std::size_t j = 0; j < y.cols(); ++j)
            order1.push_back(y.at_unchecked(0, j));
    }
    for (auto [x, y] : loader2)
    {
        (void)x;
        for (std::size_t j = 0; j < y.cols(); ++j)
            order2.push_back(y.at_unchecked(0, j));
    }

    assert(order1.size() == order2.size());
    for (std::size_t i = 0; i < order1.size(); ++i)
        assert(order1[i] == order2[i]);

    std::puts("  [DataLoader] shuffle reproducible PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// clip_grad_norm_ 测试 (F3)
// ══════════════════════════════════════════════════════════════════════════════
static void test_grad_clip_no_op_below_threshold()
{
    std::puts("  [clip_grad_norm_] no-op below threshold (returns pre-clip norm) ...");

    nn::Matrix g1(1, 1, 0.5);
    nn::Matrix g2(1, 1, 0.5);
    nn::Matrix g3(1, 1, 0.5);

    const double total_norm =
        nn::clip_grad_norm_({std::ref(g1), std::ref(g2), std::ref(g3)}, 1.0);

    assert(approx(total_norm, std::sqrt(0.75), 1e-9));

    // 各梯度未变
    for (auto &g_ref : {std::ref(g1), std::ref(g2), std::ref(g3)})
    {
        assert(approx(g_ref.get().at_unchecked(0, 0), 0.5));
    }

    std::puts("  [clip_grad_norm_] no-op below threshold PASSED");
}

static void test_grad_clip_scales_above_threshold()
{
    std::puts("  [clip_grad_norm_] scales above threshold ...");

    nn::Matrix g1(1, 1, 5.0);
    nn::Matrix g2(1, 1, 5.0);
    nn::Matrix g3(1, 1, 5.0);

    const double total_norm =
        nn::clip_grad_norm_({std::ref(g1), std::ref(g2), std::ref(g3)}, 1.0, 1e-6);

    // pre-clip 总范数
    assert(approx(total_norm, std::sqrt(75.0), 1e-9));

    // 各梯度按 max_norm / (total_norm + eps) 缩放
    const double expected = 5.0 / (std::sqrt(75.0) + 1e-6);
    for (auto &g_ref : {std::ref(g1), std::ref(g2), std::ref(g3)})
    {
        assert(approx(g_ref.get().at_unchecked(0, 0), expected, 1e-9));
    }

    std::puts("  [clip_grad_norm_] scales above threshold PASSED");
}

static void test_grad_clip_nan_throws()
{
    std::puts("  [clip_grad_norm_] NaN in grads throws ...");

    nn::Matrix g(1, 1, std::numeric_limits<double>::quiet_NaN());

    bool threw = false;
    try
    {
        nn::clip_grad_norm_({std::ref(g)}, 1.0);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    assert(threw);

    std::puts("  [clip_grad_norm_] NaN throws PASSED");
}

static void test_grad_clip_inf_throws()
{
    std::puts("  [clip_grad_norm_] Inf in grads throws ...");

    nn::Matrix g(1, 1, std::numeric_limits<double>::infinity());

    bool threw = false;
    try
    {
        nn::clip_grad_norm_({std::ref(g)}, 1.0);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    assert(threw);

    std::puts("  [clip_grad_norm_] Inf throws PASSED");
}

// ══════════════════════════════════════════════════════════════════════════════
// Model::summary() 测试 (F4)
// ══════════════════════════════════════════════════════════════════════════════
static void test_summary_contains_total_params()
{
    std::puts("  [Model::summary] total params count ...");

    nn::Model model;
    model.add<nn::Linear>(3, 4); // W=(4,3)=12, b=(4,1)=4 -> total 16

    std::string s = model.summary();
    assert(s.find("Total params: 16") != std::string::npos);
    assert(s.find("Linear") != std::string::npos);

    std::puts("  [Model::summary] total params PASSED");
}

static void test_summary_contains_all_layer_names()
{
    std::puts("  [Model::summary] all layer names appear (F7 coverage) ...");

    nn::Model model;
    model.add<nn::Linear>(2, 3)
        .add<nn::ReLU>()
        .add<nn::Sigmoid>();

    std::string s = model.summary();
    assert(s.find("Linear") != std::string::npos);
    assert(s.find("ReLU") != std::string::npos);
    assert(s.find("Sigmoid") != std::string::npos);

    std::puts("  [Model::summary] all layer names PASSED");
}

static void test_summary_output_stream()
{
    std::puts("  [Model::summary] output stream write ...");

    nn::Model model;
    model.add<nn::Linear>(2, 3);

    std::ostringstream oss;
    std::string s = model.summary(&oss);
    assert(s == oss.str());
    assert(!s.empty());
    assert(oss.str().find("Total params") != std::string::npos);

    std::puts("  [Model::summary] output stream PASSED");
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
    {"matrix_matmul_nt_tn",     test_matrix_matmul_nt_tn},
    {"matrix_norm",             test_matrix_norm},
    {"matrix_colwise_mean",     test_matrix_colwise_mean},
    {"relu",                    test_relu},
    {"leaky_relu",              test_leaky_relu},
    {"sigmoid",                 test_sigmoid},
    {"tanh",                    test_tanh},
    {"gelu",                    test_gelu},
    {"dropout",                 test_dropout},
    {"linear_forward_backward", test_linear_forward_backward},
    {"linear_backward_blocked_matmul", test_linear_backward_blocked_matmul},
    {"linear_backward_grad_b_nonsquare", test_linear_backward_grad_b_nonsquare},
    {"mse_loss",                test_mse_loss},
    {"cross_entropy_loss",      test_cross_entropy_loss},
    {"sgd",                     test_sgd},
    {"sgd_momentum",            test_sgd_momentum},
    {"adam",                    test_adam},
    {"adamw",                   test_adamw},
    {"step_lr",                 test_step_lr},
    {"cosine_lr",               test_cosine_lr},
    {"exp_lr",                  test_exp_lr},
    {"model_forward",           test_model_forward},
    {"model_default_is_training", test_model_default_is_training},
    {"model_train_eval",        test_model_train_eval},
    {"on_mode_change",          test_on_mode_change_called},
    {"model_io",                test_model_io},
    {"model_io_member",         test_model_io_member},
    {"model_save_load_v1",      test_model_save_load_v1_roundtrip},
    {"model_save_load_v2",      test_model_save_load_v2_no_batchnorm},
    {"model_load_state_count_mismatch", test_model_load_state_count_mismatch},
    {"one_hot",                 test_one_hot},
    {"e2e_train",               test_end_to_end_train},
    {"set_data_match",          test_set_data_match},
    {"set_data_resize_grow",    test_set_data_resize_grow},
    {"set_data_resize_shrink",  test_set_data_resize_shrink},
    {"set_data_from_empty",     test_set_data_from_empty},
    {"set_data_empty_throws",   test_set_data_empty_throws},
    {"set_data_inconsistent_throws", test_set_data_inconsistent_throws},
    {"set_data_exception_safety",    test_set_data_exception_safety},
    {"batchnorm1d_forward_shape",        test_batchnorm1d_forward_shape},
    {"batchnorm1d_forward_normalize",    test_batchnorm1d_forward_normalize},
    {"batchnorm1d_backward_shape",       test_batchnorm1d_backward_shape},
    {"batchnorm1d_eval_mode",            test_batchnorm1d_eval_mode},
    {"batchnorm1d_running_stats_update", test_batchnorm1d_running_stats_update},
    {"batchnorm1d_no_affine",            test_batchnorm1d_no_affine},
    {"batchnorm1d_save_load_v2",         test_batchnorm1d_save_load_v2},
    {"batchnorm1d_gradient_check",       test_batchnorm1d_gradient_check},
    {"batchnorm1d_input_gradient_check", test_batchnorm1d_input_gradient_check},
    {"batchnorm1d_batch_size_one",       test_batchnorm1d_batch_size_one},
    {"batchnorm2d_forward_shape",        test_batchnorm2d_forward_shape},
    {"batchnorm2d_normalize",            test_batchnorm2d_normalize},
    {"batchnorm2d_save_load_v2",         test_batchnorm2d_save_load_v2},
    {"dataloader_basic",                 test_dataloader_basic},
    {"dataloader_drop_last",             test_dataloader_drop_last},
    {"dataloader_shuffle",               test_dataloader_shuffle},
    {"dataloader_shuffle_reproducible_with_seed", test_dataloader_shuffle_reproducible_with_seed},
    {"grad_clip_no_op_below_threshold",  test_grad_clip_no_op_below_threshold},
    {"grad_clip_scales_above_threshold", test_grad_clip_scales_above_threshold},
    {"grad_clip_nan_throws",             test_grad_clip_nan_throws},
    {"grad_clip_inf_throws",             test_grad_clip_inf_throws},
    {"summary_contains_total_params",    test_summary_contains_total_params},
    {"summary_contains_all_layer_names", test_summary_contains_all_layer_names},
    {"summary_output_stream",            test_summary_output_stream},
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
