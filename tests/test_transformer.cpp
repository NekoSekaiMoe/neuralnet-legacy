#include <neuralnet/nn/nn.h>
#include <neuralnet/layer.h>
#include <neuralnet/loss.h>
#include <neuralnet/optimizer.h>
#include <neuralnet/model/model.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static bool approx(double a, double b, double tol = 1e-6)
{
    return std::fabs(a - b) < tol;
}

// ── Matrix helpers ──────────────────────────────────────────────────────────

static void test_matrix_resize()
{
    nn::Matrix m(3, 4);
    assert(m.rows() == 3 && m.cols() == 4);
    m.resize(5, 6);
    assert(m.rows() == 5 && m.cols() == 6);
    assert(m.size() == 30);
}

static void test_matrix_row_slice()
{
    nn::Matrix m(4, 3);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            m.set_value_unchecked(i, j, static_cast<double>(i * 10 + j));

    nn::Matrix s = m.row_slice(1, 2);
    assert(s.rows() == 2 && s.cols() == 3);
    assert(approx(s.at(0, 0), 10.0));
    assert(approx(s.at(0, 2), 12.0));
    assert(approx(s.at(1, 0), 20.0));
    assert(approx(s.at(1, 2), 22.0));
}

static void test_matrix_set_row_slice()
{
    nn::Matrix m(4, 3);
    nn::Matrix s(2, 3, 99.0);
    m.set_row_slice(1, s);
    assert(approx(m.at(1, 0), 99.0));
    assert(approx(m.at(2, 2), 99.0));
    assert(approx(m.at(0, 0), 0.0));
    assert(approx(m.at(3, 0), 0.0));
}

// ── LayerNorm ───────────────────────────────────────────────────────────────

static void test_layernorm_forward_shape()
{
    nn::LayerNormModule ln(8);
    nn::Matrix in_m(8, 4);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto &v : in_m.data()) v = dist(rng);
    nn::Tensor input = nn::Tensor(in_m);

    nn::Tensor out = ln.forward(input);
    assert(out.rows() == 8);
    assert(out.cols() == 4);
}

static void test_layernorm_normalize()
{
    nn::LayerNormModule ln(4);
    nn::Matrix in_m(4, 2);
    in_m.set_value(0, 0, 1.0); in_m.set_value(1, 0, 2.0);
    in_m.set_value(2, 0, 3.0); in_m.set_value(3, 0, 4.0);
    in_m.set_value(0, 1, 10.0); in_m.set_value(1, 1, 20.0);
    in_m.set_value(2, 1, 30.0); in_m.set_value(3, 1, 40.0);
    nn::Tensor input = nn::Tensor(in_m);

    nn::Tensor out = ln.forward(input);
    const nn::Matrix &out_m = out.data();
    for (std::size_t j = 0; j < 2; ++j)
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < 4; ++i)
            sum += out_m.at(i, j);
        assert(approx(sum, 0.0, 1e-5));
    }
}

static void test_layernorm_backward_shape()
{
    nn::LayerNormModule ln(8);
    nn::Matrix in_m(8, 4);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto &v : in_m.data()) v = dist(rng);
    nn::Tensor input = nn::Tensor(in_m, true);

    nn::Tensor out = ln.forward(input);
    nn::Matrix grad_out(8, 4, 1.0);
    out.grad() = grad_out;
    out.node()->backward_op();
    assert(input.grad().rows() == 8);
    assert(input.grad().cols() == 4);
}

static void test_layernorm_gradient_check()
{
    nn::LayerNormModule ln(4);
    nn::Matrix in_m(4, 3);
    std::mt19937_64 rng(123);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto &v : in_m.data()) v = dist(rng);
    nn::Tensor input = nn::Tensor(in_m);

    auto sum_sq = [](const nn::Matrix &m) {
        double s = 0.0;
        for (std::size_t k = 0; k < m.size(); ++k)
            s += m.data()[k] * m.data()[k];
        return s;
    };

    nn::Tensor out = ln.forward(input);
    const nn::Matrix &out_m = out.data();
    nn::Matrix grad_out(out_m.rows(), out_m.cols());
    for (std::size_t k = 0; k < out_m.size(); ++k)
        grad_out.data()[k] = 2.0 * out_m.data()[k];
    
    for (auto &p : ln.parameters()) { p.grad().zero(); }
    out.grad() = grad_out;
    out.node()->backward_op();

    auto params = ln.parameters();
    for (std::size_t pi = 0; pi < params.size(); ++pi)
    {
        nn::Matrix &p = params[pi].data();
        const nn::Matrix &g = params[pi].grad();
        for (std::size_t r = 0; r < p.rows(); ++r)
            for (std::size_t c = 0; c < p.cols(); ++c)
            {
                double orig = p.at_unchecked(r, c);
                double eps = 1e-5;
                
                p.set_value_unchecked(r, c, orig + eps);
                double loss_plus = sum_sq(ln.forward(input).data());
                
                p.set_value_unchecked(r, c, orig - eps);
                double loss_minus = sum_sq(ln.forward(input).data());
                
                p.set_value_unchecked(r, c, orig);
                
                double fd = (loss_plus - loss_minus) / (2.0 * eps);
                double analytical = g.at_unchecked(r, c);
                assert(approx(fd, analytical, 1e-3));
            }
    }
}

/**
 * @brief Verifies RMSNorm output dimensions, normalization, and parameter scaling.
 */

static void test_rmsnorm_forward_shape()
{
    std::puts("  [RMSNorm] forward shape & normalization ...");

    const std::size_t F = 4, B = 2;
    nn::RMSNorm rms(F);
    nn::Matrix x(F, B);
    std::mt19937_64 rng(123);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto &v : x.data())
        v = dist(rng);

    nn::Matrix out = rms.forward(x);
    assert(out.rows() == F && out.cols() == B);

    // γ 初始为 1：每列 RMS(normed)² = mean(x²)/(mean(x²)+ε) ≈ 1
    for (std::size_t j = 0; j < B; ++j)
    {
        double ms = 0.0;
        for (std::size_t i = 0; i < F; ++i)
            ms += out.at_unchecked(i, j) * out.at_unchecked(i, j);
        ms /= static_cast<double>(F);
        assert(ms > 0.99 && ms <= 1.0);
    }

    // γ 行缩放：γ[1]=2 → 输出行 1 恰为 γ=1 时的两倍
    auto params = rms.parameters();
    nn::Matrix &gamma = params[0].get();
    gamma.set_value_unchecked(1, 0, 2.0);
    nn::Matrix out2 = rms.forward(x);
    for (std::size_t j = 0; j < B; ++j)
        assert(approx(out2.at_unchecked(1, j), 2.0 * out.at_unchecked(1, j)));

    std::puts("  [RMSNorm] forward shape & normalization PASSED");
}

/**
 * @brief Verifies RMSNorm input and scale-parameter gradients using central finite differences.
 */
static void test_rmsnorm_gradient_check()
{
    std::puts("  [RMSNorm] gradient check (central difference) ...");

    const std::size_t F = 5, B = 3;
    nn::RMSNorm rms(F);
    nn::Matrix in_m(F, B);
    std::mt19937_64 rng(456);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto &v : in_m.data())
        v = dist(rng);

    // 固定随机上游梯度 R，损失 L = Σ out⊙R → dL/dout = R
    nn::Matrix R(F, B);
    for (auto &v : R.data())
        v = dist(rng);

    auto loss_of = [&](const nn::Matrix &inp)
    {
        nn::Matrix o = rms.forward(inp);
        double s = 0.0;
        for (std::size_t k = 0; k < o.size(); ++k)
            s += o.data()[k] * R.data()[k];
        return s;
    };

    // 解析梯度（基于原始 in_m）
    (void)rms.forward(in_m);
    nn::Matrix grad_in = rms.backward(R);
    const double eps = 1e-6;

    // 1) 输入梯度
    for (std::size_t i = 0; i < F; ++i)
        for (std::size_t j = 0; j < B; ++j)
        {
            const double orig = in_m.at_unchecked(i, j);
            in_m.set_value_unchecked(i, j, orig + eps);
            const double lp = loss_of(in_m);
            in_m.set_value_unchecked(i, j, orig - eps);
            const double lm = loss_of(in_m);
            in_m.set_value_unchecked(i, j, orig);
            const double fd = (lp - lm) / (2.0 * eps);
            assert(approx(fd, grad_in.at_unchecked(i, j), 1e-5));
        }

    // 2) γ 梯度（dgamma 由上面 backward 填充）
    auto param_refs = rms.parameters();
    auto grad_refs = rms.param_gradients();
    nn::Matrix &gamma = param_refs[0].get();
    const nn::Matrix &dgamma = grad_refs[0].get();
    for (std::size_t r = 0; r < F; ++r)
    {
        const double orig = gamma.at_unchecked(r, 0);
        gamma.set_value_unchecked(r, 0, orig + eps);
        const double lp = loss_of(in_m);
        gamma.set_value_unchecked(r, 0, orig - eps);
        const double lm = loss_of(in_m);
        gamma.set_value_unchecked(r, 0, orig);
        const double fd = (lp - lm) / (2.0 * eps);
        assert(approx(fd, dgamma.at_unchecked(r, 0), 1e-5));
    }

    std::puts("  [RMSNorm] gradient check PASSED");
}

// ── SwiGLU ──────────────────────────────────────────────────────────────────

static void test_swiglu_forward()
{
    std::puts("  [SwiGLU] forward (hand-computed) ...");

    nn::SwiGLU act(2); // d_ff = 2，输入 (4, 1)
    nn::Matrix x(std::vector<double>{1.0, -1.0,   // gate
                                     2.0, 0.5}, 4, 1); // up
    nn::Matrix out = act.forward(x);
    assert(out.rows() == 2 && out.cols() == 1);

    // out = gate·σ(gate)·up
    const double s1 = 1.0 / (1.0 + std::exp(-1.0));
    assert(approx(out.at(0, 0), 1.0 * s1 * 2.0));
    assert(approx(out.at(1, 0), -1.0 * (1.0 - s1) * 0.5)); // σ(-1) = 1-σ(1)

    std::puts("  [SwiGLU] forward PASSED");
}

static void test_swiglu_gradient_check()
{
    std::puts("  [SwiGLU] gradient check (central difference) ...");

    const std::size_t D = 4, B = 3; // 输入 (2D, B)
    nn::SwiGLU act(D);
    nn::Matrix in_m(2 * D, B);
    std::mt19937_64 rng(789);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto &v : in_m.data())
        v = dist(rng);

    nn::Matrix R(D, B);
    for (auto &v : R.data())
        v = dist(rng);

    auto loss_of = [&](const nn::Matrix &inp)
    {
        nn::Matrix o = act.forward(inp);
        double s = 0.0;
        for (std::size_t k = 0; k < o.size(); ++k)
            s += o.data()[k] * R.data()[k];
        return s;
    };

    (void)act.forward(in_m);
    nn::Matrix grad_in = act.backward(R);
    assert(grad_in.rows() == 2 * D && grad_in.cols() == B);

    const double eps = 1e-6;
    for (std::size_t i = 0; i < 2 * D; ++i)
        for (std::size_t j = 0; j < B; ++j)
        {
            const double orig = in_m.at_unchecked(i, j);
            in_m.set_value_unchecked(i, j, orig + eps);
            const double lp = loss_of(in_m);
            in_m.set_value_unchecked(i, j, orig - eps);
            const double lm = loss_of(in_m);
            in_m.set_value_unchecked(i, j, orig);
            const double fd = (lp - lm) / (2.0 * eps);
            assert(approx(fd, grad_in.at_unchecked(i, j), 1e-5));
        }

    std::puts("  [SwiGLU] gradient check PASSED");
}

/**
 * @brief Verifies composite SwiGLU feed-forward output dimensions and input gradients.
 */
static void test_swiglu_ffn_composite()
{
    std::puts("  [SwiGLUFeedForward] composite forward & gradcheck ...");

    const std::size_t d_model = 6, d_ff = 8, B = 2;
    nn::SwiGLUFeedForward ffn(d_model, d_ff);
    nn::Matrix in_m(d_model, B);
    std::mt19937_64 rng(321);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto &v : in_m.data())
        v = dist(rng);

    nn::Matrix out = ffn.forward(in_m);
    assert(out.rows() == d_model && out.cols() == B);
    // 参数量：W1(d_ff*2, d_model) + b1(2d_ff) + W2(d_model, d_ff) + b2(d_model)
    assert(ffn.param_count() == 2 * d_ff * d_model + 2 * d_ff + d_model * d_ff + d_model);

    nn::Matrix R(d_model, B);
    for (auto &v : R.data())
        v = dist(rng);

    auto loss_of = [&](const nn::Matrix &inp)
    {
        nn::Matrix o = ffn.forward(inp);
        double s = 0.0;
        for (std::size_t k = 0; k < o.size(); ++k)
            s += o.data()[k] * R.data()[k];
        return s;
    };

    (void)ffn.forward(in_m);
    nn::Matrix grad_in = ffn.backward(R);
    assert(grad_in.rows() == d_model && grad_in.cols() == B);

    const double eps = 1e-6;
    for (std::size_t i = 0; i < d_model; ++i)
        for (std::size_t j = 0; j < B; ++j)
        {
            const double orig = in_m.at_unchecked(i, j);
            in_m.set_value_unchecked(i, j, orig + eps);
            const double lp = loss_of(in_m);
            in_m.set_value_unchecked(i, j, orig - eps);
            const double lm = loss_of(in_m);
            in_m.set_value_unchecked(i, j, orig);
            const double fd = (lp - lm) / (2.0 * eps);
            assert(approx(fd, grad_in.at_unchecked(i, j), 1e-4));
        }

    std::puts("  [SwiGLUFeedForward] composite forward & gradcheck PASSED");
}

// ── Softmax ─────────────────────────────────────────────────────────────────

static void test_softmax_forward()
{
    nn::SoftmaxModule sm;
    nn::Matrix in_m(3, 2);
    // col 0
    in_m.set_value(0, 0, 1.0);
    in_m.set_value(1, 0, 2.0);
    in_m.set_value(2, 0, 3.0);
    // col 1
    in_m.set_value(0, 1, 10.0);
    in_m.set_value(1, 1, 10.0);
    in_m.set_value(2, 1, 10.0);
    nn::Tensor input = nn::Tensor(in_m);

    nn::Tensor out = sm.forward(input);
    const nn::Matrix &out_m = out.data();
    
    assert(out_m.rows() == 3);
    assert(out_m.cols() == 2);

    double sum0 = out_m.at(0, 0) + out_m.at(1, 0) + out_m.at(2, 0);
    assert(approx(sum0, 1.0, 1e-5));
    assert(out_m.at(2, 0) > out_m.at(1, 0));
    assert(out_m.at(1, 0) > out_m.at(0, 0));

    double sum1 = out_m.at(0, 1) + out_m.at(1, 1) + out_m.at(2, 1);
    assert(approx(sum1, 1.0, 1e-5));
    assert(approx(out_m.at(0, 1), 1.0 / 3.0, 1e-5));
    assert(approx(out_m.at(1, 1), 1.0 / 3.0, 1e-5));
    assert(approx(out_m.at(2, 1), 1.0 / 3.0, 1e-5));
}

static void test_softmax_backward_shape()
{
    nn::SoftmaxModule sm;
    nn::Matrix in_m(5, 4);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto &v : in_m.data()) v = dist(rng);
    nn::Tensor input = nn::Tensor(in_m, true);

    nn::Tensor out = sm.forward(input);
    nn::Matrix grad_out(5, 4, 1.0);
    out.grad() = grad_out;
    out.node()->backward_op();
    
    assert(input.grad().rows() == 5);
    assert(input.grad().cols() == 4);
}

// ── PositionalEncoding ──────────────────────────────────────────────────────

static void test_positional_encoding_shape()
{
    nn::PositionalEncodingModule pe(16, 64);
    nn::Tensor input = nn::Tensor(nn::Matrix(16, 10));
    nn::Tensor out = pe.forward(input);
    assert(out.rows() == 16 && out.cols() == 10);
}

static void test_positional_encoding_adds_values()
{
    nn::PositionalEncodingModule pe(8, 32);
    nn::Tensor input = nn::Tensor(nn::Matrix(8, 4, 0.0));
    nn::Tensor out = pe.forward(input);

    bool has_nonzero = false;
    for (std::size_t k = 0; k < out.data().size(); ++k)
        if (std::fabs(out.data().data()[k]) > 1e-10) { has_nonzero = true; break; }
    assert(has_nonzero);
}

static void test_positional_encoding_backward_passthrough()
{
    nn::PositionalEncodingModule pe(8, 32);
    nn::Tensor input = nn::Tensor(nn::Matrix(8, 4, 1.0), true);
    nn::Tensor out = pe.forward(input);
    nn::Matrix grad_out(8, 4, 3.14);
    out.grad() = grad_out;
    out.node()->backward_op();

    assert(input.grad().at(0, 0) == 3.14);
}

// ── MultiHeadAttention ──────────────────────────────────────────────────────

static void test_mha_forward_shape()
{
    nn::MultiHeadAttentionModule mha(16, 4);
    nn::Matrix in_m(16, 8);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    for (auto &v : in_m.data()) v = dist(rng);
    nn::Tensor input(in_m);

    nn::Tensor out = mha.forward(input);
    assert(out.rows() == 16);
    assert(out.cols() == 8);
}

static void test_mha_backward_shape()
{
    nn::MultiHeadAttentionModule mha(16, 4);
    nn::Matrix in_m(16, 8);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    for (auto &v : in_m.data()) v = dist(rng);
    nn::Tensor input(in_m, true);

    nn::Tensor out = mha.forward(input);
    nn::Matrix grad_out(16, 8, 0.01);
    out.grad() = grad_out;
    out.node()->backward_op();
    assert(input.grad().rows() == 16 && input.grad().cols() == 8);
}

static void test_mha_params_count()
{
    nn::MultiHeadAttentionModule mha(16, 4);
    assert(mha.param_count() == 4 * 16 * 16);
    assert(mha.parameters().size() == 4);
}

// ── FeedForward ─────────────────────────────────────────────────────────────

static void test_feedforward_shape()
{
    nn::FeedForwardModule ffn(16, 64);
    nn::Matrix in_m(16, 8);
    nn::Tensor input(in_m);
    nn::Tensor out = ffn.forward(input);
    assert(out.rows() == 16);
    assert(out.cols() == 8);
}

static void test_feedforward_params()
{
    nn::FeedForwardModule ffn(16, 64);
    assert(ffn.param_count() == 16 * 64 + 64 + 64 * 16 + 16);
    assert(ffn.parameters().size() == 4);
}

// ── TransformerEncoderLayer ─────────────────────────────────────────────────

static void test_encoder_layer_shape()
{
    std::size_t d_model = 16, num_heads = 4, d_ff = 64;
    nn::TransformerEncoderLayerModule layer(d_model, num_heads, d_ff);

    const std::size_t batch = 2, seq_len = 10;
    nn::Matrix in_m(d_model, batch * seq_len, 1.0);
    nn::Tensor input(in_m, true);

    nn::Tensor out = layer.forward(input);
    assert(out.rows() == d_model);
    assert(out.cols() == batch * seq_len);
}

static void test_encoder_layer_backward_shape()
{
    std::size_t d_model = 16, num_heads = 4, d_ff = 64;
    nn::TransformerEncoderLayerModule layer(d_model, num_heads, d_ff);

    const std::size_t batch = 2, seq_len = 10;
    nn::Matrix in_m(d_model, batch * seq_len, 1.0);
    nn::Tensor input(in_m, true);

    nn::Tensor out = layer.forward(input);
    
    out.grad() = nn::Matrix(d_model, batch * seq_len, 1.0);
    out.backward(false);
    
    assert(input.grad().rows() == d_model);
    assert(input.grad().cols() == batch * seq_len);
}

static void test_encoder_forward_shape()
{
    const std::size_t d_model = 16, num_patches = 4, batch = 2;
    nn::TransformerEncoderModule enc(d_model, 4, 64, 2, num_patches);
    nn::Matrix in_m(d_model, num_patches * batch);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    for (auto &v : in_m.data()) v = dist(rng);
    nn::Tensor input(in_m, true);

    nn::Tensor out = enc.forward(input);
    
    nn::Matrix grad_out(d_model, batch, 0.01);
    out.grad() = grad_out;
    out.node()->backward_op();
    
    assert(input.grad().rows() == d_model);
    assert(input.grad().cols() == num_patches * batch);
}

static void test_encoder_backward_shape()
{
    const std::size_t d_model = 16, num_patches = 4, batch = 2;
    nn::TransformerEncoderModule enc(d_model, 4, 64, 2, num_patches);
    nn::Matrix in_m(d_model, num_patches * batch);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    for (auto &v : in_m.data()) v = dist(rng);
    nn::Tensor input(in_m, true);

    nn::Tensor out = enc.forward(input);
    
    nn::Matrix grad_out(d_model, num_patches * batch, 0.01);
    out.grad() = grad_out;
    out.node()->backward_op();
    
    assert(input.grad().rows() == d_model);
    assert(input.grad().cols() == num_patches * batch);
}

// ── PatchEmbedding ──────────────────────────────────────────────────────────

static void test_patch_embedding_shape()
{
    const std::size_t img_size = 28;
    const std::size_t patch_size = 7;
    const std::size_t d_model = 16;
    const std::size_t num_patches = (img_size / patch_size) * (img_size / patch_size);
    const std::size_t batch = 2;

    nn::PatchEmbeddingModule patch_emb(img_size, patch_size, d_model);
    nn::Matrix in_m(img_size * img_size, batch);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    for (auto &v : in_m.data()) v = dist(rng);
    nn::Tensor input(in_m);

    nn::Tensor out = patch_emb.forward(input);
    assert(out.rows() == d_model);
    assert(out.cols() == num_patches * batch);
}

static void test_patch_embedding_backward_shape()
{
    const std::size_t img = 28, patch = 7, d_model = 16;
    const std::size_t num_patches = (img / patch) * (img / patch);
    const std::size_t batch = 2;

    nn::PatchEmbeddingModule patch_emb(img, patch, d_model);
    nn::Matrix in_m(img * img, batch);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    for (auto &v : in_m.data()) v = dist(rng);
    nn::Tensor input(in_m, true);

    nn::Tensor out = patch_emb.forward(input);
    nn::Matrix grad_out(d_model, num_patches * batch, 0.01);
    out.grad() = grad_out;
    out.node()->backward_op();

    assert(input.grad().rows() == img * img);
    assert(input.grad().cols() == batch);
}

// ── MNIST ViT end-to-end ────────────────────────────────────────────────────

static void test_vit_e2e_forward()
{
    const std::size_t img = 28, patch = 7, d_model = 16;
    const std::size_t num_patches = (img / patch) * (img / patch);
    const std::size_t batch = 2;

    nn::Sequential model;
    model.add<nn::PatchEmbeddingModule>(img, patch, d_model);
    model.add<nn::TransformerEncoderModule>(d_model, 4, 64, 1, num_patches);
    model.add<nn::LinearModule>(d_model, std::size_t{10});

    nn::Matrix in_m(img * img, batch);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    for (auto &v : in_m.data()) v = dist(rng);
    nn::Tensor input(in_m);

    nn::Tensor out = model.forward(input);
    assert(out.rows() == 10);
    assert(out.cols() == batch);
}

static void test_vit_e2e_backward()
{
    const std::size_t img = 28, patch = 7, d_model = 16;
    const std::size_t num_patches = (img / patch) * (img / patch);
    const std::size_t batch = 2;

    nn::Sequential model;
    model.add<nn::PatchEmbeddingModule>(img, patch, d_model);
    model.add<nn::TransformerEncoderModule>(d_model, 4, 64, 1, num_patches);
    model.add<nn::LinearModule>(d_model, std::size_t{10});

    nn::Matrix in_m(img * img, batch);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    for (auto &v : in_m.data()) v = dist(rng);
    nn::Tensor input(in_m, true);

    nn::Tensor out = model.forward(input);

    nn::CrossEntropyLoss loss;
    nn::Matrix target(10, batch);
    target.set_value(3, 0, 1.0);
    target.set_value(7, 1, 1.0);
    double lv = loss.forward(out.data(), target);
    (void)lv;

    out.grad() = loss.backward();
    out.backward(false);
    assert(input.grad().rows() == img * img);
    assert(input.grad().cols() == batch);
}

// ── Dispatch ────────────────────────────────────────────────────────────────

#include "test_runner.h"

TestEntry tests[] = {
    {"matrix_resize",                      test_matrix_resize},
    {"matrix_row_slice",                   test_matrix_row_slice},
    {"matrix_set_row_slice",               test_matrix_set_row_slice},
    {"layernorm_forward_shape",            test_layernorm_forward_shape},
    {"layernorm_normalize",                test_layernorm_normalize},
    {"layernorm_backward_shape",           test_layernorm_backward_shape},
    {"layernorm_gradient_check",           test_layernorm_gradient_check},
    {"rmsnorm_forward_shape",              test_rmsnorm_forward_shape},
    {"rmsnorm_gradient_check",             test_rmsnorm_gradient_check},
    {"swiglu_forward",                      test_swiglu_forward},
    {"swiglu_gradient_check",               test_swiglu_gradient_check},
    {"swiglu_ffn_composite",                test_swiglu_ffn_composite},
    {"softmax_forward",                    test_softmax_forward},
    {"softmax_backward_shape",             test_softmax_backward_shape},
    {"positional_encoding_shape",          test_positional_encoding_shape},
    {"positional_encoding_adds_values",    test_positional_encoding_adds_values},
    {"positional_encoding_backward_passthrough", test_positional_encoding_backward_passthrough},
    {"mha_forward_shape",                  test_mha_forward_shape},
    {"mha_backward_shape",                 test_mha_backward_shape},
    {"mha_params_count",                   test_mha_params_count},
    {"feedforward_shape",                  test_feedforward_shape},
    {"feedforward_params",                 test_feedforward_params},
    {"encoder_layer_shape",                test_encoder_layer_shape},
    {"encoder_layer_backward_shape",       test_encoder_layer_backward_shape},
    {"encoder_forward_shape",              test_encoder_forward_shape},
    {"encoder_backward_shape",             test_encoder_backward_shape},
    {"patch_embedding_shape",              test_patch_embedding_shape},
    {"patch_embedding_backward_shape",     test_patch_embedding_backward_shape},
    {"vit_e2e_forward",                    test_vit_e2e_forward},
    {"vit_e2e_backward",                   test_vit_e2e_backward},
};

int main(int argc, char *argv[])
{
    return run_tests(argc, argv, tests, sizeof(tests) / sizeof(tests[0]));
}
