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
    nn::TransformerEncoderLayerModule enc(16, 4, 64);
    nn::Matrix in_m(16, 8);
    nn::Tensor input(in_m);
    nn::Tensor out = enc.forward(input);
    assert(out.rows() == 16);
    assert(out.cols() == 8);
}

// ── TransformerEncoder ──────────────────────────────────────────────────────

static void test_encoder_forward_shape()
{
    const std::size_t d_model = 16, num_patches = 4, batch = 2;
    nn::TransformerEncoderModule enc(d_model, 4, 64, 2, num_patches);
    nn::Matrix in_m(d_model, num_patches * batch);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    for (auto &v : in_m.data()) v = dist(rng);
    nn::Tensor input(in_m);

    nn::Tensor out = enc.forward(input);
    assert(out.rows() == d_model);
    assert(out.cols() == batch);
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
    
    nn::Matrix grad_out(d_model, batch, 0.01);
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
    out.node()->backward_op();
    assert(input.grad().rows() == img * img);
    assert(input.grad().cols() == batch);
}

// ── Dispatch ────────────────────────────────────────────────────────────────

using TestFn = void (*)();
struct TestEntry { const char *name; TestFn fn; };

static const TestEntry tests[] = {
    {"matrix_resize",                      test_matrix_resize},
    {"matrix_row_slice",                   test_matrix_row_slice},
    {"matrix_set_row_slice",               test_matrix_set_row_slice},
    {"layernorm_forward_shape",            test_layernorm_forward_shape},
    {"layernorm_normalize",                test_layernorm_normalize},
    {"layernorm_backward_shape",           test_layernorm_backward_shape},
    {"layernorm_gradient_check",           test_layernorm_gradient_check},
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
    {"encoder_forward_shape",              test_encoder_forward_shape},
    {"encoder_backward_shape",             test_encoder_backward_shape},
    {"patch_embedding_shape",              test_patch_embedding_shape},
    {"patch_embedding_backward_shape",     test_patch_embedding_backward_shape},
    {"vit_e2e_forward",                    test_vit_e2e_forward},
    {"vit_e2e_backward",                   test_vit_e2e_backward},
};

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::size_t passed = 0;
        for (const auto &t : tests)
        {
            try { t.fn(); ++passed; std::cout << "  PASSED  " << t.name << "\n"; }
            catch (const std::exception &e)
            { std::cout << "  FAILED  " << t.name << " : " << e.what() << "\n"; }
        }
        std::cout << passed << "/" << (sizeof(tests) / sizeof(tests[0])) << " passed\n";
        return passed == sizeof(tests) / sizeof(tests[0]) ? 0 : 1;
    }

    for (const auto &t : tests)
    {
        if (std::string(argv[1]) == t.name)
        {
            try { t.fn(); std::cout << "  PASSED  " << t.name << "\n"; return 0; }
            catch (const std::exception &e)
            { std::cout << "  FAILED  " << t.name << " : " << e.what() << "\n"; return 1; }
        }
    }
    std::cerr << "Unknown test: " << argv[1] << "\n";
    return 1;
}
