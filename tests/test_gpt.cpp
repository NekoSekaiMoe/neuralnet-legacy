#include <neuralnet/nn/nn.h>
#include <neuralnet/layer.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

[[maybe_unused]] static bool approx(double a, double b, double tol = 1e-6)
{
    return std::fabs(a - b) < tol;
}

static void test_causal_self_attention_shape()
{
    // vocab_size=10, d_model=16, seq_len=5, num_heads=4
    nn::MultiHeadAttentionModule attn(16, 4, true);
    nn::Matrix in_m(16, 5); // seq_len=5, batch_size=1
    for (std::size_t i = 0; i < in_m.size(); ++i)
        in_m.data()[i] = 1.0;
    nn::Tensor input(in_m);

    nn::Tensor out = attn.forward(input);
    assert(out.rows() == 16);
    assert(out.cols() == 5);
}

static void test_causal_self_attention_masking()
{
    nn::MultiHeadAttentionModule attn(8, 2, true);
    
    // We create a dummy sequence of length 3
    nn::Matrix input1_m(8, 3);
    nn::Matrix input2_m(8, 3);

    for (std::size_t i = 0; i < 24; ++i)
    {
        input1_m.data()[i] = 0.5 * i;
        input2_m.data()[i] = 0.5 * i;
    }
    
    // Change token 2 (the 3rd token) in input2
    for (std::size_t d = 0; d < 8; ++d)
    {
        input2_m.set_value_unchecked(d, 2, input2_m.at_unchecked(d, 2) + 10.0);
    }
    
    nn::Tensor input1(input1_m);
    nn::Tensor input2(input2_m);

    nn::Tensor out1 = attn.forward(input1);
    nn::Tensor out2 = attn.forward(input2);

    // Because of causal masking, the output of the first two tokens (0 and 1)
    // should not depend on the third token (2).
    for (std::size_t t = 0; t < 2; ++t)
    {
        for (std::size_t d = 0; d < 8; ++d)
        {
            double diff = std::abs(out1.data().at_unchecked(d, t) - out2.data().at_unchecked(d, t));
            assert(diff < 1e-7);
        }
    }
}

static void test_gpt_block_shape()
{
    nn::GPTBlockModule block(16, 4, 32);
    nn::Matrix in_m(16, 10);
    for (std::size_t i = 0; i < in_m.size(); ++i)
        in_m.data()[i] = 0.5;
    nn::Tensor input(in_m, true);

    nn::Tensor out = block.forward(input);
    assert(out.rows() == 16);
    assert(out.cols() == 10);

    nn::Matrix grad_out(16, 10, 1.0);
    out.grad() = grad_out;
    out.node()->backward_op();
    
    assert(input.grad().rows() == 16);
    assert(input.grad().cols() == 10);
}

static void test_gpt_model_shape()
{
    // vocab_size=100, d_model=16, seq_len=10, num_heads=4, d_ff=32, num_layers=2
    nn::GPTModelModule model(100, 16, 10, 4, 32, 2);
    nn::Matrix in_m(5, 2); // seq_len=5, batch_size=2
    for (std::size_t i = 0; i < in_m.size(); ++i)
        in_m.data()[i] = static_cast<double>(i % 100);
    nn::Tensor input(in_m, true);

    nn::Tensor out = model.forward(input);
    assert(out.rows() == 100);
    assert(out.cols() == 10); // seq_len=5 * batch_size=2 = 10

    nn::Matrix grad_out(100, 10, 0.1);
    out.grad() = grad_out;
    out.node()->backward_op();
    
    assert(input.grad().rows() == 5);
    assert(input.grad().cols() == 2);
}

static void test_gpt_model_numeric_grad()
{
    // Small GPT model for gradient checking
    nn::GPTModelModule model(10, 8, 4, 2, 16, 1);
    nn::Matrix in_m(4, 2); // seq_len=4, batch=2
    for (std::size_t i = 0; i < in_m.size(); ++i)
        in_m.data()[i] = static_cast<double>(i % 10);
    nn::Tensor input(in_m);
    
    auto params = model.parameters();
    
    // forward
    nn::Tensor out = model.forward(input);
    
    // define a simple loss: sum of all elements
    nn::Matrix grad_out(out.rows(), out.cols(), 1.0);
    out.grad() = grad_out;
    out.node()->backward_op();
    
    // Pick one parameter to check (e.g. token_emb_)
    auto& p = params[0].data();
    auto& g = params[0].grad();
    
    if (p.size() > 0) {
        std::size_t check_idx = 0;
        double orig_val = p.data()[check_idx];
        double eps = 1e-5;
        
        p.data()[check_idx] = orig_val + eps;
        nn::Tensor out_plus = model.forward(input);
        double loss_plus = 0.0;
        for (double v : out_plus.data().data()) loss_plus += v;
        
        p.data()[check_idx] = orig_val - eps;
        nn::Tensor out_minus = model.forward(input);
        double loss_minus = 0.0;
        for (double v : out_minus.data().data()) loss_minus += v;
        
        p.data()[check_idx] = orig_val;
        
        double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
        double ana_grad = g.data()[check_idx];
        
        assert(std::abs(num_grad - ana_grad) < 1e-3);
    }
}

static void test_gpt_model_generate()
{
    nn::GPTModelModule model(100, 16, 10, 4, 32, 2);
    std::vector<std::size_t> prompt = {1, 2, 3};
    auto generated = model.generate(prompt, 5, 0.0); // greedy
    assert(generated.size() == 5);
}

// ═════════════════════════════════════════════════════════════════════════════
//  以下为移植自上游的新特性测试（RoPE / ALiBi / Doc 掩码 / KV cache 增量推理）
//  注意：这些测试针对 layer.h 的 Matrix/Layer 体系（与上方 Module 体系互补）
// ═════════════════════════════════════════════════════════════════════════════

// RoPE 已知角度验证：head_dim=2 时 theta = pos（j=0 → 指数 0），
// [x1;x2] 旋转后应为 [x1·cosθ − x2·sinθ; x1·sinθ + x2·cosθ]
static void test_rope_known_rotation()
{
    nn::RotaryEmbedding rope(2);
    rope.ensure(8);

    for (const std::size_t pos : {0u, 1u, 2u, 5u})
    {
        nn::Matrix q(2, 1);
        q.set_value_unchecked(0, 0, 3.0);
        q.set_value_unchecked(1, 0, 4.0);
        rope.apply_column(q, 0, 0, pos);
        const double theta = static_cast<double>(pos); // base^0 = 1
        const double e1 = 3.0 * std::cos(theta) - 4.0 * std::sin(theta);
        const double e2 = 3.0 * std::sin(theta) + 4.0 * std::cos(theta);
        assert(std::fabs(q.at_unchecked(0, 0) - e1) < 1e-12);
        assert(std::fabs(q.at_unchecked(1, 0) - e2) < 1e-12);
    }
    std::puts("  PASSED  rope_known_rotation");
}

// 旋转不改变每位置的向量范数
static void test_rope_preserves_norm()
{
    nn::RotaryEmbedding rope(8);
    nn::Matrix q(8, 6);
    std::mt19937_64 rng(11);
    std::normal_distribution<double> d(0.0, 1.0);
    for (std::size_t i = 0; i < q.size(); ++i)
        q.data()[i] = d(rng);

    std::vector<double> norms(q.cols());
    for (std::size_t j = 0; j < q.cols(); ++j)
    {
        double s = 0.0;
        for (std::size_t i = 0; i < q.rows(); ++i)
            s += q.at_unchecked(i, j) * q.at_unchecked(i, j);
        norms[j] = std::sqrt(s);
    }

    rope.apply_inplace(q, 0, q.cols());
    for (std::size_t j = 0; j < q.cols(); ++j)
    {
        double s = 0.0;
        for (std::size_t i = 0; i < q.rows(); ++i)
            s += q.at_unchecked(i, j) * q.at_unchecked(i, j);
        assert(std::fabs(std::sqrt(s) - norms[j]) < 1e-12);
    }
    std::puts("  PASSED  rope_preserves_norm");
}

// ALiBi：单 token 序列无距离偏置 → 与非 ALiBi 输出一致
static void test_alibi_single_token_no_bias()
{
    nn::MultiHeadAttention plain(8, 2, true);
    nn::MultiHeadAttention alibi(8, 2, true, false, true);

    // 两个实例的随机初始化不同 → 先同步权重，只对比 ALiBi 本身的效果
    auto wp = plain.parameters();
    auto wa = alibi.parameters();
    for (std::size_t k = 0; k < wp.size(); ++k)
        wa[k].get() = wp[k].get();

    nn::Matrix x(8, 1);
    for (std::size_t i = 0; i < x.size(); ++i)
        x.data()[i] = 0.3 * static_cast<double>(i);

    nn::Matrix o1 = plain.forward(x);
    nn::Matrix o2 = alibi.forward(x);
    for (std::size_t i = 0; i < o1.size(); ++i)
        assert(std::fabs(o1.data()[i] - o2.data()[i]) < 1e-12);
    std::puts("  PASSED  alibi_single_token_no_bias");
}

// ALiBi：因果性保持（未来 token 不影响历史输出）且输出与无 ALiBi 不同
static void test_alibi_causality_and_effect()
{
    nn::MultiHeadAttention attn(8, 2, true, false, true);

    nn::Matrix x1(8, 4), x2(8, 4);
    for (std::size_t i = 0; i < x1.size(); ++i)
    {
        x1.data()[i] = 0.31 * static_cast<double>(i);
        x2.data()[i] = x1.data()[i];
    }
    for (std::size_t d = 0; d < 8; ++d)
        x2.set_value_unchecked(d, 3, x2.at_unchecked(d, 3) + 5.0);

    nn::Matrix o1 = attn.forward(x1);
    nn::Matrix o2 = attn.forward(x2);
    for (std::size_t t = 0; t < 3; ++t)
        for (std::size_t d = 0; d < 8; ++d)
            assert(std::fabs(o1.at_unchecked(d, t) - o2.at_unchecked(d, t)) < 1e-12);

    // 与无 ALiBi 的输出不同（偏置生效）
    nn::MultiHeadAttention plain(8, 2, true);
    nn::Matrix o0 = plain.forward(x1);
    bool differs = false;
    for (std::size_t i = 0; i < o0.size(); ++i)
        if (std::fabs(o0.data()[i] - o1.data()[i]) > 1e-9)
            differs = true;
    assert(differs);
    std::puts("  PASSED  alibi_causality_and_effect");
}

// Doc 掩码：文档 A 内 token 的扰动不影响文档 B 后续 token 的输出
static void test_doc_mask_isolation()
{
    nn::MultiHeadAttention attn(8, 2, /*causal=*/true);
    // 4 个 token，前 2 个属文档 0，后 2 个属文档 1
    attn.set_document_ids({0, 0, 1, 1});

    nn::Matrix x1(8, 4), x2(8, 4);
    for (std::size_t i = 0; i < x1.size(); ++i)
    {
        x1.data()[i] = 0.29 * static_cast<double>(i);
        x2.data()[i] = x1.data()[i];
    }
    // 扰动文档 0 的 token 1
    for (std::size_t d = 0; d < 8; ++d)
        x2.set_value_unchecked(d, 1, x2.at_unchecked(d, 1) + 7.0);

    nn::Matrix o1 = attn.forward(x1);
    nn::Matrix o2 = attn.forward(x2);
    // token 1 在文档 0 → 扰动影响自身
    bool self_changed = false;
    for (std::size_t d = 0; d < 8; ++d)
        if (std::fabs(o1.at_unchecked(d, 1) - o2.at_unchecked(d, 1)) > 1e-9)
            self_changed = true;
    assert(self_changed);
    // token 2/3 在文档 1 → 不可见文档 0 的扰动（即使因果上在其之前）
    for (std::size_t t = 2; t < 4; ++t)
        for (std::size_t d = 0; d < 8; ++d)
            assert(std::fabs(o1.at_unchecked(d, t) - o2.at_unchecked(d, t)) < 1e-12);
    std::puts("  PASSED  doc_mask_isolation");
}

// KV cache 增量推理与全量前向数值等价
static void check_kv_matches_full(nn::GPTModel &model,
                                  const std::vector<std::size_t> &prompt)
{
    const std::size_t sl = prompt.size();
    nn::Matrix in(sl, 1);
    for (std::size_t t = 0; t < sl; ++t)
        in.set_value_unchecked(t, 0, static_cast<double>(prompt[t]));
    nn::Matrix full = model.forward(in); // (vocab, sl)

    model.clear_cache();
    nn::Matrix step = model.forward_step(prompt); // (vocab, 1)

    for (std::size_t v = 0; v < model.vocab_size(); ++v)
    {
        const double a = full.at_unchecked(v, sl - 1);
        const double b = step.at_unchecked(v, 0);
        assert(std::fabs(a - b) < 1e-8 * std::max(1.0, std::fabs(a)));
    }
}

static void test_kv_cache_matches_full()
{
    std::mt19937_64 rng(17);
    std::uniform_int_distribution<std::size_t> tok(0, 10);

    nn::GPTModel learned(11, 8, 16, 2, 12, 2, /*use_rope=*/false);
    nn::GPTModel rope(11, 8, 16, 2, 12, 2, /*use_rope=*/true);

    for (std::size_t len : {1u, 3u, 8u, 15u})
    {
        std::vector<std::size_t> prompt(len);
        for (auto &t : prompt)
            t = tok(rng);
        check_kv_matches_full(learned, prompt);
        check_kv_matches_full(rope, prompt);
    }
    std::puts("  PASSED  kv_cache_matches_full");
}

// 贪心生成：KV-cache 路径与全量重前向路径 token 序一致
static void test_generate_kv_matches_full()
{
    nn::GPTModel model(11, 8, 16, 2, 12, 2, true);
    const std::vector<std::size_t> prompt{1, 4, 4, 2};

    const auto a = model.generate(prompt, 6, /*temperature=*/0.0, /*use_kv_cache=*/true);
    const auto b = model.generate(prompt, 6, /*temperature=*/0.0, /*use_kv_cache=*/false);
    assert(a.size() == 6 && b.size() == 6);
    for (std::size_t i = 0; i < a.size(); ++i)
        assert(a[i] == b[i]);
    std::puts("  PASSED  generate_kv_matches_full");
}

// RoPE 无长度限制：生成超出 seq_len_ 不抛错（学习式绝对嵌入会钳位，RoPE 自然外推）
static void test_generate_rope_beyond_seq_len()
{
    nn::GPTModel model(9, 8, 4, 2, 12, 1, /*use_rope=*/true);
    const auto out = model.generate({1, 2, 3}, 8, 0.0, true);
    assert(out.size() == 8);
    for (const auto t : out)
        assert(t < 9);
    std::puts("  PASSED  generate_rope_beyond_seq_len");
}

static void test_generate_empty_prompt_throws()
{
    nn::GPTModel model(9, 8, 4, 2, 12, 1);
    bool threw = false;
    try
    {
        (void)model.generate({}, 3);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    assert(threw);
    std::puts("  PASSED  generate_empty_prompt_throws");
}

struct TestEntry {
    const char *name;
    void (*fn)();
};

static const TestEntry tests[] = {
    {"causal_self_attention_shape", test_causal_self_attention_shape},
    {"causal_self_attention_masking", test_causal_self_attention_masking},
    {"gpt_block_shape", test_gpt_block_shape},
    {"gpt_model_shape", test_gpt_model_shape},
    {"gpt_model_numeric_grad", test_gpt_model_numeric_grad},
    {"gpt_model_generate", test_gpt_model_generate},
    {"rope_known_rotation", test_rope_known_rotation},
    {"rope_preserves_norm", test_rope_preserves_norm},
    {"alibi_single_token_no_bias", test_alibi_single_token_no_bias},
    {"alibi_causality_and_effect", test_alibi_causality_and_effect},
    {"doc_mask_isolation", test_doc_mask_isolation},
    {"kv_cache_matches_full", test_kv_cache_matches_full},
    {"generate_kv_matches_full", test_generate_kv_matches_full},
    {"generate_rope_beyond_seq_len", test_generate_rope_beyond_seq_len},
    {"generate_empty_prompt_throws", test_generate_empty_prompt_throws}
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
