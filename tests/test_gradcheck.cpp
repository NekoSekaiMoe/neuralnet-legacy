// ═════════════════════════════════════════════════════════════════════════════
//  test_gradcheck.cpp — 数值梯度检查（移植自上游 attn/gpt/conv2d/layer gradcheck）
//
//  方法论：取标量损失 L = Σ (layer(x) ∘ R)，R 为固定随机上游梯度；
//  对参数/输入逐元素做中心差分 (L(p+ε) − L(p−ε)) / 2ε，
//  与 backward() 的解析梯度对比（相对误差 < tol）。
// ═════════════════════════════════════════════════════════════════════════════

#include <neuralnet/nn/nn.h>
#include <neuralnet/layer.h>

#include "test_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    // 断言改抛异常：assert 失败会 abort()，无法被 run_tests 捕获输出统一的
    // FAILED 汇总；抛 runtime_error 可报告失败信息并继续跑同进程的其他用例
    // （CMake 已加 -UNDEBUG，Release 下本也可生效，这里是可观测性问题）。
    void require(bool cond, const std::string &msg)
    {
        if (!cond)
            throw std::runtime_error(msg);
    }

    std::mt19937_64 &grad_rng()
    {
        static std::mt19937_64 rng(20240718);
        return rng;
    }

    nn::Matrix random_matrix(std::size_t rows, std::size_t cols, double scale = 1.0)
    {
        nn::Matrix m(rows, cols);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (std::size_t i = 0; i < m.size(); ++i)
            m.data()[i] = scale * dist(grad_rng());
        return m;
    }

    // L = Σ (out ∘ R)
    double loss_of(nn::Layer &layer, const nn::Matrix &input, const nn::Matrix &upstream)
    {
        nn::Matrix out = layer.forward(input);
        double s = 0.0;
        for (std::size_t i = 0; i < out.size(); ++i)
            s += out.data()[i] * upstream.data()[i];
        return s;
    }

    // 相对误差（分母取 max(1, |a|+|b|) 防止除零）
    double rel_err(double a, double b)
    {
        return std::fabs(a - b) / std::max(1.0, std::fabs(a) + std::fabs(b));
    }

    // ── 参数梯度检查：对第 pidx 个参数的全部（或抽样 max_elem 个）元素 ──
    void check_param_grad(nn::Layer &layer, const nn::Matrix &input,
                          const nn::Matrix &upstream, std::size_t pidx,
                          const char *what, double tol = 2e-5,
                          double eps = 1e-6, std::size_t max_elem = 64)
    {
        auto params = layer.parameters();
        auto grads = layer.param_gradients();
        require(pidx < params.size(), "param index out of range");

        nn::Matrix &p = params[pidx].get();

        layer.forward(input);
        layer.backward(upstream);
        std::vector<double> analytic(p.size());
        for (std::size_t i = 0; i < p.size(); ++i)
            analytic[i] = grads[pidx].get().data()[i];

        std::vector<std::size_t> elems;
        if (p.size() <= max_elem)
        {
            elems.resize(p.size());
            for (std::size_t i = 0; i < p.size(); ++i)
                elems[i] = i;
        }
        else
        {
            std::uniform_int_distribution<std::size_t> pick(0, p.size() - 1);
            elems.reserve(max_elem);
            for (std::size_t k = 0; k < max_elem; ++k)
                elems.push_back(pick(grad_rng()));
        }

        double worst = 0.0;
        for (const std::size_t i : elems)
        {
            const double old = p.data()[i];
            p.data()[i] = old + eps;
            const double lp = loss_of(layer, input, upstream);
            p.data()[i] = old - eps;
            const double lm = loss_of(layer, input, upstream);
            p.data()[i] = old;
            const double numeric = (lp - lm) / (2.0 * eps);
            const double e = rel_err(numeric, analytic[i]);
            // std::max 与 NaN 比较恒为 false → NaN/Inf 误差会被悄悄忽略；
            // 汇总前逐项验证有限性，让发散梯度直接失败
            require(std::isfinite(e),
                    std::string(what) + ": param[" + std::to_string(pidx)
                        + "] elem " + std::to_string(i) + " rel err not finite");
            worst = std::max(worst, e);
        }
        std::printf("  [gradcheck] %-28s param[%zu] worst rel err = %.2e\n",
                    what, pidx, worst);
        require(worst < tol,
                std::string(what) + ": param[" + std::to_string(pidx)
                    + "] rel err " + std::to_string(worst)
                    + " >= tol " + std::to_string(tol));
    }

    // ── 输入梯度检查（抽样 max_elem 个元素）────────────────────────────
    void check_input_grad(nn::Layer &layer, const nn::Matrix &input,
                          const nn::Matrix &upstream, const char *what,
                          double tol = 2e-5, double eps = 1e-6,
                          std::size_t max_elem = 48)
    {
        nn::Matrix x = input;
        layer.forward(x);
        nn::Matrix analytic = layer.backward(upstream);

        std::uniform_int_distribution<std::size_t> pick(0, x.size() - 1);
        double worst = 0.0;
        for (std::size_t k = 0; k < std::min(max_elem, x.size()); ++k)
        {
            const std::size_t i = pick(grad_rng());
            const double old = x.data()[i];
            x.data()[i] = old + eps;
            const double lp = loss_of(layer, x, upstream);
            x.data()[i] = old - eps;
            const double lm = loss_of(layer, x, upstream);
            x.data()[i] = old;
            const double numeric = (lp - lm) / (2.0 * eps);
            const double e = rel_err(numeric, analytic.data()[i]);
            require(std::isfinite(e),
                    std::string(what) + ": input elem " + std::to_string(i)
                        + " rel err not finite");
            worst = std::max(worst, e);
        }
        std::printf("  [gradcheck] %-28s input     worst rel err = %.2e\n",
                    what, worst);
        require(worst < tol,
                std::string(what) + ": input rel err " + std::to_string(worst)
                    + " >= tol " + std::to_string(tol));
    }

    // ════════════════════════════════════════════════════════════════════
    //  用例
    // ════════════════════════════════════════════════════════════════════

    void test_gradcheck_linear()
    {
        nn::Linear linear(6, 4);
        nn::Matrix x = random_matrix(6, 5);
        nn::Matrix r = random_matrix(4, 5);
        check_param_grad(linear, x, r, 0, "Linear(W)");
        check_param_grad(linear, x, r, 1, "Linear(b)");
        check_input_grad(linear, x, r, "Linear");
    }

    void test_gradcheck_gelu()
    {
        nn::GELU gelu;
        nn::Matrix x = random_matrix(8, 6, 0.8); // 控制 |x| 避免 eps 落入死区
        nn::Matrix r = random_matrix(8, 6);
        check_input_grad(gelu, x, r, "GELU");
    }

    void test_gradcheck_softmax()
    {
        nn::Softmax sm;
        nn::Matrix x = random_matrix(7, 4);
        nn::Matrix r = random_matrix(7, 4);
        check_input_grad(sm, x, r, "Softmax");
    }

    void test_gradcheck_layernorm()
    {
        nn::LayerNorm ln(8);
        nn::Matrix x = random_matrix(8, 4);
        nn::Matrix r = random_matrix(8, 4);
        check_param_grad(ln, x, r, 0, "LayerNorm(gamma)");
        check_param_grad(ln, x, r, 1, "LayerNorm(beta)");
        check_input_grad(ln, x, r, "LayerNorm");
    }

    void test_gradcheck_rmsnorm()
    {
        nn::RMSNorm rms(8);
        nn::Matrix x = random_matrix(8, 4);
        nn::Matrix r = random_matrix(8, 4);
        check_param_grad(rms, x, r, 0, "RMSNorm(gain)");
        check_input_grad(rms, x, r, "RMSNorm");
    }

    void test_gradcheck_swiglu()
    {
        nn::SwiGLU swiglu(5);
        nn::Matrix x = random_matrix(10, 3);
        nn::Matrix r = random_matrix(5, 3);
        check_input_grad(swiglu, x, r, "SwiGLU");
    }

    void test_gradcheck_feedforward()
    {
        nn::FeedForward ffn(8, 12);
        nn::Matrix x = random_matrix(8, 4);
        nn::Matrix r = random_matrix(8, 4);
        check_param_grad(ffn, x, r, 0, "FeedForward(W1)");
        check_input_grad(ffn, x, r, "FeedForward");
    }

    void test_gradcheck_mha()
    {
        nn::MultiHeadAttention mha(8, 2, /*causal=*/false);
        nn::Matrix x = random_matrix(8, 5);
        nn::Matrix r = random_matrix(8, 5);
        for (std::size_t pi = 0; pi < 4; ++pi)
            check_param_grad(mha, x, r, pi, "MHA");
        check_input_grad(mha, x, r, "MHA");
    }

    void test_gradcheck_mha_causal()
    {
        nn::MultiHeadAttention mha(8, 2, /*causal=*/true);
        nn::Matrix x = random_matrix(8, 5);
        nn::Matrix r = random_matrix(8, 5);
        for (std::size_t pi = 0; pi < 4; ++pi)
            check_param_grad(mha, x, r, pi, "MHA(causal)");
        check_input_grad(mha, x, r, "MHA(causal)");
    }

    // RoPE 反向（逆旋转）与正向旋转的复合正确性
    void test_gradcheck_mha_rope()
    {
        nn::MultiHeadAttention mha(8, 2, /*causal=*/true, /*use_rope=*/true);
        nn::Matrix x = random_matrix(8, 5);
        nn::Matrix r = random_matrix(8, 5);
        for (std::size_t pi = 0; pi < 4; ++pi)
            check_param_grad(mha, x, r, pi, "MHA(rope)");
        check_input_grad(mha, x, r, "MHA(rope)");
    }

    void test_gradcheck_mha_alibi()
    {
        nn::MultiHeadAttention mha(8, 2, /*causal=*/true, /*use_rope=*/false,
                                   /*use_alibi=*/true);
        nn::Matrix x = random_matrix(8, 5);
        nn::Matrix r = random_matrix(8, 5);
        for (std::size_t pi = 0; pi < 4; ++pi)
            check_param_grad(mha, x, r, pi, "MHA(alibi)");
        check_input_grad(mha, x, r, "MHA(alibi)");
    }

    void test_gradcheck_gptblock_rope()
    {
        nn::GPTBlock block(8, 2, 12, /*use_rope=*/true);
        nn::Matrix x = random_matrix(8, 4);
        nn::Matrix r = random_matrix(8, 4);
        // 参数序：W_q, W_k, W_v, W_o, LN1(γ,β), FFN(W1,b1,W2,b2), LN2(γ,β)
        check_param_grad(block, x, r, 0, "GPTBlock(rope,Wq)");
        check_param_grad(block, x, r, 3, "GPTBlock(rope,Wo)");
        check_input_grad(block, x, r, "GPTBlock(rope)");
    }

    void test_gradcheck_gptmodel_token_emb()
    {
        nn::GPTModel model(/*vocab=*/11, /*d_model=*/8, /*seq_len=*/6,
                           /*num_heads=*/2, /*d_ff=*/12, /*num_layers=*/1,
                           /*use_rope=*/true);
        // 输入 (sl, batch) = token ids
        nn::Matrix in(4, 1);
        in.set_value_unchecked(0, 0, 3.0);
        in.set_value_unchecked(1, 0, 7.0);
        in.set_value_unchecked(2, 0, 1.0);
        in.set_value_unchecked(3, 0, 9.0);
        nn::Matrix r = random_matrix(11, 4);
        // token_emb（param 0）：RoPE 模型下 pos_emb（param 1）梯度恒为 0
        check_param_grad(model, in, r, 0, "GPTModel(tok_emb)", 3e-5, 1e-6, 16);

        auto grads = model.param_gradients();
        const nn::Matrix &pos_grad = grads[1].get();
        for (std::size_t i = 0; i < pos_grad.size(); ++i)
            require(pos_grad.data()[i] == 0.0,
                    "pos_emb grad must be zero under RoPE (i=" + std::to_string(i)
                        + ")"); // RoPE 下位置嵌入不参与前向
    }

} // namespace

// ── 测试注册表 ──────────────────────────────────────────────────────────────
static const TestEntry all_tests[] = {
    {"gradcheck_linear", test_gradcheck_linear},
    {"gradcheck_gelu", test_gradcheck_gelu},
    {"gradcheck_softmax", test_gradcheck_softmax},
    {"gradcheck_layernorm", test_gradcheck_layernorm},
    {"gradcheck_rmsnorm", test_gradcheck_rmsnorm},
    {"gradcheck_swiglu", test_gradcheck_swiglu},
    {"gradcheck_feedforward", test_gradcheck_feedforward},
    {"gradcheck_mha", test_gradcheck_mha},
    {"gradcheck_mha_causal", test_gradcheck_mha_causal},
    {"gradcheck_mha_rope", test_gradcheck_mha_rope},
    {"gradcheck_mha_alibi", test_gradcheck_mha_alibi},
    {"gradcheck_gptblock_rope", test_gradcheck_gptblock_rope},
    {"gradcheck_gptmodel_token_emb", test_gradcheck_gptmodel_token_emb},
};

int main(int argc, char **argv)
{
    return run_tests(argc, argv, all_tests,
                     sizeof(all_tests) / sizeof(all_tests[0]));
}
