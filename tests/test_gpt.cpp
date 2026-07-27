#include <neuralnet/nn/nn.h>
#include <neuralnet/layer.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static bool approx(double a, double b, double tol = 1e-6)
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
    {"gpt_model_generate", test_gpt_model_generate}
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
