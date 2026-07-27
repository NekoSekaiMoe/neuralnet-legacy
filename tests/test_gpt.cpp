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
    nn::CausalSelfAttention attn(16, 4);
    nn::Matrix input(16, 10);
    for (std::size_t i = 0; i < input.size(); ++i)
        input.data()[i] = 1.0;

    nn::Matrix out = attn.forward(input);
    assert(out.rows() == 16);
    assert(out.cols() == 10);
}

static void test_causal_self_attention_masking()
{
    nn::CausalSelfAttention attn(8, 2);
    nn::Matrix input1(8, 4);
    nn::Matrix input2(8, 4);

    for (std::size_t i = 0; i < 8; ++i)
    {
        for (std::size_t j = 0; j < 4; ++j)
        {
            double val = static_cast<double>(i + j * 10);
            input1.set_value(i, j, val);
            input2.set_value(i, j, val);
        }
    }

    // Change a future token in input2
    for (std::size_t i = 0; i < 8; ++i)
        input2.set_value(i, 3, 999.0);

    nn::Matrix out1 = attn.forward(input1);
    nn::Matrix out2 = attn.forward(input2);

    // The output at pos 0, 1, 2 should be identical for both inputs because pos 3 is masked.
    for (std::size_t j = 0; j < 3; ++j)
    {
        for (std::size_t i = 0; i < 8; ++i)
        {
            assert(approx(out1.at(i, j), out2.at(i, j)));
        }
    }
}

static void test_gpt_block_shape()
{
    nn::GPTBlock block(16, 4, 32);
    nn::Matrix input(16, 10);
    for (std::size_t i = 0; i < input.size(); ++i)
        input.data()[i] = 0.5;

    nn::Matrix out = block.forward(input);
    assert(out.rows() == 16);
    assert(out.cols() == 10);

    nn::Matrix grad_out(16, 10, 1.0);
    nn::Matrix grad_in = block.backward(grad_out);
    assert(grad_in.rows() == 16);
    assert(grad_in.cols() == 10);
}

static void test_gpt_model_shape()
{
    // vocab_size=100, d_model=16, seq_len=10, num_heads=4, d_ff=32, num_layers=2
    nn::GPTModel model(100, 16, 10, 4, 32, 2);
    nn::Matrix input(5, 2); // seq_len=5, batch_size=2
    for (std::size_t i = 0; i < input.size(); ++i)
        input.data()[i] = static_cast<double>(i % 100);

    nn::Matrix out = model.forward(input);
    assert(out.rows() == 100);
    assert(out.cols() == 10); // seq_len=5 * batch_size=2 = 10

    nn::Matrix grad_out(100, 10, 0.1);
    nn::Matrix grad_in = model.backward(grad_out);
    assert(grad_in.rows() == 5);
    assert(grad_in.cols() == 2);
}

static void test_gpt_model_generate()
{
    nn::GPTModel model(100, 16, 10, 4, 32, 2);
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
