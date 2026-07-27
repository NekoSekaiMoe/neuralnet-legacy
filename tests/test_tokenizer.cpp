#include <neuralnet/tokenizer.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <string>

static void test_char_tokenizer()
{
    nn::CharTokenizer tok;
    std::string text = "Hello World!";
    auto encoded = tok.encode(text);
    assert(encoded.size() == text.size());
    std::string decoded = tok.decode(encoded);
    assert(decoded == text);
}

static void test_byte_zip_tokenizer_train()
{
    nn::ByteZipTokenizer tok;
    std::string text = "this is a test text for byte zip tokenizer. this is a test text.";
    nn::ByteZipTokenizer::Config config;
    config.vocab_size = 300;
    config.v1_max_len = 4;
    config.log = [](std::string_view){}; // silent
    tok.train(text, config);
    
    auto encoded = tok.encode(text);
    assert(!encoded.empty());
    std::string decoded = tok.decode(encoded);
    assert(decoded == text);
}

struct TestEntry {
    const char *name;
    void (*fn)();
};

static const TestEntry tests[] = {
    {"char_tokenizer", test_char_tokenizer},
    {"byte_zip_tokenizer_train", test_byte_zip_tokenizer_train}
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
