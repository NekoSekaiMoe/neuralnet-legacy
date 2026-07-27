#include <neuralnet/tokenizer.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>

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

static void test_bpe_tokenizer_load_vocab()
{
    std::string test_vocab_path = "test_bpe_vocab.json";
    {
        std::ofstream ofs(test_vocab_path);
        ofs << "{\n";
        ofs << "  \"vocab\": {\n";
        ofs << "    \"<unk>\": 0,\n";
        ofs << "    \"<pad>\": 1,\n";
        ofs << "    \"<num>\": 2,\n";
        ofs << "    \"hello\": 3,\n";
        ofs << "    \"world\": 4,\n";
        ofs << "    \"\\n\": 5,\n";
        ofs << "    \"\\u0041\": 6,\n";       // A
        ofs << "    \"\\u4e2d\": 7,\n";       // 中
        ofs << "    \"8\": \"74657374\"\n";   // test
        ofs << "  }\n";
        ofs << "}\n";
    }

    nn::BPETokenizer tok;
    tok.load_vocab(test_vocab_path);

    assert(tok.vocab_size() == 9);

    std::vector<std::size_t> ids = {3, 4, 6, 7, 8};
    std::string decoded = tok.decode(ids);
    // 3="hello" (len 5 -> word)
    // 4="world" (len 5 -> word)
    // 6="A" (len 1 -> not word)
    // 7="中" (len 3 -> word)
    // 8="test" (len 4 -> word)
    // Expected output logic in decode:
    // 3: "hello"
    // 4: " hello world" (prepend space before word)
    // 6: " hello worldA" (no space before non-word)
    // 7: " hello worldA 中" (prepend space before word)
    // 8: " hello worldA 中 test" (prepend space before word)
    assert(decoded == "hello worldA 中 test");

    // Test encode / decode round trip for words
    auto encoded = tok.encode("hello world test");
    assert(encoded.size() == 3);
    assert(encoded[0] == 3);
    assert(encoded[1] == 4);
    assert(encoded[2] == 8);
    std::string round_trip = tok.decode(encoded);
    assert(round_trip == "hello world test");

    std::remove(test_vocab_path.c_str());
}

struct TestEntry {
    const char *name;
    void (*fn)();
};

static const TestEntry tests[] = {
    {"char_tokenizer", test_char_tokenizer},
    {"byte_zip_tokenizer_train", test_byte_zip_tokenizer_train},
    {"bpe_tokenizer_load_vocab", test_bpe_tokenizer_load_vocab}
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
