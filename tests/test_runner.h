#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <exception>

struct TestEntry {
    const char* name;
    void (*fn)();
};

inline int run_tests(int argc, char** argv, const TestEntry* tests, std::size_t num_tests) {
    if (argc < 2) {
        std::size_t passed = 0;
        for (std::size_t i = 0; i < num_tests; ++i) {
            const auto& t = tests[i];
            try { t.fn(); ++passed; std::cout << "  PASSED  " << t.name << "\n"; }
            catch (const std::exception& e) { std::cout << "  FAILED  " << t.name << " : " << e.what() << "\n"; }
        }
        std::cout << passed << "/" << num_tests << " passed\n";
        return passed == num_tests ? 0 : 1;
    }
    std::string test_name = argv[1];
    for (std::size_t i = 0; i < num_tests; ++i) {
        const auto& t = tests[i];
        if (test_name == t.name) {
            try { t.fn(); std::cout << "  PASSED  " << t.name << "\n"; return 0; }
            catch (const std::exception& e) { std::cout << "  FAILED  " << t.name << " : " << e.what() << "\n"; return 1; }
        }
    }
    std::cerr << "Unknown test: " << test_name << "\n";
    return 1;
}
