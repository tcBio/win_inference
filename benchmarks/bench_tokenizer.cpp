#include "tokenizer/tokenizer.hpp"
#include "tokenizer/prompt_builder.hpp"

#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>

using namespace qwen;

struct BenchResult {
    double mean_us;
    double std_us;
    double min_us;
    double max_us;
    size_t iterations;
};

template <typename Func>
BenchResult benchmark(const std::string& name, Func&& func, size_t iterations = 1000) {
    std::vector<double> times;
    times.reserve(iterations);

    // Warmup
    for (size_t i = 0; i < 10; ++i) {
        func();
    }

    // Measure
    for (size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::micro>(end - start).count();
        times.push_back(duration);
    }

    // Calculate stats
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();

    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double std = std::sqrt(sq_sum / times.size());

    double min = *std::min_element(times.begin(), times.end());
    double max = *std::max_element(times.begin(), times.end());

    std::cout << name << ": "
              << "mean=" << mean << "us, "
              << "std=" << std << "us, "
              << "min=" << min << "us, "
              << "max=" << max << "us\n";

    return {mean, std, min, max, iterations};
}

int main() {
    std::cout << "=== Tokenizer Benchmarks ===\n\n";

    // Initialize tokenizer
    auto tokenizer = tokenizer::create_qwen_tokenizer();
    tokenizer::TokenizerConfig config;
    tokenizer->initialize(config);

    auto prompt_builder = std::make_shared<tokenizer::PromptBuilder>(
        tokenizer,
        tokenizer::PromptBuilderConfig{}
    );

    // Test inputs of various sizes
    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"short_10chars", "Hello world"},
        {"medium_100chars", std::string(100, 'a')},
        {"long_1000chars", std::string(1000, 'a')},
        {"very_long_10000chars", std::string(10000, 'a')}
    };

    std::cout << "--- Encode Benchmarks ---\n";
    for (const auto& [name, text] : test_cases) {
        benchmark("encode_" + name, [&]() {
            tokenizer->encode(text);
        });
    }

    std::cout << "\n--- Decode Benchmarks ---\n";
    // Create token sequences of various lengths
    std::vector<int32_t> short_tokens(10, 100);
    std::vector<int32_t> medium_tokens(100, 100);
    std::vector<int32_t> long_tokens(1000, 100);

    benchmark("decode_10_tokens", [&]() {
        tokenizer->decode(short_tokens);
    });

    benchmark("decode_100_tokens", [&]() {
        tokenizer->decode(medium_tokens);
    });

    benchmark("decode_1000_tokens", [&]() {
        tokenizer->decode(long_tokens);
    });

    std::cout << "\n--- Prompt Builder Benchmarks ---\n";

    std::vector<api::Message> single_msg = {{api::Role::User, "Hello"}};
    std::vector<api::Message> multi_msg = {
        {api::Role::System, "You are a helpful assistant."},
        {api::Role::User, "What is 2+2?"},
        {api::Role::Assistant, "4"},
        {api::Role::User, "What is 3+3?"}
    };
    std::vector<api::Message> long_msg = {
        {api::Role::System, std::string(1000, 'a')},
        {api::Role::User, std::string(1000, 'b')}
    };

    benchmark("prompt_single_message", [&]() {
        prompt_builder->build(single_msg);
    });

    benchmark("prompt_multi_turn", [&]() {
        prompt_builder->build(multi_msg);
    });

    benchmark("prompt_long_content", [&]() {
        prompt_builder->build(long_msg);
    });

    std::cout << "\n=== Benchmarks Complete ===\n";
    return 0;
}
