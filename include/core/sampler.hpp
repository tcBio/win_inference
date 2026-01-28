#pragma once

#include "backend/runtime_interface.hpp"
#include "utils/result.hpp"

#include <random>
#include <optional>
#include <vector>
#include <cstdint>

namespace qwen::core {

/// Sampling parameters
struct SamplingParams {
    double temperature = 1.0;       // [0, 2] - 0 for greedy
    double top_p = 1.0;             // [0, 1] - nucleus sampling
    double top_k = 0;               // 0 = disabled
    double repetition_penalty = 1.0;
    std::optional<int64_t> seed;    // For deterministic sampling
};

/// Sampler result
struct SampleResult {
    int32_t token_id;
    float probability;
    float logit;
};

/// Token sampler with various strategies
/// Thread-safe: each call can use different RNG state
class Sampler {
public:
    Sampler();

    /// Sample a token from logits
    [[nodiscard]] Result<SampleResult> sample(
        const backend::Logits& logits,
        const SamplingParams& params,
        const std::vector<int32_t>& previous_tokens = {}) const;

    /// Greedy sampling (argmax)
    [[nodiscard]] SampleResult sample_greedy(const backend::Logits& logits) const;

    /// Top-p (nucleus) sampling
    [[nodiscard]] SampleResult sample_top_p(
        const backend::Logits& logits,
        double temperature,
        double top_p,
        std::mt19937& rng) const;

    /// Top-k sampling
    [[nodiscard]] SampleResult sample_top_k(
        const backend::Logits& logits,
        double temperature,
        int32_t top_k,
        std::mt19937& rng) const;

    /// Apply temperature scaling to logits
    static void apply_temperature(std::vector<float>& logits, double temperature);

    /// Apply repetition penalty to logits
    static void apply_repetition_penalty(
        std::vector<float>& logits,
        const std::vector<int32_t>& previous_tokens,
        double penalty);

    /// Compute softmax probabilities
    static std::vector<float> softmax(const std::vector<float>& logits);

private:
    mutable std::mutex rng_mutex_;
    mutable std::mt19937 default_rng_;
};

}  // namespace qwen::core
