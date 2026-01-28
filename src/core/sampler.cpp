#include "core/sampler.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>

namespace qwen::core {

Sampler::Sampler() : default_rng_(std::random_device{}()) {}

Result<SampleResult> Sampler::sample(
    const backend::Logits& logits,
    const SamplingParams& params,
    const std::vector<int32_t>& previous_tokens) const {

    if (logits.empty()) {
        return Error::internal("Empty logits");
    }

    // Copy logits for modification
    std::vector<float> modified_logits = logits.data;

    // Apply repetition penalty
    if (params.repetition_penalty != 1.0 && !previous_tokens.empty()) {
        apply_repetition_penalty(modified_logits, previous_tokens, params.repetition_penalty);
    }

    // Temperature = 0 means greedy
    if (params.temperature <= 0.0) {
        return sample_greedy({modified_logits, logits.vocab_size});
    }

    // Apply temperature
    apply_temperature(modified_logits, params.temperature);

    // Create RNG with seed if provided
    std::mt19937 rng;
    if (params.seed.has_value()) {
        rng.seed(static_cast<uint32_t>(*params.seed));
    } else {
        std::lock_guard<std::mutex> lock(rng_mutex_);
        rng.seed(default_rng_());
    }

    // Top-k sampling
    if (params.top_k > 0) {
        return sample_top_k({modified_logits, logits.vocab_size},
                           params.temperature, static_cast<int32_t>(params.top_k), rng);
    }

    // Top-p (nucleus) sampling
    return sample_top_p({modified_logits, logits.vocab_size},
                       params.temperature, params.top_p, rng);
}

SampleResult Sampler::sample_greedy(const backend::Logits& logits) const {
    auto max_it = std::max_element(logits.data.begin(), logits.data.end());
    int32_t token_id = static_cast<int32_t>(std::distance(logits.data.begin(), max_it));

    // Compute probability via softmax
    auto probs = softmax(logits.data);

    return SampleResult{
        .token_id = token_id,
        .probability = probs[token_id],
        .logit = *max_it
    };
}

SampleResult Sampler::sample_top_p(
    const backend::Logits& logits,
    double /*temperature*/,  // Already applied
    double top_p,
    std::mt19937& rng) const {

    auto probs = softmax(logits.data);

    // Create sorted indices
    std::vector<size_t> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });

    // Find nucleus (tokens whose cumulative probability >= top_p)
    float cumsum = 0.0f;
    size_t nucleus_size = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        cumsum += probs[indices[i]];
        nucleus_size = i + 1;
        if (cumsum >= static_cast<float>(top_p)) {
            break;
        }
    }

    // Renormalize probabilities in nucleus
    std::vector<float> nucleus_probs(nucleus_size);
    float sum = 0.0f;
    for (size_t i = 0; i < nucleus_size; ++i) {
        nucleus_probs[i] = probs[indices[i]];
        sum += nucleus_probs[i];
    }
    for (float& p : nucleus_probs) {
        p /= sum;
    }

    // Sample from nucleus
    std::discrete_distribution<size_t> dist(nucleus_probs.begin(), nucleus_probs.end());
    size_t sampled_idx = dist(rng);
    int32_t token_id = static_cast<int32_t>(indices[sampled_idx]);

    return SampleResult{
        .token_id = token_id,
        .probability = probs[token_id],
        .logit = logits.data[token_id]
    };
}

SampleResult Sampler::sample_top_k(
    const backend::Logits& logits,
    double /*temperature*/,
    int32_t top_k,
    std::mt19937& rng) const {

    auto probs = softmax(logits.data);

    // Find top-k indices
    std::vector<size_t> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(),
                     indices.begin() + std::min(static_cast<size_t>(top_k), indices.size()),
                     indices.end(),
                     [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });

    size_t k = std::min(static_cast<size_t>(top_k), indices.size());

    // Renormalize top-k probabilities
    std::vector<float> top_k_probs(k);
    float sum = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        top_k_probs[i] = probs[indices[i]];
        sum += top_k_probs[i];
    }
    for (float& p : top_k_probs) {
        p /= sum;
    }

    // Sample
    std::discrete_distribution<size_t> dist(top_k_probs.begin(), top_k_probs.end());
    size_t sampled_idx = dist(rng);
    int32_t token_id = static_cast<int32_t>(indices[sampled_idx]);

    return SampleResult{
        .token_id = token_id,
        .probability = probs[token_id],
        .logit = logits.data[token_id]
    };
}

void Sampler::apply_temperature(std::vector<float>& logits, double temperature) {
    if (temperature == 1.0) return;

    float temp = static_cast<float>(temperature);
    for (float& logit : logits) {
        logit /= temp;
    }
}

void Sampler::apply_repetition_penalty(
    std::vector<float>& logits,
    const std::vector<int32_t>& previous_tokens,
    double penalty) {

    float penalty_f = static_cast<float>(penalty);
    for (int32_t token : previous_tokens) {
        if (token >= 0 && static_cast<size_t>(token) < logits.size()) {
            if (logits[token] > 0) {
                logits[token] /= penalty_f;
            } else {
                logits[token] *= penalty_f;
            }
        }
    }
}

std::vector<float> Sampler::softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());

    // Find max for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());

    // Compute exp(logit - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }

    // Normalize
    for (float& p : probs) {
        p /= sum;
    }

    return probs;
}

}  // namespace qwen::core
