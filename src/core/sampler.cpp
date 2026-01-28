#include "core/sampler.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>

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

    // Top-k sampling (optimized)
    if (params.top_k > 0) {
        return sample_top_k({modified_logits, logits.vocab_size},
                           params.temperature, static_cast<int32_t>(params.top_k), rng);
    }

    // Top-p (nucleus) sampling
    return sample_top_p({modified_logits, logits.vocab_size},
                       params.temperature, params.top_p, rng);
}

SampleResult Sampler::sample_greedy(const backend::Logits& logits) const {
    // Optimized greedy: just find argmax, no need for full softmax
    auto max_it = std::max_element(logits.data.begin(), logits.data.end());
    int32_t token_id = static_cast<int32_t>(std::distance(logits.data.begin(), max_it));
    float max_logit = *max_it;

    // For greedy, probability is approximately 1.0 after softmax
    // We skip computing full softmax since it's expensive and not needed
    // The probability returned is just for logging/debugging
    return SampleResult{
        .token_id = token_id,
        .probability = 1.0f,  // Approximate - greedy always picks max
        .logit = max_logit
    };
}

SampleResult Sampler::sample_top_p(
    const backend::Logits& logits,
    double /*temperature*/,  // Already applied
    double top_p,
    std::mt19937& rng) const {

    // Optimization: if top_p is very high (e.g., 0.99+), we can limit search
    // Use partial sort with a reasonable upper bound to avoid full sort

    const size_t vocab_size = logits.data.size();

    // Heuristic: for high top_p, we rarely need more than a few hundred tokens
    // For low top_p (e.g., 0.1), we might only need top 10-20
    size_t max_candidates = std::min(vocab_size,
        top_p >= 0.9 ? size_t(1000) : size_t(256));

    // Find top candidates using partial sort
    std::vector<size_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(),
                     indices.begin() + max_candidates,
                     indices.end(),
                     [&logits](size_t a, size_t b) {
                         return logits.data[a] > logits.data[b];
                     });

    // Compute softmax only for top candidates
    float max_logit = logits.data[indices[0]];
    std::vector<float> exp_logits(max_candidates);
    float sum_exp = 0.0f;

    for (size_t i = 0; i < max_candidates; ++i) {
        exp_logits[i] = std::exp(logits.data[indices[i]] - max_logit);
        sum_exp += exp_logits[i];
    }

    // Find nucleus (tokens whose cumulative probability >= top_p)
    float cumsum = 0.0f;
    size_t nucleus_size = 0;
    float top_p_f = static_cast<float>(top_p);

    for (size_t i = 0; i < max_candidates; ++i) {
        float prob = exp_logits[i] / sum_exp;
        cumsum += prob;
        nucleus_size = i + 1;
        if (cumsum >= top_p_f) {
            break;
        }
    }

    // Renormalize probabilities in nucleus
    std::vector<float> nucleus_probs(nucleus_size);
    float nucleus_sum = 0.0f;
    for (size_t i = 0; i < nucleus_size; ++i) {
        nucleus_probs[i] = exp_logits[i];
        nucleus_sum += nucleus_probs[i];
    }
    for (float& p : nucleus_probs) {
        p /= nucleus_sum;
    }

    // Sample from nucleus
    std::discrete_distribution<size_t> dist(nucleus_probs.begin(), nucleus_probs.end());
    size_t sampled_idx = dist(rng);
    int32_t token_id = static_cast<int32_t>(indices[sampled_idx]);

    return SampleResult{
        .token_id = token_id,
        .probability = nucleus_probs[sampled_idx],
        .logit = logits.data[token_id]
    };
}

SampleResult Sampler::sample_top_k(
    const backend::Logits& logits,
    double /*temperature*/,
    int32_t top_k,
    std::mt19937& rng) const {

    const size_t vocab_size = logits.data.size();
    size_t k = std::min(static_cast<size_t>(top_k), vocab_size);

    // Optimized: use partial_sort to get only top-k elements
    // This is O(n log k) instead of O(n log n) for full sort
    std::vector<size_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(),
                     indices.begin() + k,
                     indices.end(),
                     [&logits](size_t a, size_t b) {
                         return logits.data[a] > logits.data[b];
                     });

    // Compute softmax only for top-k elements (not full vocabulary)
    float max_logit = logits.data[indices[0]];
    std::vector<float> exp_logits(k);
    float sum_exp = 0.0f;

    for (size_t i = 0; i < k; ++i) {
        exp_logits[i] = std::exp(logits.data[indices[i]] - max_logit);
        sum_exp += exp_logits[i];
    }

    // Normalize to probabilities
    std::vector<float> top_k_probs(k);
    for (size_t i = 0; i < k; ++i) {
        top_k_probs[i] = exp_logits[i] / sum_exp;
    }

    // Sample
    std::discrete_distribution<size_t> dist(top_k_probs.begin(), top_k_probs.end());
    size_t sampled_idx = dist(rng);
    int32_t token_id = static_cast<int32_t>(indices[sampled_idx]);

    return SampleResult{
        .token_id = token_id,
        .probability = top_k_probs[sampled_idx],
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
