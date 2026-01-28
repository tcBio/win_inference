#include "backend/null_backend.hpp"
#include "utils/logger.hpp"

#include <chrono>
#include <thread>
#include <cmath>

namespace qwen::backend {

NullBackend::NullBackend() : rng_(42) {}

Result<void> NullBackend::initialize(const ModelConfig& config) {
    config_ = config;
    log_info("null_backend", "initialized", {
        {"vocab_size", config_.vocab_size},
        {"max_seq_len", config_.max_seq_len}
    });
    return Result<void>::success();
}

Result<void> NullBackend::load_model() {
    // Simulate model loading time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    loaded_ = true;
    log_info("null_backend", "model_loaded", {});
    return Result<void>::success();
}

void NullBackend::unload_model() {
    loaded_ = false;
    log_info("null_backend", "model_unloaded", {});
}

bool NullBackend::is_loaded() const {
    return loaded_;
}

const ModelConfig& NullBackend::config() const {
    return config_;
}

Result<PrefillOutput> NullBackend::prefill(
    const PrefillInput& input,
    CancellationTokenPtr cancel) {

    if (!loaded_) {
        return Error::internal("Model not loaded");
    }

    if (cancel && cancel->is_cancelled()) {
        return Error::internal("Request cancelled");
    }

    // Simulate prefill time based on sequence length
    // ~1ms per 100 tokens
    int delay_ms = std::max(1, input.seq_len / 100);
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));

    PrefillOutput output;
    output.logits = generate_fake_logits();
    output.processed_tokens = input.seq_len;

    return output;
}

Result<DecodeOutput> NullBackend::decode_step(
    const DecodeInput& input,
    CancellationTokenPtr cancel) {

    if (!loaded_) {
        return Error::internal("Model not loaded");
    }

    if (cancel && cancel->is_cancelled()) {
        return Error::internal("Request cancelled");
    }

    // Simulate decode time (~10ms per token)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    DecodeOutput output;

    // If we have a predetermined sequence, bias towards it
    int32_t preferred = -1;
    if (!output_sequence_.empty() && output_index_ < output_sequence_.size()) {
        preferred = output_sequence_[output_index_++];
    }

    output.logits = generate_fake_logits(preferred);

    return output;
}

Result<BatchDecodeOutput> NullBackend::decode_batch(
    const BatchDecodeInput& input,
    CancellationTokenPtr cancel) {

    BatchDecodeOutput output;

    for (const auto& req : input.requests) {
        auto result = decode_step(req, cancel);
        if (result.is_error()) {
            return result.error();
        }
        output.outputs.push_back(std::move(result.value()));
    }

    return output;
}

int32_t NullBackend::vocab_size() const {
    return config_.vocab_size;
}

int32_t NullBackend::eos_token_id() const {
    return 151645;  // Qwen EOS
}

int32_t NullBackend::pad_token_id() const {
    return 151643;  // Qwen PAD
}

void NullBackend::synchronize() {
    // No-op for null backend
}

void NullBackend::set_seed(uint64_t seed) {
    rng_.seed(static_cast<uint32_t>(seed));
    output_index_ = 0;
}

void NullBackend::set_output_sequence(const std::vector<int32_t>& tokens) {
    output_sequence_ = tokens;
    output_index_ = 0;
}

Logits NullBackend::generate_fake_logits(int32_t preferred_token) {
    Logits logits;
    logits.vocab_size = config_.vocab_size;
    logits.data.resize(config_.vocab_size, -10.0f);  // Low base logit

    if (preferred_token >= 0 && preferred_token < config_.vocab_size) {
        // Strong preference for the specified token
        logits.data[preferred_token] = 10.0f;
    } else {
        // Generate some random high logits for common tokens
        std::uniform_int_distribution<int32_t> dist(0, 1000);  // Common token range
        for (int i = 0; i < 10; ++i) {
            int32_t token = dist(rng_);
            logits.data[token] = static_cast<float>(5.0 - i * 0.5);
        }

        // Sometimes generate EOS
        if (std::uniform_real_distribution<float>(0, 1)(rng_) < 0.01f) {
            logits.data[eos_token_id()] = 10.0f;
        }
    }

    return logits;
}

}  // namespace qwen::backend
