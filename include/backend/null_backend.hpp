#pragma once

#include "backend/runtime_interface.hpp"

#include <random>

namespace qwen::backend {

/// Null backend for testing
/// Returns deterministic fake outputs without actual model execution
class NullBackend : public IModelRuntime {
public:
    NullBackend();
    ~NullBackend() override = default;

    Result<void> initialize(const ModelConfig& config) override;
    Result<void> load_model() override;
    void unload_model() override;
    [[nodiscard]] bool is_loaded() const override;
    [[nodiscard]] const ModelConfig& config() const override;

    Result<PrefillOutput> prefill(
        const PrefillInput& input,
        CancellationTokenPtr cancel = nullptr) override;

    Result<DecodeOutput> decode_step(
        const DecodeInput& input,
        CancellationTokenPtr cancel = nullptr) override;

    Result<BatchDecodeOutput> decode_batch(
        const BatchDecodeInput& input,
        CancellationTokenPtr cancel = nullptr) override;

    [[nodiscard]] int32_t vocab_size() const override;
    [[nodiscard]] int32_t eos_token_id() const override;
    [[nodiscard]] int32_t pad_token_id() const override;
    void synchronize() override;

    /// Set seed for deterministic outputs
    void set_seed(uint64_t seed);

    /// Set a specific output sequence (for testing)
    void set_output_sequence(const std::vector<int32_t>& tokens);

private:
    Logits generate_fake_logits(int32_t preferred_token = -1);

    ModelConfig config_;
    bool loaded_ = false;
    std::mt19937 rng_;
    std::vector<int32_t> output_sequence_;
    size_t output_index_ = 0;
};

}  // namespace qwen::backend
