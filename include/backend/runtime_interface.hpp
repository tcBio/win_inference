#pragma once

#include "utils/result.hpp"
#include "utils/cancellation.hpp"
#include "kvcache/kv_allocator.hpp"

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <functional>

namespace qwen::backend {

/// Model configuration
struct ModelConfig {
    std::string model_path;
    int32_t max_batch_size = 1;
    int32_t max_seq_len = 70000;
    int32_t vocab_size = 152064;        // Qwen2.5 vocab size
    int32_t num_layers = 64;
    int32_t hidden_size = 5120;
    int32_t num_heads = 40;
    int32_t num_kv_heads = 8;           // GQA
    int32_t head_dim = 128;
    bool use_fp16 = true;
    std::vector<int32_t> device_ids = {0, 1};
    bool tensor_parallel = true;
};

/// Token IDs type
using TokenIds = std::vector<int32_t>;

/// Logits tensor (simplified - vocab_size floats)
struct Logits {
    std::vector<float> data;
    int32_t vocab_size = 0;

    [[nodiscard]] float operator[](size_t idx) const { return data[idx]; }
    [[nodiscard]] bool empty() const { return data.empty(); }
};

/// Prefill request input
struct PrefillInput {
    TokenIds input_ids;
    kvcache::KVCacheHandle* kv_cache = nullptr;
    int32_t seq_len = 0;
};

/// Prefill result
struct PrefillOutput {
    Logits logits;  // Logits for last position only
    int32_t processed_tokens = 0;
};

/// Decode step input
struct DecodeInput {
    int32_t input_token;                // Single token for autoregressive
    kvcache::KVCacheHandle* kv_cache = nullptr;
    int32_t current_pos = 0;            // Current position in sequence
};

/// Decode step result
struct DecodeOutput {
    Logits logits;  // Logits for the new position
};

/// Batch decode input for continuous batching (V2)
struct BatchDecodeInput {
    std::vector<DecodeInput> requests;
};

struct BatchDecodeOutput {
    std::vector<DecodeOutput> outputs;
};

/// Abstract runtime interface for model execution
/// Implementations: NullBackend (testing), TensorRTBackend (production)
class IModelRuntime {
public:
    virtual ~IModelRuntime() = default;

    /// Initialize the runtime with model configuration
    [[nodiscard]] virtual Result<void> initialize(const ModelConfig& config) = 0;

    /// Load model weights
    [[nodiscard]] virtual Result<void> load_model() = 0;

    /// Unload model and free resources
    virtual void unload_model() = 0;

    /// Check if model is loaded
    [[nodiscard]] virtual bool is_loaded() const = 0;

    /// Get model configuration
    [[nodiscard]] virtual const ModelConfig& config() const = 0;

    /// Run prefill for a sequence (initial prompt processing)
    [[nodiscard]] virtual Result<PrefillOutput> prefill(
        const PrefillInput& input,
        CancellationTokenPtr cancel = nullptr) = 0;

    /// Run a single decode step
    [[nodiscard]] virtual Result<DecodeOutput> decode_step(
        const DecodeInput& input,
        CancellationTokenPtr cancel = nullptr) = 0;

    /// Run batched decode (for continuous batching, V2)
    [[nodiscard]] virtual Result<BatchDecodeOutput> decode_batch(
        const BatchDecodeInput& input,
        CancellationTokenPtr cancel = nullptr) = 0;

    /// Get vocabulary size
    [[nodiscard]] virtual int32_t vocab_size() const = 0;

    /// Get EOS token ID
    [[nodiscard]] virtual int32_t eos_token_id() const = 0;

    /// Get pad token ID
    [[nodiscard]] virtual int32_t pad_token_id() const = 0;

    /// Synchronize all pending operations
    virtual void synchronize() = 0;
};

/// Factory function type
using RuntimeFactory = std::function<std::unique_ptr<IModelRuntime>()>;

/// Register a runtime factory
void register_runtime(const std::string& name, RuntimeFactory factory);

/// Create a runtime by name
std::unique_ptr<IModelRuntime> create_runtime(const std::string& name);

}  // namespace qwen::backend
