#pragma once

#include "backend/runtime_interface.hpp"

#include <memory>
#include <string>

namespace qwen::backend {

/// TensorRT backend configuration
struct TensorRTConfig {
    std::string engine_path;            // Path to serialized TensorRT engine
    std::string tokenizer_path;         // Path to tokenizer files
    int32_t max_batch_size = 1;
    int32_t max_input_len = 70000;
    int32_t max_output_len = 8192;
    bool use_fp16 = true;
    bool use_int8 = false;
    std::vector<int32_t> device_ids = {0, 1};
    bool tensor_parallel = true;
    size_t workspace_size = 2ULL * 1024 * 1024 * 1024;  // 2 GB
};

/// TensorRT-LLM backend for Qwen2.5-32B
/// This is a stub implementation - full integration requires TensorRT-LLM SDK
class TensorRTBackend : public IModelRuntime {
public:
    TensorRTBackend();
    ~TensorRTBackend() override;

    /// Initialize with TensorRT-specific config
    Result<void> initialize(const TensorRTConfig& trt_config);

    // IModelRuntime interface
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

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    ModelConfig config_;
    TensorRTConfig trt_config_;
    bool loaded_ = false;
};

/// Build a TensorRT engine from ONNX model
/// This would be called offline, not at server startup
Result<void> build_engine(
    const std::string& onnx_path,
    const std::string& engine_path,
    const TensorRTConfig& config);

}  // namespace qwen::backend
