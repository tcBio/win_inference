#include "kvcache/kv_allocator.hpp"
#include "utils/logger.hpp"

namespace qwen::kvcache {

/// Memory budget calculator for admission control
/// Computes whether a request can be admitted based on current memory usage

class MemoryBudget {
public:
    struct Config {
        size_t total_gpu_memory = 48ULL * 1024 * 1024 * 1024;  // 48 GB L40S
        size_t model_weights_bytes = 32ULL * 1024 * 1024 * 1024;  // ~32B params FP16
        size_t workspace_bytes = 2ULL * 1024 * 1024 * 1024;  // 2 GB workspace
        size_t safety_margin_bytes = 1ULL * 1024 * 1024 * 1024;  // 1 GB buffer
        bool tensor_parallel = true;
        int32_t num_gpus = 2;
    };

    explicit MemoryBudget(Config config = {}) : config_(config) {
        calculate_budget();
    }

    /// Calculate available memory for KV cache
    void calculate_budget() {
        size_t per_gpu_total = config_.total_gpu_memory;

        // With tensor parallel, weights are split
        size_t per_gpu_weights = config_.tensor_parallel ?
            config_.model_weights_bytes / config_.num_gpus :
            config_.model_weights_bytes;

        size_t overhead = per_gpu_weights + config_.workspace_bytes + config_.safety_margin_bytes;

        if (overhead >= per_gpu_total) {
            kv_budget_per_gpu_ = 0;
        } else {
            kv_budget_per_gpu_ = per_gpu_total - overhead;
        }

        // Total KV budget across all GPUs (with TP, KV is also split)
        kv_budget_total_ = kv_budget_per_gpu_ * config_.num_gpus;

        log_info("memory_budget", "calculated", {
            {"per_gpu_total_gb", config_.total_gpu_memory / (1024.0 * 1024.0 * 1024.0)},
            {"per_gpu_weights_gb", per_gpu_weights / (1024.0 * 1024.0 * 1024.0)},
            {"kv_budget_per_gpu_gb", kv_budget_per_gpu_ / (1024.0 * 1024.0 * 1024.0)},
            {"kv_budget_total_gb", kv_budget_total_ / (1024.0 * 1024.0 * 1024.0)}
        });
    }

    /// Get available KV cache memory per GPU
    [[nodiscard]] size_t kv_budget_per_gpu() const { return kv_budget_per_gpu_; }

    /// Get total KV cache memory budget
    [[nodiscard]] size_t kv_budget_total() const { return kv_budget_total_; }

    /// Calculate max concurrent requests at given context length
    [[nodiscard]] size_t max_concurrent_requests(size_t context_length) const {
        size_t bytes_per_request = context_length * qwen32b::BYTES_PER_TOKEN;

        // With tensor parallel, KV is split across GPUs
        if (config_.tensor_parallel) {
            bytes_per_request /= config_.num_gpus;
        }

        if (bytes_per_request == 0) return 0;
        return kv_budget_per_gpu_ / bytes_per_request;
    }

    /// Check if a request of given context length can be admitted
    [[nodiscard]] bool can_admit(size_t context_length, size_t current_used_bytes) const {
        size_t bytes_needed = context_length * qwen32b::BYTES_PER_TOKEN;

        if (config_.tensor_parallel) {
            bytes_needed /= config_.num_gpus;
        }

        return current_used_bytes + bytes_needed <= kv_budget_per_gpu_;
    }

    /// Get configuration
    [[nodiscard]] const Config& config() const { return config_; }

private:
    Config config_;
    size_t kv_budget_per_gpu_ = 0;
    size_t kv_budget_total_ = 0;
};

/// Print memory analysis for planning
void print_memory_analysis() {
    using namespace qwen32b;

    log_info("memory_analysis", "model_params", {
        {"layers", NUM_LAYERS},
        {"kv_heads", NUM_KV_HEADS},
        {"head_dim", HEAD_DIM},
        {"dtype_bytes", DTYPE_BYTES}
    });

    log_info("memory_analysis", "kv_cache_sizing", {
        {"bytes_per_token", BYTES_PER_TOKEN},
        {"kb_per_token", BYTES_PER_TOKEN / 1024.0},
        {"bytes_for_70k", BYTES_FOR_70K},
        {"gb_for_70k", BYTES_FOR_70K / (1024.0 * 1024.0 * 1024.0)},
        {"bytes_for_70k_per_gpu_tp", BYTES_FOR_70K_PER_GPU_TP},
        {"gb_for_70k_per_gpu_tp", BYTES_FOR_70K_PER_GPU_TP / (1024.0 * 1024.0 * 1024.0)}
    });

    MemoryBudget budget;
    log_info("memory_analysis", "concurrency_limits", {
        {"max_at_70k", budget.max_concurrent_requests(70000)},
        {"max_at_50k", budget.max_concurrent_requests(50000)},
        {"max_at_20k", budget.max_concurrent_requests(20000)},
        {"max_at_10k", budget.max_concurrent_requests(10000)}
    });
}

}  // namespace qwen::kvcache
