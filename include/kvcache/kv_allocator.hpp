#pragma once

#include "utils/result.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace qwen::kvcache {

/// KV cache handle representing allocated memory
struct KVCacheHandle {
    void* k_cache = nullptr;        // Key cache memory
    void* v_cache = nullptr;        // Value cache memory
    size_t max_seq_len = 0;         // Maximum sequence length
    size_t current_len = 0;         // Current used length
    int32_t device_id = 0;          // GPU device ID
    uint64_t allocation_id = 0;     // Unique ID for tracking

    [[nodiscard]] bool is_valid() const { return k_cache != nullptr && v_cache != nullptr; }
    [[nodiscard]] size_t remaining() const { return max_seq_len - current_len; }
};

/// Memory statistics
struct MemoryStats {
    size_t total_bytes = 0;         // Total available memory
    size_t used_bytes = 0;          // Currently allocated
    size_t free_bytes = 0;          // Available for allocation
    size_t peak_bytes = 0;          // Peak usage
    uint32_t active_allocations = 0;
    uint32_t total_allocations = 0;
    uint32_t total_frees = 0;
};

/// Configuration for KV cache allocator
struct KVAllocatorConfig {
    int32_t num_layers = 64;
    int32_t num_kv_heads = 8;
    int32_t head_dim = 128;
    size_t dtype_bytes = 2;         // FP16 = 2 bytes
    size_t max_total_memory = 0;    // 0 = auto-detect from GPU
    int32_t device_id = 0;
    bool pre_allocate = false;      // Pre-allocate pool at startup

    /// Calculate bytes per token for KV cache
    [[nodiscard]] size_t bytes_per_token() const {
        // K and V caches: 2 * layers * kv_heads * head_dim * dtype
        return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes;
    }

    /// Calculate total bytes for a sequence length
    [[nodiscard]] size_t bytes_for_seq_len(size_t seq_len) const {
        return seq_len * bytes_per_token();
    }
};

/// Abstract KV cache allocator interface
/// MVP: Contiguous allocation
/// V2: Paged/block allocation
class IKVCacheAllocator {
public:
    virtual ~IKVCacheAllocator() = default;

    /// Initialize the allocator
    [[nodiscard]] virtual Result<void> initialize(const KVAllocatorConfig& config) = 0;

    /// Allocate KV cache for a request
    /// @param max_seq_len Maximum sequence length this cache will hold
    /// @return Handle to allocated cache, or error if OOM
    [[nodiscard]] virtual Result<KVCacheHandle> allocate(size_t max_seq_len) = 0;

    /// Free a previously allocated cache
    virtual void free(KVCacheHandle& handle) = 0;

    /// Check if allocation of given size would succeed
    [[nodiscard]] virtual bool can_allocate(size_t max_seq_len) const = 0;

    /// Get memory statistics
    [[nodiscard]] virtual MemoryStats stats() const = 0;

    /// Get configuration
    [[nodiscard]] virtual const KVAllocatorConfig& config() const = 0;

    /// Reset allocator state (for testing)
    virtual void reset() = 0;
};

/// Create a contiguous KV cache allocator (MVP)
std::unique_ptr<IKVCacheAllocator> create_contiguous_allocator();

/// Memory math helpers for Qwen2.5-32B
namespace qwen32b {
    constexpr int32_t NUM_LAYERS = 64;
    constexpr int32_t NUM_KV_HEADS = 8;
    constexpr int32_t HEAD_DIM = 128;
    constexpr size_t DTYPE_BYTES = 2;  // FP16

    /// KV cache bytes per token: 2 * 64 * 8 * 128 * 2 = 262,144 bytes = 256 KB
    constexpr size_t BYTES_PER_TOKEN = 2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * DTYPE_BYTES;

    /// KV cache for 70k tokens: 70000 * 256KB = 17.5 GB
    constexpr size_t BYTES_FOR_70K = 70000 * BYTES_PER_TOKEN;

    /// L40S available for KV (48GB - 32GB weights - 2GB workspace) = 14 GB per GPU
    constexpr size_t L40S_KV_BUDGET_PER_GPU = 14ULL * 1024 * 1024 * 1024;

    /// With tensor parallel, KV is split across 2 GPUs
    /// Per-GPU KV for 70k: 8.75 GB
    constexpr size_t BYTES_FOR_70K_PER_GPU_TP = BYTES_FOR_70K / 2;

    /// Max concurrent requests at 70k with TP on 2x L40S: 1
    /// Max concurrent at 10k: ~5
    inline size_t max_concurrent_requests(size_t seq_len, size_t available_bytes) {
        size_t bytes_needed = seq_len * BYTES_PER_TOKEN / 2;  // Assuming TP
        return bytes_needed > 0 ? available_bytes / bytes_needed : 0;
    }
}

}  // namespace qwen::kvcache
