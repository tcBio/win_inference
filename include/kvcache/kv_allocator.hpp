#pragma once

#include "utils/result.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace qwen::kvcache {

/// Per-device KV cache allocation for tensor-parallel
struct PerDeviceKVCache {
    void* k_cache = nullptr;        // Key cache memory on this device
    void* v_cache = nullptr;        // Value cache memory on this device
    int32_t device_id = 0;          // GPU device ID
    size_t bytes_allocated = 0;     // Bytes on this device
};

/// KV cache handle representing allocated memory (supports tensor-parallel)
struct KVCacheHandle {
    void* k_cache = nullptr;        // Key cache memory (single-GPU, or first device)
    void* v_cache = nullptr;        // Value cache memory (single-GPU, or first device)
    size_t max_seq_len = 0;         // Maximum sequence length
    size_t current_len = 0;         // Current used length
    int32_t device_id = 0;          // Primary GPU device ID
    uint64_t allocation_id = 0;     // Unique ID for tracking

    // Tensor-parallel support: per-device allocations
    std::vector<PerDeviceKVCache> device_caches;  // Empty for single-GPU
    bool is_tensor_parallel = false;

    [[nodiscard]] bool is_valid() const {
        if (is_tensor_parallel) {
            return !device_caches.empty() &&
                   device_caches[0].k_cache != nullptr;
        }
        return k_cache != nullptr && v_cache != nullptr;
    }
    [[nodiscard]] size_t remaining() const { return max_seq_len - current_len; }
    [[nodiscard]] size_t num_devices() const {
        return is_tensor_parallel ? device_caches.size() : 1;
    }
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

    // Tensor-parallel configuration
    std::vector<int32_t> device_ids = {};  // Empty = single GPU (device_id)
    int32_t tensor_parallel_size = 1;      // 1 = no TP, 2+ = split across GPUs

    /// Calculate bytes per token for KV cache (full, not split)
    [[nodiscard]] size_t bytes_per_token() const {
        // K and V caches: 2 * layers * kv_heads * head_dim * dtype
        return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes;
    }

    /// Calculate bytes per token per device (for tensor-parallel)
    [[nodiscard]] size_t bytes_per_token_per_device() const {
        // KV heads are split across devices
        int32_t kv_heads_per_device = num_kv_heads / tensor_parallel_size;
        return 2 * num_layers * kv_heads_per_device * head_dim * dtype_bytes;
    }

    /// Calculate total bytes for a sequence length
    [[nodiscard]] size_t bytes_for_seq_len(size_t seq_len) const {
        return seq_len * bytes_per_token();
    }

    /// Calculate bytes per device for a sequence length (tensor-parallel)
    [[nodiscard]] size_t bytes_for_seq_len_per_device(size_t seq_len) const {
        return seq_len * bytes_per_token_per_device();
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

    /// Allocate KV cache for a request (single-GPU)
    /// @param max_seq_len Maximum sequence length this cache will hold
    /// @return Handle to allocated cache, or error if OOM
    [[nodiscard]] virtual Result<KVCacheHandle> allocate(size_t max_seq_len) = 0;

    /// Allocate KV cache for tensor-parallel (multi-GPU)
    /// @param max_seq_len Maximum sequence length this cache will hold
    /// @param device_ids Device IDs to allocate across
    /// @return Handle with per-device allocations, or error if OOM
    [[nodiscard]] virtual Result<KVCacheHandle> allocate_tensor_parallel(
        size_t max_seq_len, const std::vector<int32_t>& device_ids) = 0;

    /// Free a previously allocated cache
    virtual void free(KVCacheHandle& handle) = 0;

    /// Check if allocation of given size would succeed (single-GPU)
    [[nodiscard]] virtual bool can_allocate(size_t max_seq_len) const = 0;

    /// Check if tensor-parallel allocation would succeed
    [[nodiscard]] virtual bool can_allocate_tensor_parallel(
        size_t max_seq_len, const std::vector<int32_t>& device_ids) const = 0;

    /// Get memory statistics
    [[nodiscard]] virtual MemoryStats stats() const = 0;

    /// Get memory statistics for a specific device
    [[nodiscard]] virtual MemoryStats stats(int32_t device_id) const = 0;

    /// Get configuration
    [[nodiscard]] virtual const KVAllocatorConfig& config() const = 0;

    /// Reset allocator state (for testing)
    virtual void reset() = 0;
};

/// Create a contiguous KV cache allocator (MVP - per-request cudaMalloc)
std::unique_ptr<IKVCacheAllocator> create_contiguous_allocator();

/// Pooled allocator configuration
struct PooledAllocatorConfig {
    size_t slot_size_tokens = 16384;    // Max tokens per slot (16k default)
    size_t num_slots = 8;                // Number of pre-allocated slots
    bool allow_overflow = true;          // Allow cudaMalloc if pool exhausted
};

/// Create a pooled KV cache allocator (optimized - pre-allocated memory pool)
/// Reduces allocation latency by avoiding per-request cudaMalloc
std::unique_ptr<IKVCacheAllocator> create_pooled_allocator(
    const PooledAllocatorConfig& pool_config = {});

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
