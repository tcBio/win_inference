#include "kvcache/kv_allocator.hpp"
#include "utils/logger.hpp"

#include <cuda_runtime.h>

namespace qwen::kvcache {

/// Contiguous KV cache allocator (MVP)
/// Allocates a single contiguous block per request
class ContiguousKVAllocator : public IKVCacheAllocator {
public:
    ContiguousKVAllocator() = default;

    ~ContiguousKVAllocator() override {
        reset();
    }

    Result<void> initialize(const KVAllocatorConfig& config) override {
        config_ = config;

        // Query GPU memory if not specified
        if (config_.max_total_memory == 0) {
            cudaDeviceProp props;
            cudaError_t err = cudaGetDeviceProperties(&props, config_.device_id);
            if (err != cudaSuccess) {
                return Error::internal("Failed to query GPU properties: " +
                    std::string(cudaGetErrorString(err)));
            }

            // Use 80% of total memory as budget (leave room for weights, workspace)
            config_.max_total_memory = static_cast<size_t>(props.totalGlobalMem * 0.8);
        }

        stats_.total_bytes = config_.max_total_memory;
        stats_.free_bytes = config_.max_total_memory;

        log_info("kv_allocator", "initialized", {
            {"device_id", config_.device_id},
            {"total_bytes", stats_.total_bytes},
            {"bytes_per_token", config_.bytes_per_token()}
        });

        return Result<void>::success();
    }

    Result<KVCacheHandle> allocate(size_t max_seq_len) override {
        size_t bytes_needed = config_.bytes_for_seq_len(max_seq_len);

        std::lock_guard<std::mutex> lock(mutex_);

        // Check if we have enough memory
        if (bytes_needed > stats_.free_bytes) {
            return Error::resource_exhausted(
                "Insufficient KV cache memory: need " + std::to_string(bytes_needed) +
                " bytes, have " + std::to_string(stats_.free_bytes));
        }

        // Allocate K cache
        void* k_cache = nullptr;
        cudaError_t err = cudaMalloc(&k_cache, bytes_needed / 2);
        if (err != cudaSuccess) {
            return Error::internal("Failed to allocate K cache: " +
                std::string(cudaGetErrorString(err)));
        }

        // Allocate V cache
        void* v_cache = nullptr;
        err = cudaMalloc(&v_cache, bytes_needed / 2);
        if (err != cudaSuccess) {
            cudaFree(k_cache);
            return Error::internal("Failed to allocate V cache: " +
                std::string(cudaGetErrorString(err)));
        }

        // Update stats
        stats_.used_bytes += bytes_needed;
        stats_.free_bytes -= bytes_needed;
        stats_.active_allocations++;
        stats_.total_allocations++;
        if (stats_.used_bytes > stats_.peak_bytes) {
            stats_.peak_bytes = stats_.used_bytes;
        }

        // Create handle
        KVCacheHandle handle{
            .k_cache = k_cache,
            .v_cache = v_cache,
            .max_seq_len = max_seq_len,
            .current_len = 0,
            .device_id = config_.device_id,
            .allocation_id = next_allocation_id_++
        };

        // Track allocation
        allocations_[handle.allocation_id] = bytes_needed;

        log_debug("kv_allocator", "allocated", {
            {"allocation_id", handle.allocation_id},
            {"max_seq_len", max_seq_len},
            {"bytes", bytes_needed},
            {"free_bytes", stats_.free_bytes}
        });

        return handle;
    }

    void free(KVCacheHandle& handle) override {
        if (!handle.is_valid()) return;

        std::lock_guard<std::mutex> lock(mutex_);

        // Free GPU memory
        cudaFree(handle.k_cache);
        cudaFree(handle.v_cache);

        // Update stats
        auto it = allocations_.find(handle.allocation_id);
        if (it != allocations_.end()) {
            stats_.used_bytes -= it->second;
            stats_.free_bytes += it->second;
            allocations_.erase(it);
        }
        stats_.active_allocations--;
        stats_.total_frees++;

        log_debug("kv_allocator", "freed", {
            {"allocation_id", handle.allocation_id},
            {"free_bytes", stats_.free_bytes}
        });

        // Clear handle
        handle.k_cache = nullptr;
        handle.v_cache = nullptr;
        handle.allocation_id = 0;
    }

    bool can_allocate(size_t max_seq_len) const override {
        size_t bytes_needed = config_.bytes_for_seq_len(max_seq_len);
        std::lock_guard<std::mutex> lock(mutex_);
        return bytes_needed <= stats_.free_bytes;
    }

    MemoryStats stats() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    const KVAllocatorConfig& config() const override {
        return config_;
    }

    void reset() override {
        std::lock_guard<std::mutex> lock(mutex_);

        // Note: In production, we'd track all allocations and free them
        // For now, just reset stats (allocations should be freed via free())

        stats_.used_bytes = 0;
        stats_.free_bytes = stats_.total_bytes;
        stats_.active_allocations = 0;
        allocations_.clear();

        log_info("kv_allocator", "reset", {});
    }

private:
    KVAllocatorConfig config_;
    MemoryStats stats_;
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, size_t> allocations_;  // allocation_id -> bytes
    uint64_t next_allocation_id_ = 1;
};

std::unique_ptr<IKVCacheAllocator> create_contiguous_allocator() {
    return std::make_unique<ContiguousKVAllocator>();
}

}  // namespace qwen::kvcache
