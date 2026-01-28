#include "kvcache/kv_allocator.hpp"
#include "utils/logger.hpp"

#include <cuda_runtime.h>
#include <algorithm>

namespace qwen::kvcache {

/// Contiguous KV cache allocator with tensor-parallel support
/// Allocates a single contiguous block per request per device
class ContiguousKVAllocator : public IKVCacheAllocator {
public:
    ContiguousKVAllocator() = default;

    ~ContiguousKVAllocator() override {
        reset();
    }

    Result<void> initialize(const KVAllocatorConfig& config) override {
        config_ = config;

        // Determine devices to manage
        if (config_.device_ids.empty()) {
            device_ids_ = {config_.device_id};
        } else {
            device_ids_ = config_.device_ids;
        }
        config_.tensor_parallel_size = static_cast<int32_t>(device_ids_.size());

        // Initialize per-device stats
        for (int32_t device_id : device_ids_) {
            size_t device_memory = config_.max_total_memory;

            if (device_memory == 0) {
                cudaDeviceProp props;
                cudaError_t err = cudaGetDeviceProperties(&props, device_id);
                if (err != cudaSuccess) {
                    return Error::internal("Failed to query GPU " + std::to_string(device_id) +
                        " properties: " + std::string(cudaGetErrorString(err)));
                }
                // Use 80% of total memory as budget
                device_memory = static_cast<size_t>(props.totalGlobalMem * 0.8);
            }

            MemoryStats stats;
            stats.total_bytes = device_memory;
            stats.free_bytes = device_memory;
            device_stats_[device_id] = stats;
        }

        log_info("kv_allocator", "initialized", {
            {"num_devices", device_ids_.size()},
            {"tensor_parallel_size", config_.tensor_parallel_size},
            {"bytes_per_token", config_.bytes_per_token()},
            {"bytes_per_token_per_device", config_.bytes_per_token_per_device()}
        });

        return Result<void>::success();
    }

    Result<KVCacheHandle> allocate(size_t max_seq_len) override {
        // Single-GPU allocation on primary device
        return allocate_on_device(max_seq_len, config_.device_id);
    }

    Result<KVCacheHandle> allocate_tensor_parallel(
            size_t max_seq_len, const std::vector<int32_t>& device_ids) override {
        if (device_ids.empty()) {
            return allocate(max_seq_len);
        }

        // Calculate per-device bytes (KV heads split across devices)
        size_t bytes_per_device = config_.bytes_for_seq_len_per_device(max_seq_len);

        std::lock_guard<std::mutex> lock(mutex_);

        // Check all devices have enough memory
        for (int32_t device_id : device_ids) {
            auto it = device_stats_.find(device_id);
            if (it == device_stats_.end()) {
                return Error::internal("Device " + std::to_string(device_id) + " not initialized");
            }
            if (bytes_per_device > it->second.free_bytes) {
                return Error::resource_exhausted(
                    "Insufficient KV cache memory on device " + std::to_string(device_id) +
                    ": need " + std::to_string(bytes_per_device) +
                    " bytes, have " + std::to_string(it->second.free_bytes));
            }
        }

        // Allocate on each device
        KVCacheHandle handle;
        handle.max_seq_len = max_seq_len;
        handle.current_len = 0;
        handle.device_id = device_ids[0];
        handle.allocation_id = next_allocation_id_++;
        handle.is_tensor_parallel = true;

        for (int32_t device_id : device_ids) {
            cudaError_t err = cudaSetDevice(device_id);
            if (err != cudaSuccess) {
                // Cleanup already allocated
                for (auto& cache : handle.device_caches) {
                    cudaSetDevice(cache.device_id);
                    cudaFree(cache.k_cache);
                    cudaFree(cache.v_cache);
                }
                return Error::internal("Failed to set device " + std::to_string(device_id));
            }

            PerDeviceKVCache device_cache;
            device_cache.device_id = device_id;
            device_cache.bytes_allocated = bytes_per_device;

            // Allocate K cache on this device
            err = cudaMalloc(&device_cache.k_cache, bytes_per_device / 2);
            if (err != cudaSuccess) {
                // Cleanup
                for (auto& cache : handle.device_caches) {
                    cudaSetDevice(cache.device_id);
                    cudaFree(cache.k_cache);
                    cudaFree(cache.v_cache);
                }
                return Error::internal("Failed to allocate K cache on device " +
                    std::to_string(device_id) + ": " + cudaGetErrorString(err));
            }

            // Allocate V cache on this device
            err = cudaMalloc(&device_cache.v_cache, bytes_per_device / 2);
            if (err != cudaSuccess) {
                cudaFree(device_cache.k_cache);
                for (auto& cache : handle.device_caches) {
                    cudaSetDevice(cache.device_id);
                    cudaFree(cache.k_cache);
                    cudaFree(cache.v_cache);
                }
                return Error::internal("Failed to allocate V cache on device " +
                    std::to_string(device_id) + ": " + cudaGetErrorString(err));
            }

            handle.device_caches.push_back(device_cache);

            // Update device stats
            auto& stats = device_stats_[device_id];
            stats.used_bytes += bytes_per_device;
            stats.free_bytes -= bytes_per_device;
            stats.active_allocations++;
            stats.total_allocations++;
            if (stats.used_bytes > stats.peak_bytes) {
                stats.peak_bytes = stats.used_bytes;
            }
        }

        // Set primary cache pointers for compatibility
        handle.k_cache = handle.device_caches[0].k_cache;
        handle.v_cache = handle.device_caches[0].v_cache;

        // Track allocation
        allocations_[handle.allocation_id] = bytes_per_device * device_ids.size();

        log_debug("kv_allocator", "allocated_tp", {
            {"allocation_id", handle.allocation_id},
            {"max_seq_len", max_seq_len},
            {"bytes_per_device", bytes_per_device},
            {"num_devices", device_ids.size()}
        });

        return handle;
    }

    void free(KVCacheHandle& handle) override {
        if (!handle.is_valid()) return;

        std::lock_guard<std::mutex> lock(mutex_);

        if (handle.is_tensor_parallel) {
            // Free on each device
            for (auto& cache : handle.device_caches) {
                cudaSetDevice(cache.device_id);
                cudaFree(cache.k_cache);
                cudaFree(cache.v_cache);

                // Update device stats
                auto it = device_stats_.find(cache.device_id);
                if (it != device_stats_.end()) {
                    it->second.used_bytes -= cache.bytes_allocated;
                    it->second.free_bytes += cache.bytes_allocated;
                    it->second.active_allocations--;
                    it->second.total_frees++;
                }
            }
        } else {
            // Single device free
            cudaSetDevice(handle.device_id);
            cudaFree(handle.k_cache);
            cudaFree(handle.v_cache);

            auto it = device_stats_.find(handle.device_id);
            if (it != device_stats_.end()) {
                auto alloc_it = allocations_.find(handle.allocation_id);
                if (alloc_it != allocations_.end()) {
                    it->second.used_bytes -= alloc_it->second;
                    it->second.free_bytes += alloc_it->second;
                }
                it->second.active_allocations--;
                it->second.total_frees++;
            }
        }

        allocations_.erase(handle.allocation_id);

        log_debug("kv_allocator", "freed", {
            {"allocation_id", handle.allocation_id},
            {"is_tensor_parallel", handle.is_tensor_parallel}
        });

        // Clear handle
        handle.k_cache = nullptr;
        handle.v_cache = nullptr;
        handle.device_caches.clear();
        handle.allocation_id = 0;
    }

    bool can_allocate(size_t max_seq_len) const override {
        size_t bytes_needed = config_.bytes_for_seq_len(max_seq_len);
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = device_stats_.find(config_.device_id);
        if (it == device_stats_.end()) return false;
        return bytes_needed <= it->second.free_bytes;
    }

    bool can_allocate_tensor_parallel(
            size_t max_seq_len, const std::vector<int32_t>& device_ids) const override {
        if (device_ids.empty()) {
            return can_allocate(max_seq_len);
        }

        size_t bytes_per_device = config_.bytes_for_seq_len_per_device(max_seq_len);
        std::lock_guard<std::mutex> lock(mutex_);

        for (int32_t device_id : device_ids) {
            auto it = device_stats_.find(device_id);
            if (it == device_stats_.end()) return false;
            if (bytes_per_device > it->second.free_bytes) return false;
        }
        return true;
    }

    MemoryStats stats() const override {
        std::lock_guard<std::mutex> lock(mutex_);

        // Aggregate stats across all devices
        MemoryStats total;
        for (const auto& [device_id, device_stats] : device_stats_) {
            total.total_bytes += device_stats.total_bytes;
            total.used_bytes += device_stats.used_bytes;
            total.free_bytes += device_stats.free_bytes;
            total.peak_bytes += device_stats.peak_bytes;
            total.active_allocations += device_stats.active_allocations;
            total.total_allocations += device_stats.total_allocations;
            total.total_frees += device_stats.total_frees;
        }
        return total;
    }

    MemoryStats stats(int32_t device_id) const override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = device_stats_.find(device_id);
        if (it != device_stats_.end()) {
            return it->second;
        }
        return MemoryStats{};
    }

    const KVAllocatorConfig& config() const override {
        return config_;
    }

    void reset() override {
        std::lock_guard<std::mutex> lock(mutex_);

        // Reset stats for all devices
        for (auto& [device_id, stats] : device_stats_) {
            stats.used_bytes = 0;
            stats.free_bytes = stats.total_bytes;
            stats.active_allocations = 0;
        }
        allocations_.clear();

        log_info("kv_allocator", "reset", {});
    }

private:
    Result<KVCacheHandle> allocate_on_device(size_t max_seq_len, int32_t device_id) {
        size_t bytes_needed = config_.bytes_for_seq_len(max_seq_len);

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = device_stats_.find(device_id);
        if (it == device_stats_.end()) {
            return Error::internal("Device " + std::to_string(device_id) + " not initialized");
        }

        // Check if we have enough memory
        if (bytes_needed > it->second.free_bytes) {
            return Error::resource_exhausted(
                "Insufficient KV cache memory: need " + std::to_string(bytes_needed) +
                " bytes, have " + std::to_string(it->second.free_bytes));
        }

        // Set device
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            return Error::internal("Failed to set device: " + std::string(cudaGetErrorString(err)));
        }

        // Allocate K cache
        void* k_cache = nullptr;
        err = cudaMalloc(&k_cache, bytes_needed / 2);
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
        it->second.used_bytes += bytes_needed;
        it->second.free_bytes -= bytes_needed;
        it->second.active_allocations++;
        it->second.total_allocations++;
        if (it->second.used_bytes > it->second.peak_bytes) {
            it->second.peak_bytes = it->second.used_bytes;
        }

        // Create handle
        KVCacheHandle handle{
            .k_cache = k_cache,
            .v_cache = v_cache,
            .max_seq_len = max_seq_len,
            .current_len = 0,
            .device_id = device_id,
            .allocation_id = next_allocation_id_++
        };

        // Track allocation
        allocations_[handle.allocation_id] = bytes_needed;

        log_debug("kv_allocator", "allocated", {
            {"allocation_id", handle.allocation_id},
            {"device_id", device_id},
            {"max_seq_len", max_seq_len},
            {"bytes", bytes_needed},
            {"free_bytes", it->second.free_bytes}
        });

        return handle;
    }

    KVAllocatorConfig config_;
    std::vector<int32_t> device_ids_;
    std::unordered_map<int32_t, MemoryStats> device_stats_;
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, size_t> allocations_;  // allocation_id -> bytes
    uint64_t next_allocation_id_ = 1;
};

std::unique_ptr<IKVCacheAllocator> create_contiguous_allocator() {
    return std::make_unique<ContiguousKVAllocator>();
}

}  // namespace qwen::kvcache
