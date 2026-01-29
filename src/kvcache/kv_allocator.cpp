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

/// Pooled KV cache allocator - pre-allocates memory to avoid per-request cudaMalloc
/// Uses fixed-size slots for O(1) allocation/deallocation
class PooledKVAllocator : public IKVCacheAllocator {
public:
    explicit PooledKVAllocator(PooledAllocatorConfig pool_config)
        : pool_config_(std::move(pool_config)) {}

    ~PooledKVAllocator() override {
        cleanup_pools();
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

        // Calculate slot size in bytes
        slot_bytes_ = config_.bytes_for_seq_len(pool_config_.slot_size_tokens);

        // Pre-allocate pool on each device
        for (int32_t device_id : device_ids_) {
            auto result = allocate_pool_for_device(device_id);
            if (result.is_error()) {
                cleanup_pools();
                return result;
            }
        }

        log_info("kv_allocator_pooled", "initialized", {
            {"num_devices", device_ids_.size()},
            {"slots_per_device", pool_config_.num_slots},
            {"slot_size_tokens", pool_config_.slot_size_tokens},
            {"slot_bytes", slot_bytes_},
            {"total_pool_bytes", slot_bytes_ * pool_config_.num_slots * device_ids_.size()}
        });

        return Result<void>::success();
    }

    Result<KVCacheHandle> allocate(size_t max_seq_len) override {
        size_t bytes_needed = config_.bytes_for_seq_len(max_seq_len);

        std::lock_guard<std::mutex> lock(mutex_);

        // Check if request fits in a slot
        if (max_seq_len <= pool_config_.slot_size_tokens) {
            // Try to get a slot from the pool
            auto& pool = device_pools_[config_.device_id];
            if (!pool.free_slots.empty()) {
                size_t slot_idx = pool.free_slots.back();
                pool.free_slots.pop_back();

                auto& slot = pool.slots[slot_idx];
                slot.in_use = true;

                KVCacheHandle handle{
                    .k_cache = slot.k_cache,
                    .v_cache = slot.v_cache,
                    .max_seq_len = max_seq_len,
                    .current_len = 0,
                    .device_id = config_.device_id,
                    .allocation_id = next_allocation_id_++
                };

                // Track which slot this allocation uses
                allocation_slots_[handle.allocation_id] = {config_.device_id, slot_idx, false};

                pool.stats.used_bytes += slot_bytes_;
                pool.stats.free_bytes -= slot_bytes_;
                pool.stats.active_allocations++;
                pool.stats.total_allocations++;
                if (pool.stats.used_bytes > pool.stats.peak_bytes) {
                    pool.stats.peak_bytes = pool.stats.used_bytes;
                }

                log_debug("kv_allocator_pooled", "allocated_from_pool", {
                    {"allocation_id", handle.allocation_id},
                    {"slot_idx", slot_idx},
                    {"device_id", config_.device_id}
                });

                return handle;
            }
        }

        // Pool exhausted or request too large - fallback to cudaMalloc
        if (!pool_config_.allow_overflow) {
            return Error::resource_exhausted("KV cache pool exhausted");
        }

        log_debug("kv_allocator_pooled", "fallback_to_malloc", {
            {"max_seq_len", max_seq_len},
            {"bytes_needed", bytes_needed}
        });

        return allocate_fallback(max_seq_len, config_.device_id);
    }

    Result<KVCacheHandle> allocate_tensor_parallel(
            size_t max_seq_len, const std::vector<int32_t>& device_ids) override {
        // For TP, use fallback allocation (pooling TP is more complex)
        // This is a simplification - production would have per-device pools
        if (device_ids.empty()) {
            return allocate(max_seq_len);
        }

        // Fallback to contiguous allocation for TP
        return allocate_tensor_parallel_fallback(max_seq_len, device_ids);
    }

    void free(KVCacheHandle& handle) override {
        if (!handle.is_valid()) return;

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocation_slots_.find(handle.allocation_id);
        if (it != allocation_slots_.end()) {
            auto& [device_id, slot_idx, is_overflow] = it->second;

            if (is_overflow) {
                // Free overflow allocation
                cudaSetDevice(device_id);
                cudaFree(handle.k_cache);
                cudaFree(handle.v_cache);
            } else {
                // Return slot to pool
                auto& pool = device_pools_[device_id];
                pool.slots[slot_idx].in_use = false;
                pool.free_slots.push_back(slot_idx);

                pool.stats.used_bytes -= slot_bytes_;
                pool.stats.free_bytes += slot_bytes_;
                pool.stats.active_allocations--;
                pool.stats.total_frees++;
            }

            allocation_slots_.erase(it);
        }

        handle.k_cache = nullptr;
        handle.v_cache = nullptr;
        handle.allocation_id = 0;
    }

    bool can_allocate(size_t max_seq_len) const override {
        std::lock_guard<std::mutex> lock(mutex_);

        // Check if fits in pool slot
        if (max_seq_len <= pool_config_.slot_size_tokens) {
            auto it = device_pools_.find(config_.device_id);
            if (it != device_pools_.end() && !it->second.free_slots.empty()) {
                return true;
            }
        }

        // Check overflow allowance
        return pool_config_.allow_overflow;
    }

    bool can_allocate_tensor_parallel(
            size_t /*max_seq_len*/, const std::vector<int32_t>& /*device_ids*/) const override {
        // Simplified - allow if overflow is permitted
        return pool_config_.allow_overflow;
    }

    MemoryStats stats() const override {
        std::lock_guard<std::mutex> lock(mutex_);

        MemoryStats total;
        for (const auto& [device_id, pool] : device_pools_) {
            total.total_bytes += pool.stats.total_bytes;
            total.used_bytes += pool.stats.used_bytes;
            total.free_bytes += pool.stats.free_bytes;
            total.peak_bytes += pool.stats.peak_bytes;
            total.active_allocations += pool.stats.active_allocations;
            total.total_allocations += pool.stats.total_allocations;
            total.total_frees += pool.stats.total_frees;
        }
        return total;
    }

    MemoryStats stats(int32_t device_id) const override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = device_pools_.find(device_id);
        if (it != device_pools_.end()) {
            return it->second.stats;
        }
        return MemoryStats{};
    }

    const KVAllocatorConfig& config() const override {
        return config_;
    }

    void reset() override {
        std::lock_guard<std::mutex> lock(mutex_);

        // Return all slots to free list
        for (auto& [device_id, pool] : device_pools_) {
            pool.free_slots.clear();
            for (size_t i = 0; i < pool.slots.size(); ++i) {
                pool.slots[i].in_use = false;
                pool.free_slots.push_back(i);
            }
            pool.stats.used_bytes = 0;
            pool.stats.free_bytes = pool.stats.total_bytes;
            pool.stats.active_allocations = 0;
        }
        allocation_slots_.clear();

        log_info("kv_allocator_pooled", "reset", {});
    }

private:
    struct Slot {
        void* k_cache = nullptr;
        void* v_cache = nullptr;
        bool in_use = false;
    };

    struct DevicePool {
        std::vector<Slot> slots;
        std::vector<size_t> free_slots;  // Stack of free slot indices
        void* pool_memory_k = nullptr;   // Contiguous pool for K
        void* pool_memory_v = nullptr;   // Contiguous pool for V
        MemoryStats stats;
    };

    Result<void> allocate_pool_for_device(int32_t device_id) {
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            return Error::internal("Failed to set device " + std::to_string(device_id));
        }

        size_t pool_bytes = slot_bytes_ * pool_config_.num_slots;
        size_t half_pool = pool_bytes / 2;  // Split between K and V

        DevicePool pool;

        // Allocate contiguous pool for K caches
        err = cudaMalloc(&pool.pool_memory_k, half_pool);
        if (err != cudaSuccess) {
            return Error::internal("Failed to allocate K pool on device " +
                std::to_string(device_id) + ": " + cudaGetErrorString(err));
        }

        // Allocate contiguous pool for V caches
        err = cudaMalloc(&pool.pool_memory_v, half_pool);
        if (err != cudaSuccess) {
            cudaFree(pool.pool_memory_k);
            return Error::internal("Failed to allocate V pool on device " +
                std::to_string(device_id) + ": " + cudaGetErrorString(err));
        }

        // Divide pool into slots
        size_t slot_k_bytes = slot_bytes_ / 2;
        size_t slot_v_bytes = slot_bytes_ / 2;

        pool.slots.resize(pool_config_.num_slots);
        for (size_t i = 0; i < pool_config_.num_slots; ++i) {
            pool.slots[i].k_cache = static_cast<char*>(pool.pool_memory_k) + i * slot_k_bytes;
            pool.slots[i].v_cache = static_cast<char*>(pool.pool_memory_v) + i * slot_v_bytes;
            pool.slots[i].in_use = false;
            pool.free_slots.push_back(i);
        }

        pool.stats.total_bytes = pool_bytes;
        pool.stats.free_bytes = pool_bytes;

        device_pools_[device_id] = std::move(pool);

        log_debug("kv_allocator_pooled", "pool_allocated", {
            {"device_id", device_id},
            {"num_slots", pool_config_.num_slots},
            {"pool_bytes", pool_bytes}
        });

        return Result<void>::success();
    }

    void cleanup_pools() {
        for (auto& [device_id, pool] : device_pools_) {
            cudaSetDevice(device_id);
            if (pool.pool_memory_k) {
                cudaFree(pool.pool_memory_k);
            }
            if (pool.pool_memory_v) {
                cudaFree(pool.pool_memory_v);
            }
        }
        device_pools_.clear();
    }

    Result<KVCacheHandle> allocate_fallback(size_t max_seq_len, int32_t device_id) {
        size_t bytes_needed = config_.bytes_for_seq_len(max_seq_len);

        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            return Error::internal("Failed to set device");
        }

        void* k_cache = nullptr;
        err = cudaMalloc(&k_cache, bytes_needed / 2);
        if (err != cudaSuccess) {
            return Error::internal("Failed to allocate K cache: " +
                std::string(cudaGetErrorString(err)));
        }

        void* v_cache = nullptr;
        err = cudaMalloc(&v_cache, bytes_needed / 2);
        if (err != cudaSuccess) {
            cudaFree(k_cache);
            return Error::internal("Failed to allocate V cache: " +
                std::string(cudaGetErrorString(err)));
        }

        KVCacheHandle handle{
            .k_cache = k_cache,
            .v_cache = v_cache,
            .max_seq_len = max_seq_len,
            .current_len = 0,
            .device_id = device_id,
            .allocation_id = next_allocation_id_++
        };

        allocation_slots_[handle.allocation_id] = {device_id, 0, true};  // is_overflow = true

        return handle;
    }

    Result<KVCacheHandle> allocate_tensor_parallel_fallback(
            size_t max_seq_len, const std::vector<int32_t>& device_ids) {
        size_t bytes_per_device = config_.bytes_for_seq_len_per_device(max_seq_len);

        KVCacheHandle handle;
        handle.max_seq_len = max_seq_len;
        handle.current_len = 0;
        handle.device_id = device_ids[0];
        handle.allocation_id = next_allocation_id_++;
        handle.is_tensor_parallel = true;

        for (int32_t device_id : device_ids) {
            cudaError_t err = cudaSetDevice(device_id);
            if (err != cudaSuccess) {
                // Cleanup
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

            err = cudaMalloc(&device_cache.k_cache, bytes_per_device / 2);
            if (err != cudaSuccess) {
                for (auto& cache : handle.device_caches) {
                    cudaSetDevice(cache.device_id);
                    cudaFree(cache.k_cache);
                    cudaFree(cache.v_cache);
                }
                return Error::internal("Failed to allocate K cache on device " +
                    std::to_string(device_id));
            }

            err = cudaMalloc(&device_cache.v_cache, bytes_per_device / 2);
            if (err != cudaSuccess) {
                cudaFree(device_cache.k_cache);
                for (auto& cache : handle.device_caches) {
                    cudaSetDevice(cache.device_id);
                    cudaFree(cache.k_cache);
                    cudaFree(cache.v_cache);
                }
                return Error::internal("Failed to allocate V cache on device " +
                    std::to_string(device_id));
            }

            handle.device_caches.push_back(device_cache);
        }

        handle.k_cache = handle.device_caches[0].k_cache;
        handle.v_cache = handle.device_caches[0].v_cache;

        allocation_slots_[handle.allocation_id] = {handle.device_id, 0, true};

        return handle;
    }

    PooledAllocatorConfig pool_config_;
    KVAllocatorConfig config_;
    std::vector<int32_t> device_ids_;
    size_t slot_bytes_ = 0;

    mutable std::mutex mutex_;
    std::unordered_map<int32_t, DevicePool> device_pools_;
    std::unordered_map<uint64_t, std::tuple<int32_t, size_t, bool>> allocation_slots_;
    uint64_t next_allocation_id_ = 1;
};

std::unique_ptr<IKVCacheAllocator> create_pooled_allocator(const PooledAllocatorConfig& pool_config) {
    return std::make_unique<PooledKVAllocator>(pool_config);
}

}  // namespace qwen::kvcache
