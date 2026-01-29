#pragma once

#include "gpu/device_manager.hpp"
#include "scheduler/request.hpp"
#include "utils/logger.hpp"

#include <memory>
#include <vector>
#include <mutex>
#include <atomic>

namespace qwen::gpu {

/// GPU routing decision for a request
struct RoutingDecision {
    GPUStrategy strategy;
    std::vector<int32_t> device_ids;
    bool requires_nccl;  // Needs NCCL for collective ops
};

/// Per-GPU memory budget tracking
struct GPUBudget {
    int32_t device_id = 0;
    size_t total_memory = 0;        // Total available for KV cache
    std::atomic<size_t> used_memory{0};  // Currently allocated
    std::atomic<size_t> reserved_memory{0};  // Reserved for pending requests

    // Default constructor
    GPUBudget() = default;

    // Move constructor (required for std::vector with non-copyable atomics)
    GPUBudget(GPUBudget&& other) noexcept
        : device_id(other.device_id)
        , total_memory(other.total_memory)
        , used_memory(other.used_memory.load(std::memory_order_relaxed))
        , reserved_memory(other.reserved_memory.load(std::memory_order_relaxed)) {}

    // Move assignment
    GPUBudget& operator=(GPUBudget&& other) noexcept {
        if (this != &other) {
            device_id = other.device_id;
            total_memory = other.total_memory;
            used_memory.store(other.used_memory.load(std::memory_order_relaxed), std::memory_order_relaxed);
            reserved_memory.store(other.reserved_memory.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    // Delete copy operations (atomics are non-copyable)
    GPUBudget(const GPUBudget&) = delete;
    GPUBudget& operator=(const GPUBudget&) = delete;

    [[nodiscard]] size_t available() const {
        size_t used = used_memory.load(std::memory_order_acquire);
        size_t reserved = reserved_memory.load(std::memory_order_acquire);
        if (used + reserved >= total_memory) return 0;
        return total_memory - used - reserved;
    }

    [[nodiscard]] bool can_fit(size_t bytes) const {
        return available() >= bytes;
    }
};

/// GPU router for request placement decisions
/// Determines which GPU(s) should handle each request based on memory availability
class GPURouter {
public:
    explicit GPURouter(std::shared_ptr<DeviceManager> device_manager);

    /// Initialize per-GPU budgets
    /// @param kv_memory_per_gpu Memory available for KV cache per GPU (bytes)
    void initialize_budgets(size_t kv_memory_per_gpu);

    /// Route a request to appropriate GPU(s)
    /// Returns routing decision with device assignments
    [[nodiscard]] RoutingDecision route(const scheduler::RequestPtr& request) const;

    /// Check if a request can be placed given current memory
    /// @param request The request to check
    /// @param total_bytes Total estimated KV cache size in bytes
    /// @param per_device_bytes Per-device bytes for TP (0 = auto-calculate as total/devices)
    [[nodiscard]] bool can_place(const scheduler::RequestPtr& request,
                                  size_t total_bytes,
                                  size_t per_device_bytes = 0) const;

    /// Reserve memory for a request (call before allocation)
    /// @param decision The routing decision
    /// @param bytes Memory to reserve per device
    void reserve(const RoutingDecision& decision, size_t bytes);

    /// Commit reserved memory to used (call after successful allocation)
    /// @param decision The routing decision
    /// @param bytes Memory that was reserved
    void commit(const RoutingDecision& decision, size_t bytes);

    /// Release reserved memory (call if allocation fails)
    /// @param decision The routing decision
    /// @param bytes Memory to unreserve
    void unreserve(const RoutingDecision& decision, size_t bytes);

    /// Free used memory (call when request completes)
    /// @param decision The routing decision (or device IDs from request)
    /// @param bytes Memory to free
    void free(const std::vector<int32_t>& device_ids, size_t bytes);

    /// Get budget for a specific device
    [[nodiscard]] const GPUBudget* get_budget(int32_t device_id) const;

    /// Get device manager
    [[nodiscard]] const std::shared_ptr<DeviceManager>& device_manager() const {
        return device_manager_;
    }

private:
    std::shared_ptr<DeviceManager> device_manager_;
    std::vector<GPUBudget> budgets_;
    mutable std::mutex budget_mutex_;
};

}  // namespace qwen::gpu
