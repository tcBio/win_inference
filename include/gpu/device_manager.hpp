#pragma once

#include "utils/result.hpp"

#include <vector>
#include <memory>
#include <string>
#include <cstdint>

namespace qwen::gpu {

/// GPU device information
struct DeviceInfo {
    int32_t device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int32_t compute_capability_major;
    int32_t compute_capability_minor;
    bool has_nvlink;
    int32_t nvlink_peer_id;  // -1 if no NVLink peer
};

/// GPU execution strategy
enum class GPUStrategy {
    TensorParallel,     // Split model across GPUs (for long context)
    PerRequest,         // Assign whole requests to single GPU
    Pipeline            // Pipeline parallel (not implemented)
};

/// Configuration for device manager
struct DeviceManagerConfig {
    std::vector<int32_t> device_ids = {0, 1};
    GPUStrategy default_strategy = GPUStrategy::TensorParallel;
    size_t context_threshold_for_tp = 20000;  // Use TP if context > this
};

/// Device manager for multi-GPU coordination
class DeviceManager {
public:
    explicit DeviceManager(DeviceManagerConfig config = {});
    ~DeviceManager();

    /// Initialize devices
    [[nodiscard]] Result<void> initialize();

    /// Get device information
    [[nodiscard]] const std::vector<DeviceInfo>& devices() const { return devices_; }

    /// Get number of devices
    [[nodiscard]] size_t num_devices() const { return devices_.size(); }

    /// Get recommended strategy for a request
    [[nodiscard]] GPUStrategy recommend_strategy(size_t context_length) const;

    /// Select GPU for a request (when not using TP)
    [[nodiscard]] int32_t select_device() const;

    /// Check if NVLink is available between devices
    [[nodiscard]] bool has_nvlink() const { return has_nvlink_; }

    /// Get configuration
    [[nodiscard]] const DeviceManagerConfig& config() const { return config_; }

    /// Synchronize all devices
    void synchronize_all();

private:
    DeviceManagerConfig config_;
    std::vector<DeviceInfo> devices_;
    bool has_nvlink_ = false;
    mutable std::atomic<size_t> round_robin_counter_{0};
};

}  // namespace qwen::gpu
