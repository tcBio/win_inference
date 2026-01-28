#include "gpu/device_manager.hpp"
#include "utils/logger.hpp"

#include <cuda_runtime.h>

namespace qwen::gpu {

DeviceManager::DeviceManager(DeviceManagerConfig config)
    : config_(std::move(config)) {}

DeviceManager::~DeviceManager() = default;

Result<void> DeviceManager::initialize() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        return Error::internal("Failed to get CUDA device count: " +
            std::string(cudaGetErrorString(err)));
    }

    if (device_count == 0) {
        return Error::internal("No CUDA devices found");
    }

    // Query each configured device
    for (int32_t device_id : config_.device_ids) {
        if (device_id >= device_count) {
            return Error::internal("Device " + std::to_string(device_id) + " not found");
        }

        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, device_id);
        if (err != cudaSuccess) {
            return Error::internal("Failed to get device properties: " +
                std::string(cudaGetErrorString(err)));
        }

        size_t free_mem, total_mem;
        cudaSetDevice(device_id);
        cudaMemGetInfo(&free_mem, &total_mem);

        DeviceInfo info{
            .device_id = device_id,
            .name = props.name,
            .total_memory = total_mem,
            .free_memory = free_mem,
            .compute_capability_major = props.major,
            .compute_capability_minor = props.minor,
            .has_nvlink = false,
            .nvlink_peer_id = -1
        };

        devices_.push_back(info);

        log_info("device_manager", "device_found", {
            {"device_id", device_id},
            {"name", props.name},
            {"total_memory_gb", total_mem / (1024.0 * 1024.0 * 1024.0)},
            {"free_memory_gb", free_mem / (1024.0 * 1024.0 * 1024.0)},
            {"compute_capability", std::to_string(props.major) + "." + std::to_string(props.minor)}
        });
    }

    // Check NVLink connectivity
    if (devices_.size() >= 2) {
        int can_access;
        err = cudaDeviceCanAccessPeer(&can_access, devices_[0].device_id, devices_[1].device_id);
        if (err == cudaSuccess && can_access) {
            has_nvlink_ = true;
            devices_[0].has_nvlink = true;
            devices_[0].nvlink_peer_id = devices_[1].device_id;
            devices_[1].has_nvlink = true;
            devices_[1].nvlink_peer_id = devices_[0].device_id;

            // Enable peer access
            cudaSetDevice(devices_[0].device_id);
            cudaDeviceEnablePeerAccess(devices_[1].device_id, 0);
            cudaSetDevice(devices_[1].device_id);
            cudaDeviceEnablePeerAccess(devices_[0].device_id, 0);

            log_info("device_manager", "nvlink_enabled", {
                {"device_0", devices_[0].device_id},
                {"device_1", devices_[1].device_id}
            });
        }
    }

    log_info("device_manager", "initialized", {
        {"num_devices", devices_.size()},
        {"has_nvlink", has_nvlink_},
        {"default_strategy", config_.default_strategy == GPUStrategy::TensorParallel ?
            "tensor_parallel" : "per_request"}
    });

    return Result<void>::success();
}

GPUStrategy DeviceManager::recommend_strategy(size_t context_length) const {
    // Use tensor parallel for long contexts
    if (context_length > config_.context_threshold_for_tp && devices_.size() >= 2) {
        return GPUStrategy::TensorParallel;
    }

    // For shorter contexts, per-request assignment is fine
    if (devices_.size() >= 2) {
        return GPUStrategy::PerRequest;
    }

    // Single GPU - no parallelism
    return GPUStrategy::PerRequest;
}

int32_t DeviceManager::select_device() const {
    if (devices_.empty()) return 0;

    // Simple round-robin for load balancing
    size_t idx = round_robin_counter_.fetch_add(1) % devices_.size();
    return devices_[idx].device_id;
}

void DeviceManager::synchronize_all() {
    for (const auto& device : devices_) {
        cudaSetDevice(device.device_id);
        cudaDeviceSynchronize();
    }
}

}  // namespace qwen::gpu
