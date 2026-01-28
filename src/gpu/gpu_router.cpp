#include "gpu/device_manager.hpp"
#include "scheduler/request.hpp"
#include "utils/logger.hpp"

namespace qwen::gpu {

/// GPU router for request placement decisions
/// Determines which GPU(s) should handle each request

class GPURouter {
public:
    struct RoutingDecision {
        GPUStrategy strategy;
        std::vector<int32_t> device_ids;
        bool requires_nccl;  // Needs NCCL for collective ops
    };

    explicit GPURouter(std::shared_ptr<DeviceManager> device_manager)
        : device_manager_(std::move(device_manager)) {}

    /// Route a request to appropriate GPU(s)
    RoutingDecision route(const scheduler::RequestPtr& request) const {
        size_t context_length = request->input_tokens.size() + request->max_tokens;
        auto strategy = device_manager_->recommend_strategy(context_length);

        RoutingDecision decision;
        decision.strategy = strategy;

        switch (strategy) {
            case GPUStrategy::TensorParallel: {
                // Use all devices for TP
                for (const auto& device : device_manager_->devices()) {
                    decision.device_ids.push_back(device.device_id);
                }
                decision.requires_nccl = true;
                break;
            }
            case GPUStrategy::PerRequest: {
                // Select single device
                decision.device_ids.push_back(device_manager_->select_device());
                decision.requires_nccl = false;
                break;
            }
            case GPUStrategy::Pipeline: {
                // Not implemented - fallback to TP
                for (const auto& device : device_manager_->devices()) {
                    decision.device_ids.push_back(device.device_id);
                }
                decision.requires_nccl = true;
                break;
            }
        }

        log_debug("gpu_router", "routed", {
            {"request_id", request->id},
            {"context_length", context_length},
            {"strategy", strategy == GPUStrategy::TensorParallel ? "tp" : "per_request"},
            {"num_devices", decision.device_ids.size()}
        });

        return decision;
    }

    /// Check if a request can be placed given current memory
    bool can_place(const scheduler::RequestPtr& request) const {
        // Simplified check - in production, would query actual memory usage
        return true;
    }

private:
    std::shared_ptr<DeviceManager> device_manager_;
};

}  // namespace qwen::gpu
