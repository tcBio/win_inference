#include "gpu/gpu_router.hpp"

namespace qwen::gpu {

GPURouter::GPURouter(std::shared_ptr<DeviceManager> device_manager)
    : device_manager_(std::move(device_manager)) {}

void GPURouter::initialize_budgets(size_t kv_memory_per_gpu) {
    std::lock_guard<std::mutex> lock(budget_mutex_);
    budgets_.clear();

    for (const auto& device : device_manager_->devices()) {
        GPUBudget budget;
        budget.device_id = device.device_id;
        budget.total_memory = kv_memory_per_gpu;
        budget.used_memory.store(0);
        budget.reserved_memory.store(0);
        budgets_.push_back(std::move(budget));
    }

    log_info("gpu_router", "budgets_initialized", {
        {"num_devices", budgets_.size()},
        {"memory_per_gpu_gb", kv_memory_per_gpu / (1024.0 * 1024.0 * 1024.0)}
    });
}

RoutingDecision GPURouter::route(const scheduler::RequestPtr& request) const {
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
            // Select device with most available memory
            int32_t best_device = device_manager_->select_device();
            size_t best_available = 0;

            std::lock_guard<std::mutex> lock(budget_mutex_);
            for (const auto& budget : budgets_) {
                size_t avail = budget.available();
                if (avail > best_available) {
                    best_available = avail;
                    best_device = budget.device_id;
                }
            }

            decision.device_ids.push_back(best_device);
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

bool GPURouter::can_place(const scheduler::RequestPtr& request, size_t estimated_bytes) const {
    size_t context_length = request->input_tokens.size() + request->max_tokens;
    auto strategy = device_manager_->recommend_strategy(context_length);

    std::lock_guard<std::mutex> lock(budget_mutex_);

    if (strategy == GPUStrategy::TensorParallel) {
        // For TP, need memory on all devices (split across devices)
        size_t per_device = estimated_bytes / budgets_.size();
        for (const auto& budget : budgets_) {
            if (!budget.can_fit(per_device)) {
                return false;
            }
        }
        return true;
    } else {
        // For per-request, need at least one device with enough memory
        for (const auto& budget : budgets_) {
            if (budget.can_fit(estimated_bytes)) {
                return true;
            }
        }
        return false;
    }
}

void GPURouter::reserve(const RoutingDecision& decision, size_t bytes) {
    size_t per_device = decision.strategy == GPUStrategy::TensorParallel
        ? bytes / decision.device_ids.size()
        : bytes;

    std::lock_guard<std::mutex> lock(budget_mutex_);
    for (int32_t device_id : decision.device_ids) {
        for (auto& budget : budgets_) {
            if (budget.device_id == device_id) {
                budget.reserved_memory.fetch_add(per_device, std::memory_order_acq_rel);
                break;
            }
        }
    }
}

void GPURouter::commit(const RoutingDecision& decision, size_t bytes) {
    size_t per_device = decision.strategy == GPUStrategy::TensorParallel
        ? bytes / decision.device_ids.size()
        : bytes;

    std::lock_guard<std::mutex> lock(budget_mutex_);
    for (int32_t device_id : decision.device_ids) {
        for (auto& budget : budgets_) {
            if (budget.device_id == device_id) {
                // Move from reserved to used
                budget.reserved_memory.fetch_sub(per_device, std::memory_order_acq_rel);
                budget.used_memory.fetch_add(per_device, std::memory_order_acq_rel);
                break;
            }
        }
    }
}

void GPURouter::unreserve(const RoutingDecision& decision, size_t bytes) {
    size_t per_device = decision.strategy == GPUStrategy::TensorParallel
        ? bytes / decision.device_ids.size()
        : bytes;

    std::lock_guard<std::mutex> lock(budget_mutex_);
    for (int32_t device_id : decision.device_ids) {
        for (auto& budget : budgets_) {
            if (budget.device_id == device_id) {
                budget.reserved_memory.fetch_sub(per_device, std::memory_order_acq_rel);
                break;
            }
        }
    }
}

void GPURouter::free(const std::vector<int32_t>& device_ids, size_t bytes) {
    size_t per_device = device_ids.size() > 1
        ? bytes / device_ids.size()
        : bytes;

    std::lock_guard<std::mutex> lock(budget_mutex_);
    for (int32_t device_id : device_ids) {
        for (auto& budget : budgets_) {
            if (budget.device_id == device_id) {
                budget.used_memory.fetch_sub(per_device, std::memory_order_acq_rel);
                break;
            }
        }
    }
}

const GPUBudget* GPURouter::get_budget(int32_t device_id) const {
    std::lock_guard<std::mutex> lock(budget_mutex_);
    for (const auto& budget : budgets_) {
        if (budget.device_id == device_id) {
            return &budget;
        }
    }
    return nullptr;
}

}  // namespace qwen::gpu
