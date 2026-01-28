#include "scheduler/scheduler.hpp"
#include "utils/logger.hpp"

namespace qwen::scheduler {

RequestRegistry::RequestRegistry() = default;
RequestRegistry::~RequestRegistry() = default;

void RequestRegistry::add(RequestPtr request) {
    std::lock_guard<std::mutex> lock(mutex_);
    requests_[request->id] = request;

    log_debug("request_registry", "request_added", {
        {"request_id", request->id},
        {"total_requests", requests_.size()}
    });
}

void RequestRegistry::remove(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = requests_.find(request_id);
    if (it != requests_.end()) {
        requests_.erase(it);
        log_debug("request_registry", "request_removed", {
            {"request_id", request_id},
            {"total_requests", requests_.size()}
        });
    }
}

RequestPtr RequestRegistry::get(const std::string& request_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = requests_.find(request_id);
    return it != requests_.end() ? it->second : nullptr;
}

std::vector<RequestPtr> RequestRegistry::get_by_state(RequestState state) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<RequestPtr> result;
    for (const auto& [id, request] : requests_) {
        if (request->get_state() == state) {
            result.push_back(request);
        }
    }
    return result;
}

std::unordered_map<RequestState, size_t> RequestRegistry::state_counts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<RequestState, size_t> counts;
    for (const auto& [id, request] : requests_) {
        counts[request->get_state()]++;
    }
    return counts;
}

void RequestRegistry::cancel(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = requests_.find(request_id);
    if (it != requests_.end() && it->second->cancel_token) {
        it->second->cancel_token->cancel();
        log_info("request_registry", "request_cancelled", {
            {"request_id", request_id}
        });
    }
}

void RequestRegistry::cancel_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [id, request] : requests_) {
        if (request->cancel_token && !request->is_terminal()) {
            request->cancel_token->cancel();
        }
    }
    log_info("request_registry", "all_requests_cancelled", {
        {"count", requests_.size()}
    });
}

size_t RequestRegistry::count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return requests_.size();
}

}  // namespace qwen::scheduler
