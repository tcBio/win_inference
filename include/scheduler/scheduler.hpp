#pragma once

#include "scheduler/request.hpp"
#include "backend/runtime_interface.hpp"
#include "kvcache/kv_allocator.hpp"
#include "tokenizer/tokenizer.hpp"
#include "core/sampler.hpp"
#include "core/stop_checker.hpp"
#include "utils/result.hpp"

#include <memory>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unordered_map>

namespace qwen::scheduler {

/// Scheduler configuration
struct SchedulerConfig {
    size_t max_queue_size = 100;            // Max pending requests
    size_t max_active_requests = 4;         // Max concurrent decoding requests
    uint32_t decode_loop_interval_us = 100; // Decode loop sleep when idle
    bool enable_continuous_batching = false; // V2 feature
};

/// Request registry for tracking all active requests
class RequestRegistry {
public:
    RequestRegistry();
    ~RequestRegistry();

    /// Register a new request
    void add(RequestPtr request);

    /// Remove a request
    void remove(const std::string& request_id);

    /// Get a request by ID
    [[nodiscard]] RequestPtr get(const std::string& request_id) const;

    /// Get all requests in a given state
    [[nodiscard]] std::vector<RequestPtr> get_by_state(RequestState state) const;

    /// Get count of requests in each state
    [[nodiscard]] std::unordered_map<RequestState, size_t> state_counts() const;

    /// Cancel a request
    void cancel(const std::string& request_id);

    /// Cancel all requests
    void cancel_all();

    /// Get total count
    [[nodiscard]] size_t count() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, RequestPtr> requests_;
};

/// Main scheduler for inference requests
/// Owns the decode loop thread and manages request lifecycle
class Scheduler {
public:
    Scheduler(
        std::shared_ptr<backend::IModelRuntime> runtime,
        std::shared_ptr<kvcache::IKVCacheAllocator> kv_allocator,
        std::shared_ptr<tokenizer::ITokenizer> tokenizer,
        SchedulerConfig config = {}
    );

    ~Scheduler();

    /// Start the scheduler (begins decode loop thread)
    Result<void> start();

    /// Stop the scheduler gracefully
    void stop();

    /// Submit a request for processing
    /// Returns immediately; use callbacks for results
    Result<void> submit(RequestPtr request);

    /// Cancel a request
    void cancel(const std::string& request_id);

    /// Get a request by ID
    [[nodiscard]] RequestPtr get_request(const std::string& request_id) const;

    /// Get scheduler statistics
    struct Stats {
        size_t queued_count = 0;
        size_t active_count = 0;
        size_t completed_count = 0;
        size_t cancelled_count = 0;
        size_t errored_count = 0;
    };
    [[nodiscard]] Stats stats() const;

    /// Check if scheduler is running
    [[nodiscard]] bool is_running() const;

private:
    /// Admission control: check if request can be admitted
    [[nodiscard]] Result<void> admit(RequestPtr request);

    /// Run prefill for a request
    [[nodiscard]] Result<void> run_prefill(RequestPtr request);

    /// Run one decode step for a request
    [[nodiscard]] Result<bool> run_decode_step(RequestPtr request);

    /// Main decode loop (runs in separate thread)
    void decode_loop();

    /// Process queued requests (try to start prefill)
    void process_queue();

    /// Complete a request
    void complete_request(RequestPtr request, api::FinishReason reason);

    /// Fail a request with error
    void fail_request(RequestPtr request, Error error);

    /// Free resources for a request
    void free_resources(RequestPtr request);

    // Dependencies
    std::shared_ptr<backend::IModelRuntime> runtime_;
    std::shared_ptr<kvcache::IKVCacheAllocator> kv_allocator_;
    std::shared_ptr<tokenizer::ITokenizer> tokenizer_;

    // Components
    std::unique_ptr<core::Sampler> sampler_;
    std::unique_ptr<core::StopChecker> stop_checker_;
    RequestRegistry registry_;

    // Configuration
    SchedulerConfig config_;

    // Queue
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<RequestPtr> pending_queue_;

    // Active requests (in prefill or decode)
    std::mutex active_mutex_;
    std::vector<RequestPtr> active_requests_;

    // Decode loop thread
    std::thread decode_thread_;
    std::atomic<bool> running_{false};

    // Statistics
    std::atomic<uint64_t> completed_count_{0};
    std::atomic<uint64_t> cancelled_count_{0};
    std::atomic<uint64_t> errored_count_{0};
};

}  // namespace qwen::scheduler
