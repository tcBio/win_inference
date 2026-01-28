#include "scheduler/scheduler.hpp"

// Decode loop implementation details
// The main decode loop is in scheduler.cpp
// This file contains helper functions for decode loop management

namespace qwen::scheduler {

/// V2 Extension Points for Continuous Batching
///
/// The current MVP implementation uses a simple per-request decode loop.
/// To extend to continuous batching, the following changes are needed:
///
/// 1. BatchedDecodeLoop: Instead of iterating over requests individually,
///    batch multiple requests into a single decode_batch() call.
///
/// 2. Dynamic Batching: Add new requests to the active batch mid-generation,
///    which requires:
///    - Padding/masking for variable sequence lengths
///    - KV cache reorganization for new batch members
///
/// 3. Iteration-level scheduling: At each decode step:
///    - Check for completed requests and remove from batch
///    - Check for new requests and add to batch
///    - Handle preemption if higher-priority requests arrive
///
/// 4. Memory management: With continuous batching, KV cache management
///    becomes more complex:
///    - Paged KV cache to avoid fragmentation
///    - Copy-on-write for prefix sharing
///    - Preemption and resumption of requests

/// Batch decode coordinator for V2 (placeholder)
class BatchDecodeCoordinator {
public:
    struct BatchConfig {
        size_t max_batch_size = 8;
        size_t max_total_tokens = 8192;  // Total tokens across batch
        bool enable_prefix_caching = false;
    };

    explicit BatchDecodeCoordinator(BatchConfig config) : config_(config) {}

    /// Try to add a request to the current batch
    bool try_add(RequestPtr request) {
        if (batch_.size() >= config_.max_batch_size) {
            return false;
        }

        // Check total tokens limit
        size_t total_tokens = 0;
        for (const auto& req : batch_) {
            total_tokens += req->kv_cache.current_len;
        }
        total_tokens += request->kv_cache.current_len;

        if (total_tokens > config_.max_total_tokens) {
            return false;
        }

        batch_.push_back(request);
        return true;
    }

    /// Remove completed requests from batch
    void remove_completed() {
        batch_.erase(
            std::remove_if(batch_.begin(), batch_.end(),
                [](const RequestPtr& r) { return r->is_terminal(); }),
            batch_.end());
    }

    /// Get current batch for decode
    const std::vector<RequestPtr>& batch() const { return batch_; }

    /// Check if batch is empty
    bool empty() const { return batch_.empty(); }

    /// Check if batch is full
    bool full() const { return batch_.size() >= config_.max_batch_size; }

private:
    BatchConfig config_;
    std::vector<RequestPtr> batch_;
};

}  // namespace qwen::scheduler
