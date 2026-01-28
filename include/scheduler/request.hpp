#pragma once

#include "api/types.hpp"
#include "kvcache/kv_allocator.hpp"
#include "utils/cancellation.hpp"
#include "utils/metrics.hpp"

#include <string>
#include <chrono>
#include <atomic>
#include <memory>
#include <vector>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

namespace qwen::scheduler {

/// Thread-safe token queue for SSE streaming
/// Producer: scheduler thread, Consumer: HTTP thread
struct TokenEvent {
    enum class Type { Token, Complete, Error };
    Type type;
    int32_t token_id = 0;
    std::string text;
    api::FinishReason finish_reason = api::FinishReason::None;
    std::optional<Error> error;
};

class TokenQueue {
public:
    explicit TokenQueue(size_t max_size = 1024) : max_size_(max_size) {}

    /// Push a token event (producer side)
    bool push(TokenEvent event) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return false;
        if (events_.size() >= max_size_) return false;  // Backpressure
        events_.push(std::move(event));
        cv_.notify_one();
        return true;
    }

    /// Pop a token event with timeout (consumer side)
    /// Returns nullopt on timeout or if queue is closed and empty
    std::optional<TokenEvent> pop(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] {
            return !events_.empty() || closed_;
        })) {
            return std::nullopt;  // Timeout
        }
        if (events_.empty()) {
            return std::nullopt;  // Closed and empty
        }
        auto event = std::move(events_.front());
        events_.pop();
        return event;
    }

    /// Close the queue (no more pushes allowed)
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }

    /// Check if closed
    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    /// Check if empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<TokenEvent> events_;
    size_t max_size_;
    bool closed_ = false;
};

/// Request state in the scheduler
enum class RequestState {
    Received,       // Just received, not yet validated
    Queued,         // Validated and waiting in queue
    Prefilling,     // Running prefill pass
    Decoding,       // In decode loop
    Completed,      // Successfully completed
    Cancelled,      // Cancelled by client or system
    Errored,        // Failed with error
    Rejected        // Rejected (e.g., OOM at admission)
};

inline std::string state_to_string(RequestState state) {
    switch (state) {
        case RequestState::Received: return "received";
        case RequestState::Queued: return "queued";
        case RequestState::Prefilling: return "prefilling";
        case RequestState::Decoding: return "decoding";
        case RequestState::Completed: return "completed";
        case RequestState::Cancelled: return "cancelled";
        case RequestState::Errored: return "errored";
        case RequestState::Rejected: return "rejected";
    }
    return "unknown";
}

/// Callback for streaming token deltas
using TokenCallback = std::function<void(int32_t token_id, const std::string& text)>;

/// Callback for completion
using CompletionCallback = std::function<void(api::FinishReason reason)>;

/// Callback for errors
using ErrorCallback = std::function<void(const Error& error)>;

/// Timing information for a request
struct RequestTiming {
    std::chrono::steady_clock::time_point received_at;
    std::chrono::steady_clock::time_point queued_at;
    std::chrono::steady_clock::time_point prefill_start;
    std::chrono::steady_clock::time_point prefill_end;
    std::chrono::steady_clock::time_point first_token_at;
    std::chrono::steady_clock::time_point completed_at;

    [[nodiscard]] double queue_time_ms() const {
        return std::chrono::duration<double, std::milli>(prefill_start - queued_at).count();
    }

    [[nodiscard]] double prefill_time_ms() const {
        return std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
    }

    [[nodiscard]] double time_to_first_token_ms() const {
        return std::chrono::duration<double, std::milli>(first_token_at - received_at).count();
    }

    [[nodiscard]] double total_time_ms() const {
        return std::chrono::duration<double, std::milli>(completed_at - received_at).count();
    }
};

/// Inference request with all state
struct Request {
    // Identity
    std::string id;
    std::string trace_id;

    // Input
    api::ChatCompletionRequest api_request;
    std::vector<int32_t> input_tokens;
    std::string formatted_prompt;

    // Sampling parameters
    double temperature = 1.0;
    double top_p = 1.0;
    int32_t max_tokens = 2048;
    std::vector<std::string> stop_sequences;
    std::optional<int64_t> seed;

    // State
    std::atomic<RequestState> state{RequestState::Received};
    CancellationTokenPtr cancel_token;

    // Output
    std::vector<int32_t> output_tokens;
    std::string output_text;
    api::FinishReason finish_reason = api::FinishReason::None;
    std::optional<Error> error;

    // Prefill output logits (for first token sampling)
    std::vector<float> prefill_logits;
    int32_t prefill_logits_vocab_size = 0;

    // Resources
    kvcache::KVCacheHandle kv_cache;
    int32_t gpu_id = -1;
    std::vector<int32_t> assigned_devices;  // For tensor parallel

    // Timing
    RequestTiming timing;

    // Callbacks (for streaming) - DEPRECATED: use token_queue instead
    TokenCallback on_token;
    CompletionCallback on_complete;
    ErrorCallback on_error;

    // Thread-safe streaming via queue (preferred)
    std::shared_ptr<TokenQueue> token_queue;

    // Streaming state
    bool is_streaming = false;

    // Statistics
    int32_t prompt_tokens() const { return static_cast<int32_t>(input_tokens.size()); }
    int32_t completion_tokens() const { return static_cast<int32_t>(output_tokens.size()); }
    int32_t total_tokens() const { return prompt_tokens() + completion_tokens(); }

    // State transitions
    void set_state(RequestState new_state) {
        state.store(new_state, std::memory_order_release);
    }

    [[nodiscard]] RequestState get_state() const {
        return state.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool is_terminal() const {
        auto s = get_state();
        return s == RequestState::Completed ||
               s == RequestState::Cancelled ||
               s == RequestState::Errored ||
               s == RequestState::Rejected;
    }

    [[nodiscard]] bool is_cancelled() const {
        return cancel_token && cancel_token->is_cancelled();
    }
};

using RequestPtr = std::shared_ptr<Request>;

/// Create a new request with generated ID
RequestPtr make_request(const api::ChatCompletionRequest& api_request);

/// Generate a unique request ID
std::string generate_request_id();

}  // namespace qwen::scheduler
