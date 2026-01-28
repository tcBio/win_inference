#pragma once

#include "api/types.hpp"
#include "utils/cancellation.hpp"

#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <optional>

namespace qwen::api {

/// Event types for SSE streaming
enum class SSEEventType {
    Chunk,      // Normal token chunk
    Done,       // Stream complete
    Error       // Error occurred
};

/// SSE event to be sent to client
struct SSEEvent {
    SSEEventType type;
    std::optional<ChatCompletionChunk> chunk;
    std::optional<ErrorResponse> error;
};

/// Configuration for SSE streamer
struct SSEStreamerConfig {
    size_t max_queue_size = 256;        // Max events in queue before backpressure
    uint32_t flush_interval_ms = 10;    // How often to flush writes
    bool enable_backpressure = true;    // Enable backpressure on slow clients
};

/// Callback type for writing SSE data
using SSEWriteCallback = std::function<bool(const std::string& data)>;

/// SSE streamer for a single request
/// Thread-safe: producer threads push events, consumer thread writes to socket
class SSEStreamer {
public:
    explicit SSEStreamer(
        SSEWriteCallback write_callback,
        CancellationTokenPtr cancel_token,
        SSEStreamerConfig config = {}
    );

    ~SSEStreamer();

    /// Push a token chunk event (called by decode loop)
    /// Returns false if queue is full and backpressure is enabled
    bool push_chunk(ChatCompletionChunk chunk);

    /// Push an error event
    void push_error(ErrorResponse error);

    /// Signal stream completion
    void complete();

    /// Check if streaming is still active
    [[nodiscard]] bool is_active() const;

    /// Check if client disconnected
    [[nodiscard]] bool is_disconnected() const;

    /// Get number of events pending
    [[nodiscard]] size_t pending_count() const;

    /// Run the streaming loop (blocking, call from dedicated thread)
    void run();

    /// Stop the streaming loop
    void stop();

private:
    std::string format_sse_event(const SSEEvent& event) const;
    bool try_write(const std::string& data);

    SSEWriteCallback write_callback_;
    CancellationTokenPtr cancel_token_;
    SSEStreamerConfig config_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<SSEEvent> events_;

    std::atomic<bool> active_{true};
    std::atomic<bool> disconnected_{false};
    std::atomic<bool> completed_{false};
};

/// Factory function
std::unique_ptr<SSEStreamer> make_sse_streamer(
    SSEWriteCallback write_callback,
    CancellationTokenPtr cancel_token,
    SSEStreamerConfig config = {}
);

}  // namespace qwen::api
