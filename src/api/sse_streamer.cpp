#include "api/sse_streamer.hpp"
#include "utils/logger.hpp"

#include <chrono>

namespace qwen::api {

SSEStreamer::SSEStreamer(
    SSEWriteCallback write_callback,
    CancellationTokenPtr cancel_token,
    SSEStreamerConfig config
)
    : write_callback_(std::move(write_callback))
    , cancel_token_(std::move(cancel_token))
    , config_(config) {

    // Register cancellation callback
    if (cancel_token_) {
        cancel_token_->on_cancel([this]() {
            stop();
        });
    }
}

SSEStreamer::~SSEStreamer() {
    stop();
}

bool SSEStreamer::push_chunk(ChatCompletionChunk chunk) {
    if (!active_.load() || disconnected_.load()) {
        return false;
    }

    std::unique_lock<std::mutex> lock(mutex_);

    // Backpressure check
    if (config_.enable_backpressure && events_.size() >= config_.max_queue_size) {
        log_warn("sse_streamer", "queue_full", {{"queue_size", events_.size()}});
        return false;
    }

    events_.push(SSEEvent{
        .type = SSEEventType::Chunk,
        .chunk = std::move(chunk),
        .error = std::nullopt
    });

    cv_.notify_one();
    return true;
}

void SSEStreamer::push_error(ErrorResponse error) {
    if (!active_.load()) return;

    std::unique_lock<std::mutex> lock(mutex_);
    events_.push(SSEEvent{
        .type = SSEEventType::Error,
        .chunk = std::nullopt,
        .error = std::move(error)
    });
    cv_.notify_one();
}

void SSEStreamer::complete() {
    if (!active_.load()) return;

    std::unique_lock<std::mutex> lock(mutex_);
    completed_.store(true);
    events_.push(SSEEvent{
        .type = SSEEventType::Done,
        .chunk = std::nullopt,
        .error = std::nullopt
    });
    cv_.notify_one();
}

bool SSEStreamer::is_active() const {
    return active_.load() && !disconnected_.load();
}

bool SSEStreamer::is_disconnected() const {
    return disconnected_.load();
}

size_t SSEStreamer::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_.size();
}

void SSEStreamer::run() {
    while (active_.load()) {
        SSEEvent event;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(config_.flush_interval_ms),
                        [this] { return !events_.empty() || !active_.load(); });

            if (!active_.load() && events_.empty()) {
                break;
            }

            if (events_.empty()) {
                continue;
            }

            event = std::move(events_.front());
            events_.pop();
        }

        // Format and write event
        std::string data = format_sse_event(event);
        if (!try_write(data)) {
            // Write failed, client disconnected
            disconnected_.store(true);
            if (cancel_token_) {
                cancel_token_->cancel();
            }
            break;
        }

        // Check for terminal events
        if (event.type == SSEEventType::Done || event.type == SSEEventType::Error) {
            active_.store(false);
            break;
        }
    }
}

void SSEStreamer::stop() {
    active_.store(false);
    cv_.notify_all();
}

std::string SSEStreamer::format_sse_event(const SSEEvent& event) const {
    std::string result;

    switch (event.type) {
        case SSEEventType::Chunk: {
            if (event.chunk) {
                nlohmann::json j;
                to_json(j, *event.chunk);
                result = "data: " + j.dump() + "\n\n";
            }
            break;
        }
        case SSEEventType::Done: {
            result = "data: [DONE]\n\n";
            break;
        }
        case SSEEventType::Error: {
            if (event.error) {
                nlohmann::json j;
                to_json(j, *event.error);
                result = "data: " + j.dump() + "\n\n";
            }
            break;
        }
    }

    return result;
}

bool SSEStreamer::try_write(const std::string& data) {
    if (!write_callback_) return false;

    try {
        return write_callback_(data);
    } catch (...) {
        return false;
    }
}

std::unique_ptr<SSEStreamer> make_sse_streamer(
    SSEWriteCallback write_callback,
    CancellationTokenPtr cancel_token,
    SSEStreamerConfig config
) {
    return std::make_unique<SSEStreamer>(
        std::move(write_callback),
        std::move(cancel_token),
        config
    );
}

}  // namespace qwen::api
