#pragma once

#include <atomic>
#include <memory>
#include <functional>
#include <mutex>
#include <vector>

namespace qwen {

/// Thread-safe cancellation token for async operations
class CancellationToken {
public:
    CancellationToken() : cancelled_(false) {}

    /// Request cancellation
    void cancel() {
        if (cancelled_.exchange(true)) {
            return;  // Already cancelled
        }

        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& callback : callbacks_) {
            callback();
        }
        callbacks_.clear();
    }

    /// Check if cancelled
    [[nodiscard]] bool is_cancelled() const {
        return cancelled_.load(std::memory_order_acquire);
    }

    /// Register a callback to be called on cancellation
    /// If already cancelled, callback is invoked immediately
    void on_cancel(std::function<void()> callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cancelled_.load(std::memory_order_acquire)) {
            callback();
        } else {
            callbacks_.push_back(std::move(callback));
        }
    }

    /// Reset the token (for reuse)
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_.store(false, std::memory_order_release);
        callbacks_.clear();
    }

private:
    std::atomic<bool> cancelled_;
    std::mutex mutex_;
    std::vector<std::function<void()>> callbacks_;
};

/// Shared cancellation token for multiple consumers
using CancellationTokenPtr = std::shared_ptr<CancellationToken>;

/// Create a new cancellation token
inline CancellationTokenPtr make_cancellation_token() {
    return std::make_shared<CancellationToken>();
}

/// RAII guard that cancels on destruction
class CancellationGuard {
public:
    explicit CancellationGuard(CancellationTokenPtr token)
        : token_(std::move(token)) {}

    ~CancellationGuard() {
        if (token_) {
            token_->cancel();
        }
    }

    void release() { token_.reset(); }

private:
    CancellationTokenPtr token_;
};

}  // namespace qwen
