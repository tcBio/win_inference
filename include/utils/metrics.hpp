#pragma once

#include <atomic>
#include <string>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <chrono>

namespace qwen {

/// Simple counter metric
class Counter {
public:
    Counter() : value_(0) {}

    void increment(uint64_t delta = 1) {
        value_.fetch_add(delta, std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t value() const {
        return value_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> value_;
};

/// Gauge metric (can go up or down)
class Gauge {
public:
    Gauge() : value_(0) {}

    void set(double v) {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        value_.store(bits, std::memory_order_relaxed);
    }

    void increment(double delta = 1.0) {
        double current = value();
        set(current + delta);
    }

    void decrement(double delta = 1.0) {
        double current = value();
        set(current - delta);
    }

    [[nodiscard]] double value() const {
        uint64_t bits = value_.load(std::memory_order_relaxed);
        double v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }

private:
    std::atomic<uint64_t> value_;
};

/// Histogram for latency distributions
class Histogram {
public:
    explicit Histogram(std::vector<double> bucket_bounds = {
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    }) : bounds_(std::move(bucket_bounds))
       , buckets_(bounds_.size() + 1, 0)
       , sum_(0)
       , count_(0) {}

    void observe(double value) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Find bucket
        size_t idx = bounds_.size();
        for (size_t i = 0; i < bounds_.size(); ++i) {
            if (value <= bounds_[i]) {
                idx = i;
                break;
            }
        }
        buckets_[idx]++;

        sum_ += value;
        count_++;
    }

    [[nodiscard]] std::vector<std::pair<double, uint64_t>> buckets() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::pair<double, uint64_t>> result;
        for (size_t i = 0; i < bounds_.size(); ++i) {
            result.emplace_back(bounds_[i], buckets_[i]);
        }
        result.emplace_back(std::numeric_limits<double>::infinity(), buckets_.back());
        return result;
    }

    [[nodiscard]] double sum() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sum_;
    }

    [[nodiscard]] uint64_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<double> bounds_;
    std::vector<uint64_t> buckets_;
    double sum_;
    uint64_t count_;
};

/// Global metrics registry
class MetricsRegistry {
public:
    static MetricsRegistry& instance();

    Counter& counter(const std::string& name);
    Gauge& gauge(const std::string& name);
    Histogram& histogram(const std::string& name);

    /// Export metrics in Prometheus text format
    [[nodiscard]] std::string export_prometheus() const;

    /// Reset all metrics (for testing)
    void reset();

private:
    MetricsRegistry() = default;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, Counter> counters_;
    std::unordered_map<std::string, Gauge> gauges_;
    std::unordered_map<std::string, Histogram> histograms_;
};

/// Convenience accessors
Counter& counter(const std::string& name);
Gauge& gauge(const std::string& name);
Histogram& histogram(const std::string& name);

/// Scoped histogram observation
class ScopedHistogramTimer {
public:
    ScopedHistogramTimer(Histogram& hist)
        : hist_(hist)
        , start_(std::chrono::steady_clock::now()) {}

    ~ScopedHistogramTimer() {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<double>(end - start_).count();
        hist_.observe(duration);
    }

private:
    Histogram& hist_;
    std::chrono::steady_clock::time_point start_;
};

// Pre-defined metric names
namespace metrics {
    // Request counters
    constexpr const char* REQUESTS_TOTAL = "inference_requests_total";
    constexpr const char* REQUESTS_ACTIVE = "inference_requests_active";
    constexpr const char* QUEUE_DEPTH = "inference_queue_depth";

    // Latency histograms
    constexpr const char* TIME_TO_FIRST_TOKEN = "inference_time_to_first_token_seconds";
    constexpr const char* PREFILL_TIME = "inference_prefill_time_seconds";
    constexpr const char* DECODE_TIME = "inference_decode_time_seconds";

    // Resource gauges
    constexpr const char* KV_CACHE_USED = "inference_kv_cache_used_bytes";
    constexpr const char* KV_CACHE_TOTAL = "inference_kv_cache_total_bytes";
    constexpr const char* GPU_MEMORY_USED = "inference_gpu_memory_used_bytes";

    // Throughput
    constexpr const char* TOKENS_GENERATED = "inference_tokens_generated_total";
    constexpr const char* TOKENS_PER_SECOND = "inference_tokens_per_second";
}

}  // namespace qwen
