#include "utils/metrics.hpp"

#include <sstream>
#include <iomanip>

namespace qwen {

MetricsRegistry& MetricsRegistry::instance() {
    static MetricsRegistry registry;
    return registry;
}

Counter& MetricsRegistry::counter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return counters_[name];
}

Gauge& MetricsRegistry::gauge(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return gauges_[name];
}

Histogram& MetricsRegistry::histogram(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = histograms_.find(name);
    if (it == histograms_.end()) {
        auto [inserted, _] = histograms_.emplace(name, Histogram{});
        return inserted->second;
    }
    return it->second;
}

std::string MetricsRegistry::export_prometheus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream ss;

    // Export counters
    for (const auto& [name, counter] : counters_) {
        ss << "# TYPE " << name << " counter\n";
        ss << name << " " << counter.value() << "\n";
    }

    // Export gauges
    for (const auto& [name, gauge] : gauges_) {
        ss << "# TYPE " << name << " gauge\n";
        ss << name << " " << std::fixed << std::setprecision(6) << gauge.value() << "\n";
    }

    // Export histograms
    for (const auto& [name, hist] : histograms_) {
        ss << "# TYPE " << name << " histogram\n";

        auto buckets = hist.buckets();
        uint64_t cumulative = 0;
        for (const auto& [bound, count] : buckets) {
            cumulative += count;
            if (std::isinf(bound)) {
                ss << name << "_bucket{le=\"+Inf\"} " << cumulative << "\n";
            } else {
                ss << name << "_bucket{le=\"" << std::fixed << std::setprecision(3)
                   << bound << "\"} " << cumulative << "\n";
            }
        }
        ss << name << "_sum " << std::fixed << std::setprecision(6) << hist.sum() << "\n";
        ss << name << "_count " << hist.count() << "\n";
    }

    return ss.str();
}

void MetricsRegistry::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_.clear();
    gauges_.clear();
    histograms_.clear();
}

Counter& counter(const std::string& name) {
    return MetricsRegistry::instance().counter(name);
}

Gauge& gauge(const std::string& name) {
    return MetricsRegistry::instance().gauge(name);
}

Histogram& histogram(const std::string& name) {
    return MetricsRegistry::instance().histogram(name);
}

}  // namespace qwen
