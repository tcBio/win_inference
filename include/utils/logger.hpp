#pragma once

#include <string>
#include <memory>
#include <chrono>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace qwen {

/// Structured log context for request tracing
struct LogContext {
    std::string request_id;
    std::string trace_id;
    std::string span_id;

    nlohmann::json to_json() const {
        return {
            {"request_id", request_id},
            {"trace_id", trace_id},
            {"span_id", span_id}
        };
    }
};

/// Logger configuration
struct LoggerConfig {
    std::string level = "info";          // debug, info, warn, error
    std::string output = "console";      // console, file, both
    std::string file_path = "server.log";
    bool json_format = true;
    size_t max_file_size = 100 * 1024 * 1024;  // 100MB
    size_t max_files = 5;
};

/// Initialize the logging system
void init_logger(const LoggerConfig& config);

/// Get the global logger instance
std::shared_ptr<spdlog::logger> get_logger();

/// Structured logging helpers
void log_info(const std::string& component, const std::string& event,
              const nlohmann::json& fields = {}, const LogContext& ctx = {});

void log_warn(const std::string& component, const std::string& event,
              const nlohmann::json& fields = {}, const LogContext& ctx = {});

void log_error(const std::string& component, const std::string& event,
               const nlohmann::json& fields = {}, const LogContext& ctx = {});

void log_debug(const std::string& component, const std::string& event,
               const nlohmann::json& fields = {}, const LogContext& ctx = {});

/// Scoped timer for measuring durations
class ScopedTimer {
public:
    ScopedTimer(std::string component, std::string event, LogContext ctx = {});
    ~ScopedTimer();

    void add_field(const std::string& key, const nlohmann::json& value);
    double elapsed_ms() const;

private:
    std::string component_;
    std::string event_;
    LogContext ctx_;
    nlohmann::json fields_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace qwen
