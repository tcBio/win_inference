#include "utils/logger.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/pattern_formatter.h>

#include <iomanip>
#include <sstream>

namespace qwen {

namespace {

std::shared_ptr<spdlog::logger> g_logger;

std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::ostringstream ss;
    ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%S")
       << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    return ss.str();
}

void log_structured(spdlog::level::level_enum level,
                   const std::string& component,
                   const std::string& event,
                   const nlohmann::json& fields,
                   const LogContext& ctx) {
    if (!g_logger) return;

    nlohmann::json log_entry = {
        {"timestamp", get_iso_timestamp()},
        {"level", spdlog::level::to_string_view(level).data()},
        {"component", component},
        {"event", event}
    };

    if (!ctx.request_id.empty()) {
        log_entry["request_id"] = ctx.request_id;
    }
    if (!ctx.trace_id.empty()) {
        log_entry["trace_id"] = ctx.trace_id;
    }
    if (!ctx.span_id.empty()) {
        log_entry["span_id"] = ctx.span_id;
    }

    for (auto& [key, value] : fields.items()) {
        log_entry[key] = value;
    }

    g_logger->log(level, "{}", log_entry.dump());
}

}  // namespace

void init_logger(const LoggerConfig& config) {
    std::vector<spdlog::sink_ptr> sinks;

    if (config.output == "console" || config.output == "both") {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sinks.push_back(console_sink);
    }

    if (config.output == "file" || config.output == "both") {
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            config.file_path, config.max_file_size, config.max_files);
        sinks.push_back(file_sink);
    }

    g_logger = std::make_shared<spdlog::logger>("qwen", sinks.begin(), sinks.end());

    // Set log level
    if (config.level == "debug") {
        g_logger->set_level(spdlog::level::debug);
    } else if (config.level == "info") {
        g_logger->set_level(spdlog::level::info);
    } else if (config.level == "warn") {
        g_logger->set_level(spdlog::level::warn);
    } else if (config.level == "error") {
        g_logger->set_level(spdlog::level::err);
    }

    // Use simple pattern since we format JSON ourselves
    g_logger->set_pattern("%v");

    spdlog::set_default_logger(g_logger);
    spdlog::flush_every(std::chrono::seconds(1));
}

std::shared_ptr<spdlog::logger> get_logger() {
    return g_logger;
}

void log_info(const std::string& component, const std::string& event,
              const nlohmann::json& fields, const LogContext& ctx) {
    log_structured(spdlog::level::info, component, event, fields, ctx);
}

void log_warn(const std::string& component, const std::string& event,
              const nlohmann::json& fields, const LogContext& ctx) {
    log_structured(spdlog::level::warn, component, event, fields, ctx);
}

void log_error(const std::string& component, const std::string& event,
               const nlohmann::json& fields, const LogContext& ctx) {
    log_structured(spdlog::level::err, component, event, fields, ctx);
}

void log_debug(const std::string& component, const std::string& event,
               const nlohmann::json& fields, const LogContext& ctx) {
    log_structured(spdlog::level::debug, component, event, fields, ctx);
}

ScopedTimer::ScopedTimer(std::string component, std::string event, LogContext ctx)
    : component_(std::move(component))
    , event_(std::move(event))
    , ctx_(std::move(ctx))
    , start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    fields_["duration_ms"] = elapsed_ms();
    log_info(component_, event_ + "_completed", fields_, ctx_);
}

void ScopedTimer::add_field(const std::string& key, const nlohmann::json& value) {
    fields_[key] = value;
}

double ScopedTimer::elapsed_ms() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_).count();
}

}  // namespace qwen
