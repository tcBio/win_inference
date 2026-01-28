#pragma once

#include "api/types.hpp"
#include "api/request_validator.hpp"
#include "scheduler/scheduler.hpp"
#include "tokenizer/prompt_builder.hpp"
#include "utils/result.hpp"

#include <memory>
#include <string>
#include <atomic>
#include <thread>

namespace qwen::api {

/// Server configuration
struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int num_threads = 4;              // HTTP worker threads
    size_t max_request_size = 10 * 1024 * 1024;  // 10 MB
    int read_timeout_sec = 300;       // 5 minutes
    int write_timeout_sec = 300;
    bool enable_cors = true;
    std::string model_name = "qwen2.5-32b-instruct";
};

/// HTTP server for OpenAI-compatible API
class Server {
public:
    Server(
        std::shared_ptr<scheduler::Scheduler> scheduler,
        std::shared_ptr<tokenizer::PromptBuilder> prompt_builder,
        ServerConfig config = {}
    );

    ~Server();

    /// Start the server (non-blocking)
    Result<void> start();

    /// Stop the server
    void stop();

    /// Check if server is running
    [[nodiscard]] bool is_running() const;

    /// Get server configuration
    [[nodiscard]] const ServerConfig& config() const { return config_; }

    /// Wait for server to stop
    void wait();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    std::shared_ptr<scheduler::Scheduler> scheduler_;
    std::shared_ptr<tokenizer::PromptBuilder> prompt_builder_;
    ServerConfig config_;
    RequestValidator validator_;

    std::atomic<bool> running_{false};
    std::thread server_thread_;
};

}  // namespace qwen::api
