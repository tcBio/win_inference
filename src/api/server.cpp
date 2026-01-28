#include "api/server.hpp"
#include "api/routes.hpp"
#include "utils/logger.hpp"

#include <httplib.h>

namespace qwen::api {

class Server::Impl {
public:
    httplib::Server http_server;
};

Server::Server(
    std::shared_ptr<scheduler::Scheduler> scheduler,
    std::shared_ptr<tokenizer::PromptBuilder> prompt_builder,
    ServerConfig config
)
    : impl_(std::make_unique<Impl>())
    , scheduler_(std::move(scheduler))
    , prompt_builder_(std::move(prompt_builder))
    , config_(std::move(config)) {}

Server::~Server() {
    stop();
}

Result<void> Server::start() {
    if (running_.load()) {
        return Error::internal("Server already running");
    }

    // Configure server
    impl_->http_server.set_read_timeout(config_.read_timeout_sec, 0);
    impl_->http_server.set_write_timeout(config_.write_timeout_sec, 0);
    impl_->http_server.set_payload_max_length(config_.max_request_size);

    // CORS headers
    if (config_.enable_cors) {
        impl_->http_server.set_default_headers({
            {"Access-Control-Allow-Origin", "*"},
            {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
            {"Access-Control-Allow-Headers", "Content-Type, Authorization"}
        });

        // Handle OPTIONS preflight
        impl_->http_server.Options(".*", [](const httplib::Request&, httplib::Response& res) {
            res.status = 204;
        });
    }

    // Register routes
    RouteContext ctx{
        .scheduler = scheduler_,
        .prompt_builder = prompt_builder_,
        .validator = &validator_,
        .model_name = config_.model_name
    };
    register_routes(impl_->http_server, ctx);

    // Start server in background thread
    running_.store(true);
    server_thread_ = std::thread([this]() {
        log_info("server", "starting", {
            {"host", config_.host},
            {"port", config_.port}
        });

        if (!impl_->http_server.listen(config_.host, config_.port)) {
            log_error("server", "listen_failed", {
                {"host", config_.host},
                {"port", config_.port}
            });
            running_.store(false);
        }
    });

    // Wait briefly for server to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!running_.load()) {
        return Error::internal("Failed to start server");
    }

    log_info("server", "started", {
        {"host", config_.host},
        {"port", config_.port},
        {"model", config_.model_name}
    });

    return Result<void>::success();
}

void Server::stop() {
    if (!running_.load()) return;

    log_info("server", "stopping", {});

    impl_->http_server.stop();
    running_.store(false);

    if (server_thread_.joinable()) {
        server_thread_.join();
    }

    log_info("server", "stopped", {});
}

bool Server::is_running() const {
    return running_.load();
}

void Server::wait() {
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

}  // namespace qwen::api
