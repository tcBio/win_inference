#include "api/routes.hpp"
#include "utils/logger.hpp"
#include "utils/metrics.hpp"

#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>

namespace qwen::api {

std::string generate_completion_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint64_t> dist;

    std::ostringstream ss;
    ss << "chatcmpl-" << std::hex << std::setfill('0')
       << std::setw(16) << dist(gen);
    return ss.str();
}

int64_t current_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void send_error(
    httplib::Response& res,
    int status,
    const std::string& type,
    const std::string& code,
    const std::string& message) {

    ErrorResponse err{
        .type = type,
        .code = code,
        .message = message
    };

    nlohmann::json j;
    to_json(j, err);

    res.status = status;
    res.set_content(j.dump(), "application/json");
}

void send_error(httplib::Response& res, int status, const Error& error) {
    send_error(res, status, "error", error.code, error.message);
}

void register_routes(httplib::Server& server, RouteContext& ctx) {
    // Store context in a shared_ptr for lambda capture
    auto ctx_ptr = std::make_shared<RouteContext>(ctx);

    // POST /v1/chat/completions
    server.Post("/v1/chat/completions",
        [ctx_ptr](const httplib::Request& req, httplib::Response& res) {
            handlers::chat_completions(req, res, *ctx_ptr);
        });

    // GET /v1/models
    server.Get("/v1/models",
        [ctx_ptr](const httplib::Request& req, httplib::Response& res) {
            handlers::list_models(req, res, *ctx_ptr);
        });

    // GET /health
    server.Get("/health",
        [ctx_ptr](const httplib::Request& req, httplib::Response& res) {
            handlers::health_check(req, res, *ctx_ptr);
        });

    // GET /metrics
    server.Get("/metrics",
        [ctx_ptr](const httplib::Request& req, httplib::Response& res) {
            handlers::metrics(req, res, *ctx_ptr);
        });

    log_info("routes", "registered", {
        {"endpoints", std::vector<std::string>{
            "/v1/chat/completions",
            "/v1/models",
            "/health",
            "/metrics"
        }}
    });
}

namespace handlers {

void chat_completions(
    const httplib::Request& req,
    httplib::Response& res,
    RouteContext& ctx) {

    LogContext log_ctx;
    log_ctx.request_id = generate_completion_id();

    log_info("api", "request_received", {
        {"endpoint", "/v1/chat/completions"},
        {"content_length", req.body.size()}
    }, log_ctx);

    // Parse request
    ChatCompletionRequest api_request;
    try {
        auto j = nlohmann::json::parse(req.body);
        from_json(j, api_request);
    } catch (const std::exception& e) {
        log_warn("api", "parse_error", {{"error", e.what()}}, log_ctx);
        send_error(res, 400, "invalid_request_error", "parse_error",
            std::string("Failed to parse request: ") + e.what());
        return;
    }

    // Validate request
    auto validation_result = ctx.validator->validate(api_request);
    if (validation_result.is_error()) {
        log_warn("api", "validation_error", {
            {"error", validation_result.error().message}
        }, log_ctx);
        send_error(res, 400, validation_result.error());
        return;
    }

    // Apply defaults
    ctx.validator->apply_defaults(api_request);

    // Build prompt
    auto prompt_result = ctx.prompt_builder->build(api_request.messages);
    if (prompt_result.is_error()) {
        log_error("api", "prompt_build_error", {
            {"error", prompt_result.error().message}
        }, log_ctx);
        send_error(res, 400, prompt_result.error());
        return;
    }

    const auto& built_prompt = prompt_result.value();

    // Create scheduler request
    auto request = scheduler::make_request(api_request);
    request->trace_id = log_ctx.request_id;
    request->input_tokens = built_prompt.token_ids;
    request->formatted_prompt = built_prompt.formatted_text;

    if (api_request.stream) {
        // Streaming response
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        std::string completion_id = log_ctx.request_id;
        std::string model_name = ctx.model_name;

        // Set up streaming
        res.set_chunked_content_provider(
            "text/event-stream",
            [request, completion_id, model_name, &ctx](
                size_t /*offset*/, httplib::DataSink& sink) {

                // Send initial chunk with role
                ChatCompletionChunk initial_chunk{
                    .id = completion_id,
                    .created = current_timestamp(),
                    .model = model_name,
                    .choices = {{
                        .index = 0,
                        .delta = {.role = Role::Assistant}
                    }}
                };
                nlohmann::json j;
                to_json(j, initial_chunk);
                sink.write("data: " + j.dump() + "\n\n");

                // Set up callbacks
                std::atomic<bool> done{false};
                std::mutex mtx;
                std::condition_variable cv;

                request->on_token = [&](int32_t /*token_id*/, const std::string& text) {
                    if (text.empty()) return;

                    ChatCompletionChunk chunk{
                        .id = completion_id,
                        .created = current_timestamp(),
                        .model = model_name,
                        .choices = {{
                            .index = 0,
                            .delta = {.content = text}
                        }}
                    };
                    nlohmann::json j;
                    to_json(j, chunk);
                    sink.write("data: " + j.dump() + "\n\n");
                };

                request->on_complete = [&](FinishReason reason) {
                    ChatCompletionChunk chunk{
                        .id = completion_id,
                        .created = current_timestamp(),
                        .model = model_name,
                        .choices = {{
                            .index = 0,
                            .delta = {},
                            .finish_reason = reason
                        }}
                    };
                    nlohmann::json j;
                    to_json(j, chunk);
                    sink.write("data: " + j.dump() + "\n\n");
                    sink.write("data: [DONE]\n\n");

                    std::lock_guard<std::mutex> lock(mtx);
                    done.store(true);
                    cv.notify_one();
                };

                request->on_error = [&](const Error& /*error*/) {
                    std::lock_guard<std::mutex> lock(mtx);
                    done.store(true);
                    cv.notify_one();
                };

                // Submit request
                auto submit_result = ctx.scheduler->submit(request);
                if (submit_result.is_error()) {
                    sink.done();
                    return false;
                }

                // Wait for completion
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return done.load(); });

                sink.done();
                return true;
            });

    } else {
        // Non-streaming response
        std::promise<void> done_promise;
        auto done_future = done_promise.get_future();

        request->on_complete = [&done_promise](FinishReason /*reason*/) {
            done_promise.set_value();
        };

        request->on_error = [&done_promise, &res](const Error& error) {
            send_error(res, 500, error);
            done_promise.set_value();
        };

        // Submit and wait
        auto submit_result = ctx.scheduler->submit(request);
        if (submit_result.is_error()) {
            send_error(res, 503, submit_result.error());
            return;
        }

        done_future.wait();

        if (request->error) {
            return;  // Error already sent
        }

        // Build response
        ChatCompletionResponse response{
            .id = log_ctx.request_id,
            .created = current_timestamp(),
            .model = ctx.model_name,
            .choices = {{
                .index = 0,
                .message = {
                    .role = Role::Assistant,
                    .content = request->output_text
                },
                .finish_reason = request->finish_reason
            }},
            .usage = {
                .prompt_tokens = request->prompt_tokens(),
                .completion_tokens = request->completion_tokens(),
                .total_tokens = request->total_tokens()
            }
        };

        nlohmann::json j;
        to_json(j, response);
        res.set_content(j.dump(), "application/json");
    }

    log_info("api", "request_completed", {
        {"streaming", api_request.stream},
        {"prompt_tokens", request->prompt_tokens()},
        {"completion_tokens", request->completion_tokens()}
    }, log_ctx);
}

void list_models(
    const httplib::Request& /*req*/,
    httplib::Response& res,
    RouteContext& ctx) {

    nlohmann::json response = {
        {"object", "list"},
        {"data", nlohmann::json::array({
            {
                {"id", ctx.model_name},
                {"object", "model"},
                {"created", current_timestamp()},
                {"owned_by", "organization"}
            }
        })}
    };

    res.set_content(response.dump(), "application/json");
}

void health_check(
    const httplib::Request& /*req*/,
    httplib::Response& res,
    RouteContext& ctx) {

    bool scheduler_running = ctx.scheduler && ctx.scheduler->is_running();

    nlohmann::json response = {
        {"status", scheduler_running ? "healthy" : "unhealthy"},
        {"model", ctx.model_name},
        {"scheduler", scheduler_running ? "running" : "stopped"}
    };

    res.status = scheduler_running ? 200 : 503;
    res.set_content(response.dump(), "application/json");
}

void metrics(
    const httplib::Request& /*req*/,
    httplib::Response& res,
    RouteContext& /*ctx*/) {

    std::string prometheus_output = MetricsRegistry::instance().export_prometheus();
    res.set_content(prometheus_output, "text/plain; version=0.0.4");
}

}  // namespace handlers

}  // namespace qwen::api
