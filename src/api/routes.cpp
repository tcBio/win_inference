#include "api/routes.hpp"
#include "utils/logger.hpp"
#include "utils/metrics.hpp"
#include "utils/cancellation.hpp"

#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>

namespace qwen::api {

namespace {

/// Add CORS headers to response
void add_cors_headers(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.set_header("Access-Control-Max-Age", "86400");  // 24 hours
}

}  // namespace

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
    const std::string& message,
    const std::optional<std::string>& param) {

    ErrorResponse err{
        .type = type,
        .code = code,
        .message = message,
        .param = param
    };

    nlohmann::json j;
    to_json(j, err);

    res.status = status;
    res.set_content(j.dump(), "application/json");
}

void send_error(httplib::Response& res, int status, const Error& error,
    const std::optional<std::string>& param) {
    send_error(res, status, "error", error.code, error.message, param);
}

void register_routes(httplib::Server& server, RouteContext& ctx) {
    // Store context in a shared_ptr for lambda capture
    auto ctx_ptr = std::make_shared<RouteContext>(ctx);

    // CORS preflight handlers
    server.Options("/v1/chat/completions",
        [](const httplib::Request& /*req*/, httplib::Response& res) {
            add_cors_headers(res);
            res.status = 204;
        });
    server.Options("/v1/models",
        [](const httplib::Request& /*req*/, httplib::Response& res) {
            add_cors_headers(res);
            res.status = 204;
        });

    // POST /v1/chat/completions
    server.Post("/v1/chat/completions",
        [ctx_ptr](const httplib::Request& req, httplib::Response& res) {
            add_cors_headers(res);
            handlers::chat_completions(req, res, *ctx_ptr);
        });

    // GET /v1/models
    server.Get("/v1/models",
        [ctx_ptr](const httplib::Request& req, httplib::Response& res) {
            add_cors_headers(res);
            handlers::list_models(req, res, *ctx_ptr);
        });

    // GET /health
    server.Get("/health",
        [ctx_ptr](const httplib::Request& req, httplib::Response& res) {
            add_cors_headers(res);
            handlers::health_check(req, res, *ctx_ptr);
        });

    // GET /metrics
    server.Get("/metrics",
        [ctx_ptr](const httplib::Request& req, httplib::Response& res) {
            add_cors_headers(res);
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

    // Validate prompt length against model limits
    const auto& validation_config = ctx.validator->config();
    int32_t prompt_tokens = static_cast<int32_t>(built_prompt.token_ids.size());
    int32_t max_tokens = api_request.max_tokens.value_or(validation_config.default_max_tokens);
    int32_t total_required = prompt_tokens + max_tokens;

    if (prompt_tokens > validation_config.max_context_length) {
        log_warn("api", "prompt_too_long", {
            {"prompt_tokens", prompt_tokens},
            {"max_context", validation_config.max_context_length}
        }, log_ctx);
        send_error(res, 400, "invalid_request_error", "context_length_exceeded",
            "Prompt contains " + std::to_string(prompt_tokens) +
            " tokens which exceeds the maximum context length of " +
            std::to_string(validation_config.max_context_length),
            "messages");
        return;
    }

    if (total_required > validation_config.max_context_length) {
        // Adjust max_tokens to fit within context limit
        int32_t available_for_completion = validation_config.max_context_length - prompt_tokens;
        if (available_for_completion < 1) {
            send_error(res, 400, "invalid_request_error", "context_length_exceeded",
                "Prompt uses all available context. No room for completion.",
                "messages");
            return;
        }
        log_info("api", "max_tokens_adjusted", {
            {"original", max_tokens},
            {"adjusted", available_for_completion},
            {"prompt_tokens", prompt_tokens}
        }, log_ctx);
        api_request.max_tokens = available_for_completion;
    }

    // Create scheduler request with cancellation token
    auto request = scheduler::make_request(api_request);
    request->trace_id = log_ctx.request_id;
    request->input_tokens = built_prompt.token_ids;
    request->formatted_prompt = built_prompt.formatted_text;
    request->cancel_token = make_cancellation_token();

    if (api_request.stream) {
        // Streaming response
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        std::string completion_id = log_ctx.request_id;
        std::string model_name = ctx.model_name;

        // Create thread-safe token queue for scheduler -> HTTP thread communication
        request->token_queue = std::make_shared<scheduler::TokenQueue>();
        request->is_streaming = true;

        // Set up streaming using chunked content provider
        // The HTTP thread will poll the token queue and write to sink
        res.set_chunked_content_provider(
            "text/event-stream",
            [request, completion_id, model_name, &ctx](
                size_t /*offset*/, httplib::DataSink& sink) {

                // Helper to write and detect client disconnect
                auto safe_write = [&sink, &request](const std::string& data) -> bool {
                    if (!sink.write(data.data(), data.size())) {
                        // Client disconnected - cancel the request
                        if (request->cancel_token) {
                            request->cancel_token->cancel();
                        }
                        return false;
                    }
                    return true;
                };

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
                if (!safe_write("data: " + j.dump() + "\n\n")) {
                    sink.done();
                    return false;
                }

                // Submit request to scheduler
                auto submit_result = ctx.scheduler->submit(request);
                if (submit_result.is_error()) {
                    sink.done();
                    return false;
                }

                // Poll token queue from HTTP thread (thread-safe)
                // All sink writes happen on this thread, avoiding race conditions
                bool done = false;
                bool client_disconnected = false;
                auto last_token_time = std::chrono::steady_clock::now();
                const auto idle_timeout = std::chrono::seconds(ctx.stream_idle_timeout_sec);

                while (!done && !client_disconnected) {
                    auto event = request->token_queue->pop(std::chrono::milliseconds(100));
                    if (!event) {
                        // Timeout - check if request was cancelled or errored
                        if (request->is_terminal()) {
                            done = true;
                            continue;
                        }

                        // Check for idle timeout (no tokens received)
                        if (ctx.stream_idle_timeout_sec > 0) {
                            auto idle_duration = std::chrono::steady_clock::now() - last_token_time;
                            if (idle_duration > idle_timeout) {
                                log_warn("api", "stream_idle_timeout", {
                                    {"request_id", request->id},
                                    {"idle_sec", std::chrono::duration_cast<std::chrono::seconds>(idle_duration).count()}
                                });
                                // Cancel the request due to timeout
                                if (request->cancel_token) {
                                    request->cancel_token->cancel();
                                }
                                // Send error event before closing stream (OpenAI-compatible)
                                nlohmann::json error_event = {
                                    {"error", {
                                        {"message", "Stream idle timeout - no tokens received"},
                                        {"type", "timeout"},
                                        {"code", "stream_idle_timeout"}
                                    }}
                                };
                                safe_write("data: " + error_event.dump() + "\n\n");
                                safe_write("data: [DONE]\n\n");
                                done = true;
                            }
                        }
                        continue;
                    }

                    // Reset idle timer on receiving an event
                    last_token_time = std::chrono::steady_clock::now();

                    switch (event->type) {
                        case scheduler::TokenEvent::Type::Token: {
                            if (event->text.empty()) break;

                            ChatCompletionChunk chunk{
                                .id = completion_id,
                                .created = current_timestamp(),
                                .model = model_name,
                                .choices = {{
                                    .index = 0,
                                    .delta = {.content = event->text}
                                }}
                            };
                            nlohmann::json token_json;
                            to_json(token_json, chunk);
                            if (!safe_write("data: " + token_json.dump() + "\n\n")) {
                                client_disconnected = true;
                            }
                            break;
                        }

                        case scheduler::TokenEvent::Type::Complete: {
                            ChatCompletionChunk chunk{
                                .id = completion_id,
                                .created = current_timestamp(),
                                .model = model_name,
                                .choices = {{
                                    .index = 0,
                                    .delta = {},
                                    .finish_reason = event->finish_reason
                                }}
                            };
                            nlohmann::json complete_json;
                            to_json(complete_json, chunk);
                            safe_write("data: " + complete_json.dump() + "\n\n");
                            safe_write("data: [DONE]\n\n");
                            done = true;
                            break;
                        }

                        case scheduler::TokenEvent::Type::Error: {
                            // Send error event before terminating (OpenAI-compatible)
                            std::string error_msg = "Internal error";
                            std::string error_code = "internal_error";
                            if (event->error) {
                                error_msg = event->error->message;
                                error_code = event->error->code;
                            }
                            nlohmann::json error_event = {
                                {"error", {
                                    {"message", error_msg},
                                    {"type", "error"},
                                    {"code", error_code}
                                }}
                            };
                            safe_write("data: " + error_event.dump() + "\n\n");
                            safe_write("data: [DONE]\n\n");
                            done = true;
                            break;
                        }
                    }
                }

                // Log if client disconnected before completion
                if (client_disconnected) {
                    log_info("api", "client_disconnected", {
                        {"request_id", request->id}
                    });
                }

                sink.done();
                return !client_disconnected;
            });

    } else {
        // Non-streaming response - use TokenQueue for thread-safe completion signaling
        request->token_queue = std::make_shared<scheduler::TokenQueue>();
        request->is_streaming = false;

        // Submit request
        auto submit_result = ctx.scheduler->submit(request);
        if (submit_result.is_error()) {
            send_error(res, 503, submit_result.error());
            return;
        }

        // Wait for completion via token queue (thread-safe)
        // The scheduler will push a Complete or Error event when done
        bool got_result = false;
        std::optional<Error> error_result;
        auto start_time = std::chrono::steady_clock::now();
        const auto response_timeout = std::chrono::seconds(ctx.response_timeout_sec);

        while (!got_result) {
            auto event = request->token_queue->pop(std::chrono::milliseconds(100));
            if (!event) {
                // Timeout - check if request was cancelled externally
                if (request->is_terminal()) {
                    got_result = true;
                    continue;
                }

                // Check for response timeout
                if (ctx.response_timeout_sec > 0) {
                    auto elapsed = std::chrono::steady_clock::now() - start_time;
                    if (elapsed > response_timeout) {
                        log_warn("api", "response_timeout", {
                            {"request_id", request->id},
                            {"elapsed_sec", std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()}
                        });
                        // Cancel the request due to timeout
                        if (request->cancel_token) {
                            request->cancel_token->cancel();
                        }
                        error_result = Error::timeout("Request timed out");
                        got_result = true;
                    }
                }
                continue;
            }

            switch (event->type) {
                case scheduler::TokenEvent::Type::Token:
                    // Tokens are accumulated in request->output_text by scheduler
                    break;

                case scheduler::TokenEvent::Type::Complete:
                    got_result = true;
                    break;

                case scheduler::TokenEvent::Type::Error:
                    got_result = true;
                    error_result = event->error;
                    break;
            }
        }

        // Handle error case
        if (error_result || request->error) {
            const auto& err = error_result ? *error_result : *request->error;
            int status = 500;
            if (err.code == "timeout") {
                status = 504;  // Gateway Timeout
            } else if (err.code == "resource_exhausted") {
                status = 503;  // Service Unavailable
            }
            send_error(res, status, err);
            return;
        }

        // Build response on HTTP thread (thread-safe)
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
