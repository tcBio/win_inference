#pragma once

#include "api/types.hpp"
#include "api/server.hpp"
#include "api/sse_streamer.hpp"
#include "scheduler/scheduler.hpp"
#include "tokenizer/prompt_builder.hpp"

#include <httplib.h>
#include <functional>

namespace qwen::api {

/// Route handler context
struct RouteContext {
    std::shared_ptr<scheduler::Scheduler> scheduler;
    std::shared_ptr<tokenizer::PromptBuilder> prompt_builder;
    RequestValidator* validator;
    std::string model_name;
};

/// Register all API routes
void register_routes(httplib::Server& server, RouteContext& ctx);

/// Individual route handlers
namespace handlers {

/// POST /v1/chat/completions
void chat_completions(
    const httplib::Request& req,
    httplib::Response& res,
    RouteContext& ctx);

/// GET /v1/models
void list_models(
    const httplib::Request& req,
    httplib::Response& res,
    RouteContext& ctx);

/// GET /health
void health_check(
    const httplib::Request& req,
    httplib::Response& res,
    RouteContext& ctx);

/// GET /metrics
void metrics(
    const httplib::Request& req,
    httplib::Response& res,
    RouteContext& ctx);

}  // namespace handlers

/// Error response helpers
void send_error(
    httplib::Response& res,
    int status,
    const std::string& type,
    const std::string& code,
    const std::string& message);

void send_error(httplib::Response& res, int status, const Error& error);

/// Generate unique completion ID
std::string generate_completion_id();

/// Get current Unix timestamp
int64_t current_timestamp();

}  // namespace qwen::api
