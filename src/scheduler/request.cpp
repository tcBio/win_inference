#include "scheduler/request.hpp"

#include <random>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace qwen::scheduler {

std::string generate_request_id() {
    // Generate a unique request ID in format: chatcmpl-XXXXXXXXXXXX
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint64_t> dist;

    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    uint64_t random_part = dist(gen);

    std::ostringstream ss;
    ss << "chatcmpl-" << std::hex << std::setfill('0')
       << std::setw(8) << (timestamp & 0xFFFFFFFF)
       << std::setw(8) << (random_part & 0xFFFFFFFF);

    return ss.str();
}

RequestPtr make_request(const api::ChatCompletionRequest& api_request) {
    auto request = std::make_shared<Request>();

    request->id = generate_request_id();
    request->api_request = api_request;
    request->cancel_token = make_cancellation_token();
    request->timing.received_at = std::chrono::steady_clock::now();
    request->is_streaming = api_request.stream;

    // Copy sampling parameters
    if (api_request.temperature) {
        request->temperature = *api_request.temperature;
    }
    if (api_request.top_p) {
        request->top_p = *api_request.top_p;
    }
    if (api_request.max_tokens) {
        request->max_tokens = *api_request.max_tokens;
    }
    if (api_request.stop) {
        request->stop_sequences = *api_request.stop;
    }
    request->seed = api_request.seed;

    return request;
}

}  // namespace qwen::scheduler
