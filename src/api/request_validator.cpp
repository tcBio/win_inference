#include "api/request_validator.hpp"

namespace qwen::api {

RequestValidator::RequestValidator(ValidationConfig config)
    : config_(std::move(config)) {}

Result<void> RequestValidator::validate(const ChatCompletionRequest& request) const {
    // Validate model
    auto model_result = validate_model(request.model);
    if (model_result.is_error()) return model_result;

    // Validate messages
    auto messages_result = validate_messages(request.messages);
    if (messages_result.is_error()) return messages_result;

    // Validate temperature if provided
    if (request.temperature) {
        auto temp_result = validate_temperature(*request.temperature);
        if (temp_result.is_error()) return temp_result;
    }

    // Validate top_p if provided
    if (request.top_p) {
        auto top_p_result = validate_top_p(*request.top_p);
        if (top_p_result.is_error()) return top_p_result;
    }

    // Validate max_tokens if provided
    if (request.max_tokens) {
        auto max_tokens_result = validate_max_tokens(*request.max_tokens);
        if (max_tokens_result.is_error()) return max_tokens_result;
    }

    // Validate stop sequences if provided
    if (request.stop) {
        auto stop_result = validate_stop(*request.stop);
        if (stop_result.is_error()) return stop_result;
    }

    return Result<void>::success();
}

void RequestValidator::apply_defaults(ChatCompletionRequest& request) const {
    if (!request.temperature) {
        request.temperature = config_.default_temperature;
    }
    if (!request.top_p) {
        request.top_p = config_.default_top_p;
    }
    if (!request.max_tokens) {
        request.max_tokens = config_.default_max_tokens;
    }
}

Result<void> RequestValidator::validate_model(const std::string& model) const {
    if (model.empty()) {
        return Error::invalid_request("'model' is required");
    }

    // Normalize model name for comparison (lowercase)
    std::string normalized = model;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);

    bool found = false;
    for (const auto& supported : config_.supported_models) {
        std::string supported_lower = supported;
        std::transform(supported_lower.begin(), supported_lower.end(),
                      supported_lower.begin(), ::tolower);
        if (normalized == supported_lower) {
            found = true;
            break;
        }
    }

    if (!found) {
        return Error::invalid_request("Model '" + model + "' not supported");
    }

    return Result<void>::success();
}

Result<void> RequestValidator::validate_messages(const std::vector<Message>& messages) const {
    if (messages.empty()) {
        return Error::invalid_request("'messages' must contain at least one message");
    }

    bool has_user_message = false;
    for (const auto& msg : messages) {
        if (msg.role == Role::User) {
            has_user_message = true;
        }

        // Tool messages require tool_call_id
        if (msg.role == Role::Tool && !msg.tool_call_id) {
            return Error::invalid_request("Tool message requires 'tool_call_id'");
        }
    }

    if (!has_user_message) {
        return Error::invalid_request("'messages' must contain at least one user message");
    }

    return Result<void>::success();
}

Result<void> RequestValidator::validate_temperature(double temp) const {
    if (temp < config_.min_temperature || temp > config_.max_temperature) {
        return Error::invalid_request(
            "temperature must be between " + std::to_string(config_.min_temperature) +
            " and " + std::to_string(config_.max_temperature));
    }
    return Result<void>::success();
}

Result<void> RequestValidator::validate_top_p(double top_p) const {
    if (top_p < config_.min_top_p || top_p > config_.max_top_p) {
        return Error::invalid_request(
            "top_p must be between " + std::to_string(config_.min_top_p) +
            " and " + std::to_string(config_.max_top_p));
    }
    return Result<void>::success();
}

Result<void> RequestValidator::validate_max_tokens(int32_t max_tokens) const {
    if (max_tokens < 1) {
        return Error::invalid_request("max_tokens must be at least 1");
    }
    if (max_tokens > config_.max_completion_tokens) {
        return Error::invalid_request(
            "max_tokens cannot exceed " + std::to_string(config_.max_completion_tokens));
    }
    return Result<void>::success();
}

Result<void> RequestValidator::validate_stop(const std::vector<std::string>& stop) const {
    if (stop.size() > config_.max_stop_sequences) {
        return Error::invalid_request(
            "Maximum " + std::to_string(config_.max_stop_sequences) + " stop sequences allowed");
    }
    for (const auto& seq : stop) {
        if (seq.empty()) {
            return Error::invalid_request("Stop sequence cannot be empty");
        }
        if (seq.length() > config_.max_stop_sequence_length) {
            return Error::invalid_request(
                "Stop sequence exceeds maximum length of " +
                std::to_string(config_.max_stop_sequence_length));
        }
    }
    return Result<void>::success();
}

}  // namespace qwen::api
