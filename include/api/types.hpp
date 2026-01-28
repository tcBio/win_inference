#pragma once

#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <nlohmann/json.hpp>

namespace qwen::api {

/// Chat message role
enum class Role {
    System,
    User,
    Assistant,
    Tool
};

inline std::string role_to_string(Role role) {
    switch (role) {
        case Role::System: return "system";
        case Role::User: return "user";
        case Role::Assistant: return "assistant";
        case Role::Tool: return "tool";
    }
    return "user";
}

inline Role string_to_role(const std::string& s) {
    if (s == "system") return Role::System;
    if (s == "assistant") return Role::Assistant;
    if (s == "tool") return Role::Tool;
    return Role::User;
}

/// Chat message
struct Message {
    Role role;
    std::string content;
    std::optional<std::string> name;  // For tool messages
    std::optional<nlohmann::json> tool_calls;  // Passthrough
    std::optional<std::string> tool_call_id;   // For tool responses
};

/// Tool definition (passthrough)
struct Tool {
    std::string type;
    nlohmann::json function;
};

/// Chat completion request
struct ChatCompletionRequest {
    std::string model;
    std::vector<Message> messages;
    std::optional<double> temperature;      // [0, 2], default 1.0
    std::optional<double> top_p;            // [0, 1], default 1.0
    std::optional<int32_t> max_tokens;      // Max completion tokens
    std::optional<std::vector<std::string>> stop;  // Stop sequences
    bool stream = false;
    std::optional<int64_t> seed;            // For deterministic sampling
    std::optional<std::vector<Tool>> tools; // Passthrough
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<std::string> user;        // End-user identifier
};

/// Finish reason for generation
enum class FinishReason {
    Stop,       // Natural stop or stop sequence
    Length,     // Hit max_tokens
    ToolCalls,  // Model wants to call tools
    ContentFilter,
    None        // Still generating
};

inline std::string finish_reason_to_string(FinishReason reason) {
    switch (reason) {
        case FinishReason::Stop: return "stop";
        case FinishReason::Length: return "length";
        case FinishReason::ToolCalls: return "tool_calls";
        case FinishReason::ContentFilter: return "content_filter";
        case FinishReason::None: return "";
    }
    return "";
}

/// Usage statistics
struct Usage {
    int32_t prompt_tokens = 0;
    int32_t completion_tokens = 0;
    int32_t total_tokens = 0;
};

/// Non-streaming response choice
struct Choice {
    int32_t index = 0;
    Message message;
    FinishReason finish_reason = FinishReason::None;
};

/// Non-streaming chat completion response
struct ChatCompletionResponse {
    std::string id;
    std::string object = "chat.completion";
    int64_t created = 0;
    std::string model;
    std::vector<Choice> choices;
    Usage usage;
};

/// Streaming delta content
struct Delta {
    std::optional<Role> role;
    std::optional<std::string> content;
    std::optional<nlohmann::json> tool_calls;
};

/// Streaming choice
struct StreamChoice {
    int32_t index = 0;
    Delta delta;
    std::optional<FinishReason> finish_reason;
};

/// Streaming chunk response
struct ChatCompletionChunk {
    std::string id;
    std::string object = "chat.completion.chunk";
    int64_t created = 0;
    std::string model;
    std::vector<StreamChoice> choices;
};

/// API error response
struct ErrorResponse {
    std::string type;
    std::string code;
    std::string message;
    std::optional<std::string> param;
};

/// Model info for /v1/models
struct ModelInfo {
    std::string id;
    std::string object = "model";
    int64_t created = 0;
    std::string owned_by;
};

/// JSON conversion functions
void to_json(nlohmann::json& j, const Message& m);
void from_json(const nlohmann::json& j, Message& m);

void to_json(nlohmann::json& j, const ChatCompletionRequest& r);
void from_json(const nlohmann::json& j, ChatCompletionRequest& r);

void to_json(nlohmann::json& j, const Usage& u);
void to_json(nlohmann::json& j, const Choice& c);
void to_json(nlohmann::json& j, const ChatCompletionResponse& r);

void to_json(nlohmann::json& j, const Delta& d);
void to_json(nlohmann::json& j, const StreamChoice& c);
void to_json(nlohmann::json& j, const ChatCompletionChunk& c);

void to_json(nlohmann::json& j, const ErrorResponse& e);
void to_json(nlohmann::json& j, const ModelInfo& m);

}  // namespace qwen::api
