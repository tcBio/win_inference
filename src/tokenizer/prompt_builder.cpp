#include "tokenizer/prompt_builder.hpp"
#include "utils/logger.hpp"

namespace qwen::tokenizer {

PromptBuilder::PromptBuilder(
    std::shared_ptr<ITokenizer> tokenizer,
    PromptBuilderConfig config
)
    : tokenizer_(std::move(tokenizer))
    , config_(std::move(config)) {}

Result<BuiltPrompt> PromptBuilder::build(
    const std::vector<api::Message>& messages) const {
    return build(messages, config_.add_generation_prompt);
}

Result<BuiltPrompt> PromptBuilder::build(
    const std::vector<api::Message>& messages,
    bool add_generation_prompt) const {

    if (!tokenizer_) {
        return Error::internal("Tokenizer not initialized");
    }

    if (messages.empty()) {
        return Error::invalid_request("Messages cannot be empty");
    }

    // Build the formatted prompt string
    std::string formatted = build_formatted_prompt(messages, add_generation_prompt);

    // Check for max length if configured
    if (config_.max_prompt_length > 0 && formatted.length() > config_.max_prompt_length) {
        return Error::invalid_request(
            "Prompt exceeds maximum length of " +
            std::to_string(config_.max_prompt_length) + " characters");
    }

    // Tokenize
    auto token_result = tokenizer_->encode(formatted, false);
    if (token_result.is_error()) {
        return token_result.error();
    }

    // Build result
    BuiltPrompt result;
    result.formatted_text = std::move(formatted);
    result.token_ids = std::move(token_result.value());
    result.num_messages = messages.size();
    result.has_generation_prompt = add_generation_prompt;

    // Check if system message was included
    for (const auto& msg : messages) {
        if (msg.role == api::Role::System) {
            result.has_system_message = true;
            break;
        }
    }

    log_debug("prompt_builder", "built_prompt", {
        {"num_messages", result.num_messages},
        {"num_tokens", result.token_ids.size()},
        {"has_system", result.has_system_message},
        {"has_gen_prompt", result.has_generation_prompt}
    });

    return result;
}

std::string PromptBuilder::format_message(const api::Message& message) const {
    std::string role_str = api::role_to_string(message.role);

    // Qwen2.5 chat template format:
    // <|im_start|>{role}
    // {content}<|im_end|>
    std::string result = "<|im_start|>" + role_str + "\n";
    result += message.content;
    result += "<|im_end|>\n";

    return result;
}

bool PromptBuilder::is_already_formatted(const std::string& text) const {
    // Check for Qwen chat template markers
    return text.find("<|im_start|>") != std::string::npos ||
           text.find("<|im_end|>") != std::string::npos;
}

std::string PromptBuilder::build_formatted_prompt(
    const std::vector<api::Message>& messages,
    bool add_generation_prompt) const {

    std::string prompt;

    // Check if first message content appears to be already formatted
    if (!messages.empty() && config_.allow_raw_prompt) {
        const auto& first = messages.front();
        if (is_already_formatted(first.content)) {
            // Pass through raw content
            log_warn("prompt_builder", "raw_prompt_passthrough", {
                {"message", "Detected pre-formatted prompt, passing through"}
            });
            return first.content;
        }
    }

    // Add default system message if configured and no system message provided
    bool has_system = false;
    for (const auto& msg : messages) {
        if (msg.role == api::Role::System) {
            has_system = true;
            break;
        }
    }

    if (!has_system && !config_.default_system_message.empty()) {
        prompt += "<|im_start|>system\n";
        prompt += config_.default_system_message;
        prompt += "<|im_end|>\n";
    }

    // Format each message
    for (const auto& message : messages) {
        prompt += format_message(message);
    }

    // Add generation prompt (assistant start) for inference
    if (add_generation_prompt) {
        prompt += "<|im_start|>assistant\n";
    }

    return prompt;
}

}  // namespace qwen::tokenizer
