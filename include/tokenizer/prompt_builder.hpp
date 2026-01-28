#pragma once

#include "api/types.hpp"
#include "tokenizer/tokenizer.hpp"
#include "utils/result.hpp"

#include <string>
#include <vector>

namespace qwen::tokenizer {

/// Qwen2.5 chat template format:
/// <|im_start|>system
/// {system_message}<|im_end|>
/// <|im_start|>user
/// {user_message}<|im_end|>
/// <|im_start|>assistant
/// {assistant_message}<|im_end|>
///
/// For generation, end with:
/// <|im_start|>assistant
///
/// This builder ensures exact format compliance for accuracy.

/// Configuration for prompt building
struct PromptBuilderConfig {
    bool add_generation_prompt = true;   // Add assistant prefix for generation
    std::string default_system_message;  // Default system prompt if none provided
    bool allow_raw_prompt = false;       // Allow passing raw prompt without templating
    size_t max_prompt_length = 0;        // Max chars (0 = unlimited)
};

/// Result of prompt building
struct BuiltPrompt {
    std::string formatted_text;          // The formatted prompt string
    TokenIds token_ids;                  // Tokenized prompt
    bool has_system_message = false;
    bool has_generation_prompt = false;
    size_t num_messages = 0;
};

/// Prompt builder for Qwen2.5 chat format
class PromptBuilder {
public:
    explicit PromptBuilder(
        std::shared_ptr<ITokenizer> tokenizer,
        PromptBuilderConfig config = {}
    );

    /// Build prompt from OpenAI-style messages
    [[nodiscard]] Result<BuiltPrompt> build(
        const std::vector<api::Message>& messages) const;

    /// Build prompt with explicit control over generation prompt
    [[nodiscard]] Result<BuiltPrompt> build(
        const std::vector<api::Message>& messages,
        bool add_generation_prompt) const;

    /// Format a single message (for debugging/testing)
    [[nodiscard]] std::string format_message(const api::Message& message) const;

    /// Check if input appears to be already formatted
    /// (to avoid double-templating)
    [[nodiscard]] bool is_already_formatted(const std::string& text) const;

    /// Get the tokenizer
    [[nodiscard]] std::shared_ptr<ITokenizer> tokenizer() const { return tokenizer_; }

    /// Get configuration
    [[nodiscard]] const PromptBuilderConfig& config() const { return config_; }

private:
    std::string build_formatted_prompt(
        const std::vector<api::Message>& messages,
        bool add_generation_prompt) const;

    std::shared_ptr<ITokenizer> tokenizer_;
    PromptBuilderConfig config_;
};

/// Thread-safety notes:
/// - PromptBuilder is thread-safe for concurrent build() calls
/// - The underlying tokenizer must also be thread-safe
/// - No internal mutable state beyond config

}  // namespace qwen::tokenizer
