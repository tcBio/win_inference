#pragma once

#include "tokenizer/tokenizer.hpp"
#include "api/types.hpp"

#include <string>
#include <vector>
#include <memory>

namespace qwen::core {

/// Stop condition configuration
struct StopConfig {
    std::vector<std::string> stop_sequences;  // Text sequences to stop on
    std::vector<int32_t> stop_token_ids;      // Token IDs to stop on
    int32_t max_tokens = 0;                   // Max tokens to generate (0 = unlimited)
    bool include_stop_str_in_output = false;  // Include the stop sequence in output
};

/// Result of stop checking
struct StopCheckResult {
    bool should_stop = false;
    api::FinishReason reason = api::FinishReason::None;
    std::string matched_stop_sequence;        // If stopped by sequence
    int32_t matched_stop_token = -1;          // If stopped by token
};

/// Stop condition checker
/// Handles EOS tokens, stop sequences, and max_tokens
class StopChecker {
public:
    explicit StopChecker(std::shared_ptr<tokenizer::ITokenizer> tokenizer);

    /// Check if generation should stop
    /// @param generated_tokens All tokens generated so far
    /// @param generated_text The decoded text so far
    /// @param config Stop configuration
    [[nodiscard]] StopCheckResult check(
        const std::vector<int32_t>& generated_tokens,
        const std::string& generated_text,
        const StopConfig& config) const;

    /// Check if a single token is a stop token
    [[nodiscard]] bool is_stop_token(
        int32_t token_id,
        const StopConfig& config) const;

    /// Check if text ends with any stop sequence
    [[nodiscard]] std::optional<std::string> find_stop_sequence(
        const std::string& text,
        const std::vector<std::string>& stop_sequences) const;

    /// Trim stop sequence from text if present
    [[nodiscard]] std::string trim_stop_sequence(
        const std::string& text,
        const std::string& stop_sequence) const;

private:
    std::shared_ptr<tokenizer::ITokenizer> tokenizer_;
};

}  // namespace qwen::core
