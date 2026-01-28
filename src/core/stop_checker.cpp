#include "core/stop_checker.hpp"

#include <algorithm>

namespace qwen::core {

StopChecker::StopChecker(std::shared_ptr<tokenizer::ITokenizer> tokenizer)
    : tokenizer_(std::move(tokenizer)) {}

StopCheckResult StopChecker::check(
    const std::vector<int32_t>& generated_tokens,
    const std::string& generated_text,
    const StopConfig& config) const {

    StopCheckResult result;

    // Check max_tokens limit
    if (config.max_tokens > 0 &&
        static_cast<int32_t>(generated_tokens.size()) >= config.max_tokens) {
        result.should_stop = true;
        result.reason = api::FinishReason::Length;
        return result;
    }

    // Check for stop tokens
    if (!generated_tokens.empty()) {
        int32_t last_token = generated_tokens.back();

        // Check EOS token
        if (tokenizer_ && last_token == tokenizer_->special_tokens().eos_token_id) {
            result.should_stop = true;
            result.reason = api::FinishReason::Stop;
            result.matched_stop_token = last_token;
            return result;
        }

        // Check im_end token (Qwen specific)
        if (tokenizer_ && last_token == tokenizer_->special_tokens().im_end_id) {
            result.should_stop = true;
            result.reason = api::FinishReason::Stop;
            result.matched_stop_token = last_token;
            return result;
        }

        // Check custom stop token IDs
        for (int32_t stop_token : config.stop_token_ids) {
            if (last_token == stop_token) {
                result.should_stop = true;
                result.reason = api::FinishReason::Stop;
                result.matched_stop_token = stop_token;
                return result;
            }
        }
    }

    // Check for stop sequences in text
    if (!config.stop_sequences.empty() && !generated_text.empty()) {
        auto matched = find_stop_sequence(generated_text, config.stop_sequences);
        if (matched.has_value()) {
            result.should_stop = true;
            result.reason = api::FinishReason::Stop;
            result.matched_stop_sequence = *matched;
            return result;
        }
    }

    return result;
}

bool StopChecker::is_stop_token(int32_t token_id, const StopConfig& config) const {
    // Check EOS
    if (tokenizer_ && token_id == tokenizer_->special_tokens().eos_token_id) {
        return true;
    }

    // Check im_end
    if (tokenizer_ && token_id == tokenizer_->special_tokens().im_end_id) {
        return true;
    }

    // Check custom stop tokens
    return std::find(config.stop_token_ids.begin(),
                    config.stop_token_ids.end(),
                    token_id) != config.stop_token_ids.end();
}

std::optional<std::string> StopChecker::find_stop_sequence(
    const std::string& text,
    const std::vector<std::string>& stop_sequences) const {

    for (const auto& seq : stop_sequences) {
        if (seq.empty()) continue;

        // Check if text ends with this sequence
        if (text.length() >= seq.length()) {
            if (text.compare(text.length() - seq.length(), seq.length(), seq) == 0) {
                return seq;
            }
        }
    }

    return std::nullopt;
}

std::string StopChecker::trim_stop_sequence(
    const std::string& text,
    const std::string& stop_sequence) const {

    if (stop_sequence.empty()) return text;

    if (text.length() >= stop_sequence.length() &&
        text.compare(text.length() - stop_sequence.length(),
                    stop_sequence.length(), stop_sequence) == 0) {
        return text.substr(0, text.length() - stop_sequence.length());
    }

    return text;
}

}  // namespace qwen::core
