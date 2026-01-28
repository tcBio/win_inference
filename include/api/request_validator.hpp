#pragma once

#include "api/types.hpp"
#include "utils/result.hpp"

#include <string>
#include <vector>
#include <unordered_set>

namespace qwen::api {

/// Validation configuration
struct ValidationConfig {
    std::unordered_set<std::string> supported_models = {"qwen2.5-32b-instruct"};
    int32_t max_context_length = 70000;
    int32_t max_completion_tokens = 8192;
    int32_t default_max_tokens = 2048;
    double min_temperature = 0.0;
    double max_temperature = 2.0;
    double default_temperature = 1.0;
    double min_top_p = 0.0;
    double max_top_p = 1.0;
    double default_top_p = 1.0;
    size_t max_stop_sequences = 4;
    size_t max_stop_sequence_length = 64;
};

/// Request validator for chat completion requests
class RequestValidator {
public:
    explicit RequestValidator(ValidationConfig config = {});

    /// Validate a chat completion request
    /// Returns error if validation fails
    [[nodiscard]] Result<void> validate(const ChatCompletionRequest& request) const;

    /// Apply defaults to a request (mutating)
    void apply_defaults(ChatCompletionRequest& request) const;

    /// Get validation config
    [[nodiscard]] const ValidationConfig& config() const { return config_; }

private:
    [[nodiscard]] Result<void> validate_model(const std::string& model) const;
    [[nodiscard]] Result<void> validate_messages(const std::vector<Message>& messages) const;
    [[nodiscard]] Result<void> validate_temperature(double temp) const;
    [[nodiscard]] Result<void> validate_top_p(double top_p) const;
    [[nodiscard]] Result<void> validate_max_tokens(int32_t max_tokens) const;
    [[nodiscard]] Result<void> validate_stop(const std::vector<std::string>& stop) const;

    ValidationConfig config_;
};

}  // namespace qwen::api
