#include <gtest/gtest.h>
#include "tokenizer/tokenizer.hpp"
#include "tokenizer/prompt_builder.hpp"
#include "core/stop_checker.hpp"

namespace qwen {

/// Golden tests verify that outputs match expected reference values.
/// These tests are critical for accuracy and should be updated carefully.

class GoldenTest : public ::testing::Test {
protected:
    void SetUp() override {
        tokenizer_ = tokenizer::create_qwen_tokenizer();
        tokenizer::TokenizerConfig config;
        tokenizer_->initialize(config);

        prompt_builder_ = std::make_unique<tokenizer::PromptBuilder>(
            tokenizer_,
            tokenizer::PromptBuilderConfig{}
        );

        stop_checker_ = std::make_unique<core::StopChecker>(tokenizer_);
    }

    std::shared_ptr<tokenizer::ITokenizer> tokenizer_;
    std::unique_ptr<tokenizer::PromptBuilder> prompt_builder_;
    std::unique_ptr<core::StopChecker> stop_checker_;
};

/// Test vector for chat template formatting
struct ChatTemplateTestVector {
    std::vector<api::Message> messages;
    std::string expected_output;
    std::string description;
};

/// Golden test vectors for Qwen2.5 chat template
const std::vector<ChatTemplateTestVector> CHAT_TEMPLATE_VECTORS = {
    {
        {{api::Role::User, "Hello"}},
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "Single user message"
    },
    {
        {{api::Role::System, "You are helpful."}, {api::Role::User, "Hi"}},
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        "System + user message"
    },
    {
        {
            {api::Role::User, "What is 2+2?"},
            {api::Role::Assistant, "4"},
            {api::Role::User, "And 3+3?"}
        },
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n<|im_start|>user\nAnd 3+3?<|im_end|>\n<|im_start|>assistant\n",
        "Multi-turn conversation"
    }
};

TEST_F(GoldenTest, ChatTemplateFormatting) {
    for (const auto& vector : CHAT_TEMPLATE_VECTORS) {
        auto result = prompt_builder_->build(vector.messages);
        ASSERT_TRUE(result.ok()) << "Failed for: " << vector.description;

        EXPECT_EQ(result.value().formatted_text, vector.expected_output)
            << "Mismatch for: " << vector.description
            << "\nExpected:\n" << vector.expected_output
            << "\nGot:\n" << result.value().formatted_text;
    }
}

/// Test vector for stop sequence detection
struct StopSequenceTestVector {
    std::string generated_text;
    std::vector<std::string> stop_sequences;
    bool should_stop;
    std::string matched_sequence;
    std::string description;
};

const std::vector<StopSequenceTestVector> STOP_SEQUENCE_VECTORS = {
    {
        "Hello world\n\n",
        {"\n\n"},
        true,
        "\n\n",
        "Double newline stop"
    },
    {
        "The answer is 42.",
        {"."},
        true,
        ".",
        "Period stop"
    },
    {
        "Hello world",
        {"\n\n"},
        false,
        "",
        "No match"
    },
    {
        "User: Hello\nAssistant: Hi",
        {"User:", "Human:"},
        false,
        "",
        "Stop sequence in middle, not at end"
    },
    {
        "Some text\nUser:",
        {"User:", "Human:"},
        true,
        "User:",
        "Stop sequence at end"
    }
};

TEST_F(GoldenTest, StopSequenceDetection) {
    for (const auto& vector : STOP_SEQUENCE_VECTORS) {
        auto matched = stop_checker_->find_stop_sequence(
            vector.generated_text,
            vector.stop_sequences
        );

        if (vector.should_stop) {
            ASSERT_TRUE(matched.has_value())
                << "Expected stop for: " << vector.description;
            EXPECT_EQ(*matched, vector.matched_sequence)
                << "Wrong sequence for: " << vector.description;
        } else {
            EXPECT_FALSE(matched.has_value())
                << "Unexpected stop for: " << vector.description;
        }
    }
}

/// Test vector for stop sequence trimming
struct TrimTestVector {
    std::string text;
    std::string stop_sequence;
    std::string expected;
    std::string description;
};

const std::vector<TrimTestVector> TRIM_VECTORS = {
    {"Hello world.", ".", "Hello world", "Trim period"},
    {"Hello world\n\n", "\n\n", "Hello world", "Trim double newline"},
    {"Hello world", ".", "Hello world", "No match, no change"},
    {"Hello.world", ".", "Hello.world", "Match in middle, no change"}
};

TEST_F(GoldenTest, StopSequenceTrimming) {
    for (const auto& vector : TRIM_VECTORS) {
        auto result = stop_checker_->trim_stop_sequence(
            vector.text,
            vector.stop_sequence
        );

        EXPECT_EQ(result, vector.expected)
            << "Mismatch for: " << vector.description;
    }
}

/// Test vector for max_tokens enforcement
struct MaxTokensTestVector {
    int generated_count;
    int max_tokens;
    bool should_stop;
    std::string description;
};

const std::vector<MaxTokensTestVector> MAX_TOKENS_VECTORS = {
    {10, 10, true, "Exactly at limit"},
    {5, 10, false, "Under limit"},
    {15, 10, true, "Over limit"},
    {0, 10, false, "No tokens yet"},
    {5, 0, false, "Unlimited (max_tokens=0)"}
};

TEST_F(GoldenTest, MaxTokensEnforcement) {
    for (const auto& vector : MAX_TOKENS_VECTORS) {
        std::vector<int32_t> tokens(vector.generated_count, 1);

        core::StopConfig config;
        config.max_tokens = vector.max_tokens;

        auto result = stop_checker_->check(tokens, "", config);

        EXPECT_EQ(result.should_stop, vector.should_stop)
            << "Mismatch for: " << vector.description;

        if (result.should_stop && vector.max_tokens > 0) {
            EXPECT_EQ(result.reason, api::FinishReason::Length)
                << "Wrong reason for: " << vector.description;
        }
    }
}

/// Streaming vs non-streaming equivalence test
TEST_F(GoldenTest, StreamingEquivalence) {
    // The final concatenated output from streaming should equal non-streaming output
    // This is tested at the integration level, but we verify the building blocks here

    std::vector<api::Message> messages = {
        {api::Role::User, "Test message"}
    };

    auto result = prompt_builder_->build(messages);
    ASSERT_TRUE(result.ok());

    // Same prompt should produce same tokens
    auto result2 = prompt_builder_->build(messages);
    ASSERT_TRUE(result2.ok());

    EXPECT_EQ(result.value().token_ids, result2.value().token_ids);
    EXPECT_EQ(result.value().formatted_text, result2.value().formatted_text);
}

/// Determinism test
TEST_F(GoldenTest, Determinism) {
    std::vector<api::Message> messages = {
        {api::Role::System, "You are a helpful assistant."},
        {api::Role::User, "What is the capital of France?"}
    };

    // Run 100 times, should get identical output each time
    std::string first_output;
    std::vector<int32_t> first_tokens;

    for (int i = 0; i < 100; ++i) {
        auto result = prompt_builder_->build(messages);
        ASSERT_TRUE(result.ok());

        if (i == 0) {
            first_output = result.value().formatted_text;
            first_tokens = result.value().token_ids;
        } else {
            EXPECT_EQ(result.value().formatted_text, first_output)
                << "Text mismatch on iteration " << i;
            EXPECT_EQ(result.value().token_ids, first_tokens)
                << "Token mismatch on iteration " << i;
        }
    }
}

/// Test for double-templating detection
TEST_F(GoldenTest, DoubleTemplatingDetection) {
    // Pre-formatted prompt
    std::string pre_formatted = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";

    EXPECT_TRUE(prompt_builder_->is_already_formatted(pre_formatted));

    // Regular text
    EXPECT_FALSE(prompt_builder_->is_already_formatted("Hello world"));
    EXPECT_FALSE(prompt_builder_->is_already_formatted("What is 2+2?"));
}

}  // namespace qwen
