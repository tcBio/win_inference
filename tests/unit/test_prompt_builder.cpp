#include <gtest/gtest.h>
#include "tokenizer/prompt_builder.hpp"
#include "tokenizer/tokenizer.hpp"

namespace qwen::tokenizer {

class PromptBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        tokenizer_ = create_qwen_tokenizer();
        TokenizerConfig config;
        tokenizer_->initialize(config);

        builder_ = std::make_unique<PromptBuilder>(
            tokenizer_,
            PromptBuilderConfig{}
        );
    }

    std::shared_ptr<ITokenizer> tokenizer_;
    std::unique_ptr<PromptBuilder> builder_;
};

TEST_F(PromptBuilderTest, BuildEmptyMessagesFails) {
    std::vector<api::Message> messages;
    auto result = builder_->build(messages);
    EXPECT_TRUE(result.is_error());
}

TEST_F(PromptBuilderTest, BuildSingleUserMessage) {
    std::vector<api::Message> messages = {
        {api::Role::User, "Hello"}
    };

    auto result = builder_->build(messages);
    ASSERT_TRUE(result.ok());

    const auto& prompt = result.value();
    EXPECT_FALSE(prompt.formatted_text.empty());
    EXPECT_FALSE(prompt.token_ids.empty());
    EXPECT_EQ(prompt.num_messages, 1);
    EXPECT_TRUE(prompt.has_generation_prompt);
}

TEST_F(PromptBuilderTest, BuildSystemAndUserMessages) {
    std::vector<api::Message> messages = {
        {api::Role::System, "You are a helpful assistant."},
        {api::Role::User, "Hello"}
    };

    auto result = builder_->build(messages);
    ASSERT_TRUE(result.ok());

    const auto& prompt = result.value();
    EXPECT_TRUE(prompt.has_system_message);
    EXPECT_EQ(prompt.num_messages, 2);
}

TEST_F(PromptBuilderTest, BuildMultiTurnConversation) {
    std::vector<api::Message> messages = {
        {api::Role::System, "You are a helpful assistant."},
        {api::Role::User, "Hello"},
        {api::Role::Assistant, "Hi! How can I help?"},
        {api::Role::User, "What's 2+2?"}
    };

    auto result = builder_->build(messages);
    ASSERT_TRUE(result.ok());

    const auto& prompt = result.value();
    EXPECT_EQ(prompt.num_messages, 4);
}

TEST_F(PromptBuilderTest, FormatMessageCorrectly) {
    api::Message msg{api::Role::User, "Hello world"};
    std::string formatted = builder_->format_message(msg);

    EXPECT_NE(formatted.find("<|im_start|>user"), std::string::npos);
    EXPECT_NE(formatted.find("Hello world"), std::string::npos);
    EXPECT_NE(formatted.find("<|im_end|>"), std::string::npos);
}

TEST_F(PromptBuilderTest, DetectsAlreadyFormattedPrompt) {
    EXPECT_TRUE(builder_->is_already_formatted("<|im_start|>user\nHello<|im_end|>"));
    EXPECT_FALSE(builder_->is_already_formatted("Hello world"));
}

TEST_F(PromptBuilderTest, GenerationPromptIsAdded) {
    std::vector<api::Message> messages = {
        {api::Role::User, "Hello"}
    };

    auto result = builder_->build(messages, true);
    ASSERT_TRUE(result.ok());

    const auto& prompt = result.value();
    EXPECT_NE(prompt.formatted_text.find("<|im_start|>assistant\n"),
              std::string::npos);
}

TEST_F(PromptBuilderTest, NoGenerationPromptWhenDisabled) {
    std::vector<api::Message> messages = {
        {api::Role::User, "Hello"}
    };

    auto result = builder_->build(messages, false);
    ASSERT_TRUE(result.ok());

    const auto& prompt = result.value();
    // Should end with im_end from user message, not assistant prefix
    size_t last_im_start = prompt.formatted_text.rfind("<|im_start|>");
    size_t user_pos = prompt.formatted_text.rfind("user");
    EXPECT_LT(last_im_start, user_pos);  // Last im_start should be for user
}

// Golden test: exact template format
TEST_F(PromptBuilderTest, ExactTemplateFormat) {
    std::vector<api::Message> messages = {
        {api::Role::System, "You are helpful."},
        {api::Role::User, "Hi"}
    };

    auto result = builder_->build(messages);
    ASSERT_TRUE(result.ok());

    const auto& formatted = result.value().formatted_text;

    // Expected format:
    // <|im_start|>system
    // You are helpful.<|im_end|>
    // <|im_start|>user
    // Hi<|im_end|>
    // <|im_start|>assistant

    std::string expected =
        "<|im_start|>system\n"
        "You are helpful.<|im_end|>\n"
        "<|im_start|>user\n"
        "Hi<|im_end|>\n"
        "<|im_start|>assistant\n";

    EXPECT_EQ(formatted, expected);
}

}  // namespace qwen::tokenizer
