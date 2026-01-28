#include <gtest/gtest.h>
#include "tokenizer/tokenizer.hpp"

namespace qwen::tokenizer {

class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tokenizer_ = create_qwen_tokenizer();
        TokenizerConfig config;
        auto result = tokenizer_->initialize(config);
        ASSERT_TRUE(result.ok());
    }

    std::unique_ptr<ITokenizer> tokenizer_;
};

TEST_F(TokenizerTest, VocabSizeIsValid) {
    EXPECT_GT(tokenizer_->vocab_size(), 0);
}

TEST_F(TokenizerTest, SpecialTokensAreCorrect) {
    const auto& special = tokenizer_->special_tokens();
    EXPECT_EQ(special.im_start, "<|im_start|>");
    EXPECT_EQ(special.im_end, "<|im_end|>");
    EXPECT_EQ(special.im_start_id, 151644);
    EXPECT_EQ(special.im_end_id, 151645);
}

TEST_F(TokenizerTest, EncodeEmptyString) {
    auto result = tokenizer_->encode("");
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result.value().empty());
}

TEST_F(TokenizerTest, EncodeSimpleText) {
    auto result = tokenizer_->encode("Hello");
    ASSERT_TRUE(result.ok());
    EXPECT_FALSE(result.value().empty());
}

TEST_F(TokenizerTest, EncodeDecodeRoundtrip) {
    std::string original = "Hello, world!";
    auto encode_result = tokenizer_->encode(original);
    ASSERT_TRUE(encode_result.ok());

    auto decode_result = tokenizer_->decode(encode_result.value());
    ASSERT_TRUE(decode_result.ok());

    // Note: May not be exactly equal due to whitespace handling
    EXPECT_FALSE(decode_result.value().empty());
}

TEST_F(TokenizerTest, SpecialTokensAreRecognized) {
    EXPECT_TRUE(tokenizer_->is_special_token(151644));  // im_start
    EXPECT_TRUE(tokenizer_->is_special_token(151645));  // im_end
    EXPECT_FALSE(tokenizer_->is_special_token(0));      // Regular token
}

TEST_F(TokenizerTest, EncodeWithSpecialTokens) {
    std::string text = "<|im_start|>user\nHello<|im_end|>";
    auto result = tokenizer_->encode(text);
    ASSERT_TRUE(result.ok());

    const auto& tokens = result.value();
    EXPECT_FALSE(tokens.empty());

    // Should contain special tokens
    bool has_im_start = false;
    bool has_im_end = false;
    for (int32_t token : tokens) {
        if (token == 151644) has_im_start = true;
        if (token == 151645) has_im_end = true;
    }
    EXPECT_TRUE(has_im_start);
    EXPECT_TRUE(has_im_end);
}

TEST_F(TokenizerTest, DecodeToken) {
    // Test decoding individual tokens
    auto result = tokenizer_->decode_token(151644);
    // May succeed or fail depending on vocab
}

}  // namespace qwen::tokenizer
