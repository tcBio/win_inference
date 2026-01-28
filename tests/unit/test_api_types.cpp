#include <gtest/gtest.h>
#include "api/types.hpp"
#include "api/request_validator.hpp"

namespace qwen::api {

TEST(ApiTypesTest, RoleConversion) {
    EXPECT_EQ(role_to_string(Role::System), "system");
    EXPECT_EQ(role_to_string(Role::User), "user");
    EXPECT_EQ(role_to_string(Role::Assistant), "assistant");
    EXPECT_EQ(role_to_string(Role::Tool), "tool");

    EXPECT_EQ(string_to_role("system"), Role::System);
    EXPECT_EQ(string_to_role("user"), Role::User);
    EXPECT_EQ(string_to_role("assistant"), Role::Assistant);
    EXPECT_EQ(string_to_role("tool"), Role::Tool);
    EXPECT_EQ(string_to_role("unknown"), Role::User);  // Default
}

TEST(ApiTypesTest, FinishReasonConversion) {
    EXPECT_EQ(finish_reason_to_string(FinishReason::Stop), "stop");
    EXPECT_EQ(finish_reason_to_string(FinishReason::Length), "length");
    EXPECT_EQ(finish_reason_to_string(FinishReason::ToolCalls), "tool_calls");
    EXPECT_EQ(finish_reason_to_string(FinishReason::None), "");
}

TEST(ApiTypesTest, MessageJsonSerialization) {
    Message msg{Role::User, "Hello", std::nullopt, std::nullopt, std::nullopt};

    nlohmann::json j;
    to_json(j, msg);

    EXPECT_EQ(j["role"], "user");
    EXPECT_EQ(j["content"], "Hello");
    EXPECT_FALSE(j.contains("name"));
}

TEST(ApiTypesTest, MessageJsonDeserialization) {
    nlohmann::json j = {
        {"role", "assistant"},
        {"content", "Hi there!"}
    };

    Message msg;
    from_json(j, msg);

    EXPECT_EQ(msg.role, Role::Assistant);
    EXPECT_EQ(msg.content, "Hi there!");
}

TEST(ApiTypesTest, ChatCompletionRequestParsing) {
    nlohmann::json j = {
        {"model", "qwen2.5-32b-instruct"},
        {"messages", {
            {{"role", "user"}, {"content", "Hello"}}
        }},
        {"temperature", 0.7},
        {"max_tokens", 100},
        {"stream", true}
    };

    ChatCompletionRequest req;
    from_json(j, req);

    EXPECT_EQ(req.model, "qwen2.5-32b-instruct");
    EXPECT_EQ(req.messages.size(), 1);
    EXPECT_EQ(req.messages[0].role, Role::User);
    EXPECT_TRUE(req.temperature.has_value());
    EXPECT_DOUBLE_EQ(*req.temperature, 0.7);
    EXPECT_TRUE(req.max_tokens.has_value());
    EXPECT_EQ(*req.max_tokens, 100);
    EXPECT_TRUE(req.stream);
}

TEST(ApiTypesTest, ChatCompletionResponseSerialization) {
    ChatCompletionResponse resp{
        .id = "chatcmpl-123",
        .created = 1234567890,
        .model = "qwen2.5-32b-instruct",
        .choices = {{
            .index = 0,
            .message = {Role::Assistant, "Hello!"},
            .finish_reason = FinishReason::Stop
        }},
        .usage = {10, 5, 15}
    };

    nlohmann::json j;
    to_json(j, resp);

    EXPECT_EQ(j["id"], "chatcmpl-123");
    EXPECT_EQ(j["object"], "chat.completion");
    EXPECT_EQ(j["choices"].size(), 1);
    EXPECT_EQ(j["choices"][0]["finish_reason"], "stop");
    EXPECT_EQ(j["usage"]["total_tokens"], 15);
}

TEST(ApiTypesTest, StreamingChunkSerialization) {
    ChatCompletionChunk chunk{
        .id = "chatcmpl-123",
        .created = 1234567890,
        .model = "qwen2.5-32b-instruct",
        .choices = {{
            .index = 0,
            .delta = {.content = "Hello"},
            .finish_reason = std::nullopt
        }}
    };

    nlohmann::json j;
    to_json(j, chunk);

    EXPECT_EQ(j["object"], "chat.completion.chunk");
    EXPECT_EQ(j["choices"][0]["delta"]["content"], "Hello");
    EXPECT_TRUE(j["choices"][0]["finish_reason"].is_null());
}

class RequestValidatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        ValidationConfig config;
        config.supported_models = {"qwen2.5-32b-instruct"};
        validator_ = std::make_unique<RequestValidator>(config);
    }

    std::unique_ptr<RequestValidator> validator_;
};

TEST_F(RequestValidatorTest, ValidRequest) {
    ChatCompletionRequest req;
    req.model = "qwen2.5-32b-instruct";
    req.messages = {{Role::User, "Hello"}};

    auto result = validator_->validate(req);
    EXPECT_TRUE(result.ok());
}

TEST_F(RequestValidatorTest, EmptyModelFails) {
    ChatCompletionRequest req;
    req.model = "";
    req.messages = {{Role::User, "Hello"}};

    auto result = validator_->validate(req);
    EXPECT_TRUE(result.is_error());
}

TEST_F(RequestValidatorTest, UnsupportedModelFails) {
    ChatCompletionRequest req;
    req.model = "gpt-4";
    req.messages = {{Role::User, "Hello"}};

    auto result = validator_->validate(req);
    EXPECT_TRUE(result.is_error());
}

TEST_F(RequestValidatorTest, EmptyMessagesFails) {
    ChatCompletionRequest req;
    req.model = "qwen2.5-32b-instruct";
    req.messages = {};

    auto result = validator_->validate(req);
    EXPECT_TRUE(result.is_error());
}

TEST_F(RequestValidatorTest, NoUserMessageFails) {
    ChatCompletionRequest req;
    req.model = "qwen2.5-32b-instruct";
    req.messages = {{Role::System, "You are helpful"}};

    auto result = validator_->validate(req);
    EXPECT_TRUE(result.is_error());
}

TEST_F(RequestValidatorTest, InvalidTemperatureFails) {
    ChatCompletionRequest req;
    req.model = "qwen2.5-32b-instruct";
    req.messages = {{Role::User, "Hello"}};
    req.temperature = 3.0;  // Out of range

    auto result = validator_->validate(req);
    EXPECT_TRUE(result.is_error());
}

TEST_F(RequestValidatorTest, InvalidTopPFails) {
    ChatCompletionRequest req;
    req.model = "qwen2.5-32b-instruct";
    req.messages = {{Role::User, "Hello"}};
    req.top_p = 1.5;  // Out of range

    auto result = validator_->validate(req);
    EXPECT_TRUE(result.is_error());
}

TEST_F(RequestValidatorTest, AppliesDefaults) {
    ChatCompletionRequest req;
    req.model = "qwen2.5-32b-instruct";
    req.messages = {{Role::User, "Hello"}};

    EXPECT_FALSE(req.temperature.has_value());
    EXPECT_FALSE(req.top_p.has_value());
    EXPECT_FALSE(req.max_tokens.has_value());

    validator_->apply_defaults(req);

    EXPECT_TRUE(req.temperature.has_value());
    EXPECT_TRUE(req.top_p.has_value());
    EXPECT_TRUE(req.max_tokens.has_value());
}

}  // namespace qwen::api
