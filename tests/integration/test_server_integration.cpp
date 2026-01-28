#include <gtest/gtest.h>
#include "api/server.hpp"
#include "scheduler/scheduler.hpp"
#include "backend/null_backend.hpp"
#include "tokenizer/tokenizer.hpp"
#include "tokenizer/prompt_builder.hpp"
#include "kvcache/kv_allocator.hpp"

#include <httplib.h>
#include <thread>
#include <chrono>

namespace qwen::integration {

class ServerIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create backend
        backend_ = std::make_shared<backend::NullBackend>();
        backend::ModelConfig model_config;
        model_config.vocab_size = 1000;
        backend_->initialize(model_config);
        backend_->load_model();

        // Set deterministic output for testing
        backend_->set_seed(42);
        backend_->set_output_sequence({100, 101, 102, 103, 151645});  // 4 tokens + EOS

        // Create tokenizer
        tokenizer_ = tokenizer::create_qwen_tokenizer();
        tokenizer::TokenizerConfig tok_config;
        tokenizer_->initialize(tok_config);

        // Create prompt builder
        prompt_builder_ = std::make_shared<tokenizer::PromptBuilder>(
            tokenizer_,
            tokenizer::PromptBuilderConfig{}
        );

        // Create KV allocator
        kv_allocator_ = kvcache::create_contiguous_allocator();
        kvcache::KVAllocatorConfig kv_config;
        kv_config.max_total_memory = 1024 * 1024 * 100;
        kv_allocator_->initialize(kv_config);

        // Create scheduler
        scheduler::SchedulerConfig sched_config;
        scheduler_ = std::make_shared<scheduler::Scheduler>(
            backend_,
            kv_allocator_,
            tokenizer_,
            sched_config
        );
        scheduler_->start();

        // Create server
        api::ServerConfig server_config;
        server_config.port = 18080;  // Use non-standard port for testing
        server_config.model_name = "test-model";

        server_ = std::make_unique<api::Server>(
            scheduler_,
            prompt_builder_,
            server_config
        );
    }

    void TearDown() override {
        if (server_ && server_->is_running()) {
            server_->stop();
        }
        if (scheduler_ && scheduler_->is_running()) {
            scheduler_->stop();
        }
    }

    void StartServer() {
        auto result = server_->start();
        ASSERT_TRUE(result.ok()) << "Failed to start server";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::shared_ptr<backend::NullBackend> backend_;
    std::shared_ptr<tokenizer::ITokenizer> tokenizer_;
    std::shared_ptr<tokenizer::PromptBuilder> prompt_builder_;
    std::shared_ptr<kvcache::IKVCacheAllocator> kv_allocator_;
    std::shared_ptr<scheduler::Scheduler> scheduler_;
    std::unique_ptr<api::Server> server_;
};

TEST_F(ServerIntegrationTest, HealthCheck) {
    StartServer();

    httplib::Client client("localhost", 18080);
    auto res = client.Get("/health");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);

    auto json = nlohmann::json::parse(res->body);
    EXPECT_EQ(json["status"], "healthy");
}

TEST_F(ServerIntegrationTest, ListModels) {
    StartServer();

    httplib::Client client("localhost", 18080);
    auto res = client.Get("/v1/models");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);

    auto json = nlohmann::json::parse(res->body);
    EXPECT_EQ(json["object"], "list");
    EXPECT_FALSE(json["data"].empty());
    EXPECT_EQ(json["data"][0]["id"], "test-model");
}

TEST_F(ServerIntegrationTest, MetricsEndpoint) {
    StartServer();

    httplib::Client client("localhost", 18080);
    auto res = client.Get("/metrics");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("inference_"), std::string::npos);
}

TEST_F(ServerIntegrationTest, InvalidRequestReturnsError) {
    StartServer();

    httplib::Client client("localhost", 18080);

    // Empty body
    auto res = client.Post("/v1/chat/completions", "{}", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);

    // Invalid JSON
    res = client.Post("/v1/chat/completions", "not json", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(ServerIntegrationTest, MissingModelReturnsError) {
    StartServer();

    httplib::Client client("localhost", 18080);

    nlohmann::json req = {
        {"messages", {{{"role", "user"}, {"content", "Hello"}}}}
        // Missing "model"
    };

    auto res = client.Post("/v1/chat/completions", req.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(ServerIntegrationTest, UnsupportedModelReturnsError) {
    StartServer();

    httplib::Client client("localhost", 18080);

    nlohmann::json req = {
        {"model", "gpt-4"},
        {"messages", {{{"role", "user"}, {"content", "Hello"}}}}
    };

    auto res = client.Post("/v1/chat/completions", req.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

// Note: Full streaming tests require more sophisticated client handling
// These tests verify the basic request/response flow

}  // namespace qwen::integration
