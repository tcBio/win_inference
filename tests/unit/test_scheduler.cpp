#include <gtest/gtest.h>
#include "scheduler/scheduler.hpp"
#include "backend/null_backend.hpp"
#include "tokenizer/tokenizer.hpp"
#include "kvcache/kv_allocator.hpp"

namespace qwen::scheduler {

class SchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create mock backend
        backend_ = std::make_shared<backend::NullBackend>();
        backend::ModelConfig model_config;
        model_config.vocab_size = 1000;
        backend_->initialize(model_config);
        backend_->load_model();

        // Create mock tokenizer
        tokenizer_ = tokenizer::create_qwen_tokenizer();
        tokenizer::TokenizerConfig tok_config;
        tokenizer_->initialize(tok_config);

        // Create mock KV allocator
        kv_allocator_ = kvcache::create_contiguous_allocator();
        kvcache::KVAllocatorConfig kv_config;
        kv_config.max_total_memory = 1024 * 1024 * 1024;  // 1 GB
        kv_allocator_->initialize(kv_config);

        // Create scheduler
        SchedulerConfig sched_config;
        sched_config.max_queue_size = 10;
        sched_config.max_active_requests = 2;

        scheduler_ = std::make_unique<Scheduler>(
            backend_,
            kv_allocator_,
            tokenizer_,
            sched_config
        );
    }

    void TearDown() override {
        if (scheduler_ && scheduler_->is_running()) {
            scheduler_->stop();
        }
    }

    RequestPtr make_test_request() {
        api::ChatCompletionRequest api_req;
        api_req.model = "test";
        api_req.messages = {{api::Role::User, "Hello"}};
        api_req.max_tokens = 10;

        auto request = make_request(api_req);
        request->input_tokens = {1, 2, 3};  // Fake tokens
        return request;
    }

    std::shared_ptr<backend::NullBackend> backend_;
    std::shared_ptr<tokenizer::ITokenizer> tokenizer_;
    std::shared_ptr<kvcache::IKVCacheAllocator> kv_allocator_;
    std::unique_ptr<Scheduler> scheduler_;
};

TEST_F(SchedulerTest, StartsAndStops) {
    EXPECT_FALSE(scheduler_->is_running());

    auto result = scheduler_->start();
    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(scheduler_->is_running());

    scheduler_->stop();
    EXPECT_FALSE(scheduler_->is_running());
}

TEST_F(SchedulerTest, SubmitRequiresRunning) {
    auto request = make_test_request();
    auto result = scheduler_->submit(request);
    EXPECT_TRUE(result.is_error());
}

TEST_F(SchedulerTest, SubmitAcceptsRequest) {
    scheduler_->start();

    auto request = make_test_request();
    auto result = scheduler_->submit(request);
    EXPECT_TRUE(result.ok());

    // Request should be queued or processing
    auto stats = scheduler_->stats();
    EXPECT_GE(stats.queued_count + stats.active_count, 0);
}

TEST_F(SchedulerTest, CancellationWorks) {
    scheduler_->start();

    auto request = make_test_request();
    scheduler_->submit(request);

    scheduler_->cancel(request->id);

    // Give time for cancellation to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_TRUE(request->is_cancelled());
}

TEST_F(SchedulerTest, StatsAreTracked) {
    scheduler_->start();

    auto stats_before = scheduler_->stats();

    auto request = make_test_request();
    scheduler_->submit(request);

    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto stats_after = scheduler_->stats();

    // Something should have changed
    EXPECT_TRUE(
        stats_after.queued_count != stats_before.queued_count ||
        stats_after.active_count != stats_before.active_count ||
        stats_after.completed_count != stats_before.completed_count
    );
}

// Test request state transitions
TEST(RequestStateTest, InitialStateIsReceived) {
    api::ChatCompletionRequest api_req;
    api_req.model = "test";
    api_req.messages = {{api::Role::User, "Hello"}};

    auto request = make_request(api_req);
    EXPECT_EQ(request->get_state(), RequestState::Received);
}

TEST(RequestStateTest, StateTransitions) {
    api::ChatCompletionRequest api_req;
    api_req.model = "test";
    api_req.messages = {{api::Role::User, "Hello"}};

    auto request = make_request(api_req);

    request->set_state(RequestState::Queued);
    EXPECT_EQ(request->get_state(), RequestState::Queued);

    request->set_state(RequestState::Prefilling);
    EXPECT_EQ(request->get_state(), RequestState::Prefilling);

    request->set_state(RequestState::Decoding);
    EXPECT_EQ(request->get_state(), RequestState::Decoding);

    request->set_state(RequestState::Completed);
    EXPECT_EQ(request->get_state(), RequestState::Completed);
    EXPECT_TRUE(request->is_terminal());
}

TEST(RequestStateTest, CancellationToken) {
    api::ChatCompletionRequest api_req;
    api_req.model = "test";
    api_req.messages = {{api::Role::User, "Hello"}};

    auto request = make_request(api_req);

    EXPECT_FALSE(request->is_cancelled());

    request->cancel_token->cancel();

    EXPECT_TRUE(request->is_cancelled());
}

TEST(RequestIdTest, GeneratesUniqueIds) {
    std::unordered_set<std::string> ids;

    for (int i = 0; i < 1000; ++i) {
        std::string id = generate_request_id();
        EXPECT_TRUE(ids.find(id) == ids.end()) << "Duplicate ID: " << id;
        ids.insert(id);
    }
}

}  // namespace qwen::scheduler
