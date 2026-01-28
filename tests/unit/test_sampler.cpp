#include <gtest/gtest.h>
#include "core/sampler.hpp"

namespace qwen::core {

class SamplerTest : public ::testing::Test {
protected:
    void SetUp() override {
        sampler_ = std::make_unique<Sampler>();
    }

    backend::Logits make_logits(int vocab_size, int preferred_token = -1) {
        backend::Logits logits;
        logits.vocab_size = vocab_size;
        logits.data.resize(vocab_size, -10.0f);

        if (preferred_token >= 0 && preferred_token < vocab_size) {
            logits.data[preferred_token] = 10.0f;
        }

        return logits;
    }

    std::unique_ptr<Sampler> sampler_;
};

TEST_F(SamplerTest, GreedySamplingSelectsMaxLogit) {
    auto logits = make_logits(100, 42);

    auto result = sampler_->sample_greedy(logits);

    EXPECT_EQ(result.token_id, 42);
    EXPECT_GT(result.probability, 0.9f);  // Should be nearly 1.0
}

TEST_F(SamplerTest, GreedySamplingWithMultiplePeaks) {
    backend::Logits logits;
    logits.vocab_size = 100;
    logits.data.resize(100, -10.0f);
    logits.data[50] = 5.0f;
    logits.data[60] = 10.0f;  // Highest
    logits.data[70] = 7.0f;

    auto result = sampler_->sample_greedy(logits);

    EXPECT_EQ(result.token_id, 60);
}

TEST_F(SamplerTest, ZeroTemperatureIsGreedy) {
    auto logits = make_logits(100, 42);

    SamplingParams params;
    params.temperature = 0.0;

    auto result = sampler_->sample(logits, params);
    ASSERT_TRUE(result.ok());

    EXPECT_EQ(result.value().token_id, 42);
}

TEST_F(SamplerTest, DeterministicWithSeed) {
    auto logits = make_logits(100);
    logits.data[10] = 5.0f;
    logits.data[20] = 5.0f;
    logits.data[30] = 5.0f;

    SamplingParams params;
    params.temperature = 1.0;
    params.seed = 12345;

    // Sample multiple times with same seed
    std::vector<int32_t> results;
    for (int i = 0; i < 10; ++i) {
        params.seed = 12345;
        auto result = sampler_->sample(logits, params);
        ASSERT_TRUE(result.ok());
        results.push_back(result.value().token_id);
    }

    // All results should be the same
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_EQ(results[0], results[i]);
    }
}

TEST_F(SamplerTest, TopPFiltersTokens) {
    backend::Logits logits;
    logits.vocab_size = 10;
    logits.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

    SamplingParams params;
    params.temperature = 1.0;
    params.top_p = 0.1;  // Very restrictive
    params.seed = 42;

    auto result = sampler_->sample(logits, params);
    ASSERT_TRUE(result.ok());

    // Should be one of the top tokens
    EXPECT_GE(result.value().token_id, 8);  // Top 2 tokens have most probability
}

TEST_F(SamplerTest, TopKFiltersTokens) {
    backend::Logits logits;
    logits.vocab_size = 10;
    logits.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

    SamplingParams params;
    params.temperature = 1.0;
    params.top_k = 2;
    params.seed = 42;

    auto result = sampler_->sample(logits, params);
    ASSERT_TRUE(result.ok());

    // Should be one of top 2 tokens
    EXPECT_TRUE(result.value().token_id == 8 || result.value().token_id == 9);
}

TEST_F(SamplerTest, TemperatureScalesLogits) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    Sampler::apply_temperature(logits, 2.0);

    EXPECT_FLOAT_EQ(logits[0], 0.5f);
    EXPECT_FLOAT_EQ(logits[1], 1.0f);
    EXPECT_FLOAT_EQ(logits[2], 1.5f);
}

TEST_F(SamplerTest, SoftmaxNormalizesToOne) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    auto probs = Sampler::softmax(logits);

    float sum = 0.0f;
    for (float p : probs) {
        sum += p;
        EXPECT_GE(p, 0.0f);
        EXPECT_LE(p, 1.0f);
    }

    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(SamplerTest, RepetitionPenaltyReducesProbability) {
    std::vector<float> logits = {5.0f, 5.0f, 5.0f};
    std::vector<int32_t> previous_tokens = {0, 1};

    Sampler::apply_repetition_penalty(logits, previous_tokens, 2.0);

    // Tokens 0 and 1 should have lower logits
    EXPECT_LT(logits[0], 5.0f);
    EXPECT_LT(logits[1], 5.0f);
    EXPECT_FLOAT_EQ(logits[2], 5.0f);  // Unchanged
}

TEST_F(SamplerTest, EmptyLogitsReturnsError) {
    backend::Logits logits;
    logits.vocab_size = 0;

    SamplingParams params;
    auto result = sampler_->sample(logits, params);

    EXPECT_TRUE(result.is_error());
}

}  // namespace qwen::core
