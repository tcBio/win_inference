#include <gtest/gtest.h>
#include "kvcache/kv_allocator.hpp"

namespace qwen::kvcache {

class MockKVAllocator : public IKVCacheAllocator {
public:
    Result<void> initialize(const KVAllocatorConfig& config) override {
        config_ = config;
        stats_.total_bytes = config.max_total_memory;
        stats_.free_bytes = config.max_total_memory;
        return Result<void>::success();
    }

    Result<KVCacheHandle> allocate(size_t max_seq_len) override {
        size_t bytes = config_.bytes_for_seq_len(max_seq_len);

        if (bytes > stats_.free_bytes) {
            return Error::resource_exhausted("OOM");
        }

        stats_.used_bytes += bytes;
        stats_.free_bytes -= bytes;
        stats_.active_allocations++;

        KVCacheHandle handle;
        handle.k_cache = reinterpret_cast<void*>(next_id_++);
        handle.v_cache = reinterpret_cast<void*>(next_id_++);
        handle.max_seq_len = max_seq_len;
        handle.allocation_id = next_id_;

        allocations_[handle.allocation_id] = bytes;

        return handle;
    }

    void free(KVCacheHandle& handle) override {
        auto it = allocations_.find(handle.allocation_id);
        if (it != allocations_.end()) {
            stats_.used_bytes -= it->second;
            stats_.free_bytes += it->second;
            stats_.active_allocations--;
            allocations_.erase(it);
        }
        handle.k_cache = nullptr;
        handle.v_cache = nullptr;
    }

    bool can_allocate(size_t max_seq_len) const override {
        return config_.bytes_for_seq_len(max_seq_len) <= stats_.free_bytes;
    }

    MemoryStats stats() const override { return stats_; }
    const KVAllocatorConfig& config() const override { return config_; }
    void reset() override {
        stats_.used_bytes = 0;
        stats_.free_bytes = stats_.total_bytes;
        stats_.active_allocations = 0;
        allocations_.clear();
    }

private:
    KVAllocatorConfig config_;
    MemoryStats stats_;
    std::unordered_map<uint64_t, size_t> allocations_;
    uint64_t next_id_ = 1;
};

class KVAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        allocator_ = std::make_unique<MockKVAllocator>();

        KVAllocatorConfig config;
        config.num_layers = 64;
        config.num_kv_heads = 8;
        config.head_dim = 128;
        config.dtype_bytes = 2;
        config.max_total_memory = 1024 * 1024 * 100;  // 100 MB for testing

        allocator_->initialize(config);
    }

    std::unique_ptr<MockKVAllocator> allocator_;
};

TEST_F(KVAllocatorTest, InitialStats) {
    auto stats = allocator_->stats();
    EXPECT_EQ(stats.used_bytes, 0);
    EXPECT_EQ(stats.active_allocations, 0);
    EXPECT_GT(stats.free_bytes, 0);
}

TEST_F(KVAllocatorTest, AllocationSucceeds) {
    auto result = allocator_->allocate(100);
    ASSERT_TRUE(result.ok());

    auto& handle = result.value();
    EXPECT_TRUE(handle.is_valid());
    EXPECT_EQ(handle.max_seq_len, 100);
    EXPECT_EQ(handle.current_len, 0);
}

TEST_F(KVAllocatorTest, AllocationUpdatesStats) {
    auto stats_before = allocator_->stats();

    auto result = allocator_->allocate(100);
    ASSERT_TRUE(result.ok());

    auto stats_after = allocator_->stats();
    EXPECT_GT(stats_after.used_bytes, stats_before.used_bytes);
    EXPECT_LT(stats_after.free_bytes, stats_before.free_bytes);
    EXPECT_EQ(stats_after.active_allocations, 1);
}

TEST_F(KVAllocatorTest, FreeReleasesMemory) {
    auto result = allocator_->allocate(100);
    ASSERT_TRUE(result.ok());
    auto handle = std::move(result.value());

    auto stats_after_alloc = allocator_->stats();

    allocator_->free(handle);

    auto stats_after_free = allocator_->stats();
    EXPECT_LT(stats_after_free.used_bytes, stats_after_alloc.used_bytes);
    EXPECT_EQ(stats_after_free.active_allocations, 0);
    EXPECT_FALSE(handle.is_valid());
}

TEST_F(KVAllocatorTest, OOMWhenExhausted) {
    // Try to allocate more than available
    auto result = allocator_->allocate(1000000);  // Very large
    EXPECT_TRUE(result.is_error());
}

TEST_F(KVAllocatorTest, CanAllocateChecksMemory) {
    EXPECT_TRUE(allocator_->can_allocate(100));
    EXPECT_FALSE(allocator_->can_allocate(1000000));
}

TEST_F(KVAllocatorTest, MultipleAllocations) {
    std::vector<KVCacheHandle> handles;

    for (int i = 0; i < 5; ++i) {
        auto result = allocator_->allocate(50);
        ASSERT_TRUE(result.ok());
        handles.push_back(std::move(result.value()));
    }

    EXPECT_EQ(allocator_->stats().active_allocations, 5);

    for (auto& handle : handles) {
        allocator_->free(handle);
    }

    EXPECT_EQ(allocator_->stats().active_allocations, 0);
}

TEST_F(KVAllocatorTest, ResetClearsState) {
    auto result = allocator_->allocate(100);
    ASSERT_TRUE(result.ok());

    allocator_->reset();

    auto stats = allocator_->stats();
    EXPECT_EQ(stats.used_bytes, 0);
    EXPECT_EQ(stats.active_allocations, 0);
}

// Memory math tests
TEST(MemoryMathTest, BytesPerToken) {
    EXPECT_EQ(qwen32b::BYTES_PER_TOKEN, 262144);  // 256 KB
}

TEST(MemoryMathTest, BytesFor70K) {
    size_t expected = 70000 * 262144;  // ~17.5 GB
    EXPECT_EQ(qwen32b::BYTES_FOR_70K, expected);
}

TEST(MemoryMathTest, MaxConcurrency) {
    size_t budget = 14ULL * 1024 * 1024 * 1024;  // 14 GB

    // At 70k with TP (8.75 GB per request per GPU)
    size_t max_70k = qwen32b::max_concurrent_requests(70000, budget);
    EXPECT_EQ(max_70k, 1);

    // At 10k (1.25 GB per request per GPU)
    size_t max_10k = qwen32b::max_concurrent_requests(10000, budget);
    EXPECT_GE(max_10k, 5);
}

}  // namespace qwen::kvcache
