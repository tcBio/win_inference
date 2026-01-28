#include "kvcache/kv_allocator.hpp"

#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include <random>

using namespace qwen;

template <typename Func>
double benchmark_single(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

int main() {
    std::cout << "=== KV Cache Benchmarks ===\n\n";

    auto allocator = kvcache::create_contiguous_allocator();

    kvcache::KVAllocatorConfig config;
    config.num_layers = 64;
    config.num_kv_heads = 8;
    config.head_dim = 128;
    config.dtype_bytes = 2;
    config.max_total_memory = 8ULL * 1024 * 1024 * 1024;  // 8 GB for testing

    auto init_result = allocator->initialize(config);
    if (init_result.is_error()) {
        std::cerr << "Failed to initialize allocator\n";
        return 1;
    }

    std::cout << "Bytes per token: " << config.bytes_per_token() << "\n";
    std::cout << "Available memory: " << allocator->stats().total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n\n";

    // Benchmark allocations of various sizes
    std::vector<size_t> seq_lengths = {100, 1000, 5000, 10000, 20000, 50000, 70000};

    std::cout << "--- Allocation Time vs Sequence Length ---\n";
    for (size_t seq_len : seq_lengths) {
        if (!allocator->can_allocate(seq_len)) {
            std::cout << "seq_len=" << seq_len << ": Cannot allocate (OOM)\n";
            continue;
        }

        // Measure allocation
        double alloc_time = benchmark_single([&]() {
            auto result = allocator->allocate(seq_len);
            if (result.ok()) {
                auto handle = std::move(result.value());
                allocator->free(handle);
            }
        });

        size_t bytes = config.bytes_for_seq_len(seq_len);
        std::cout << "seq_len=" << seq_len
                  << ", bytes=" << bytes / (1024.0 * 1024.0) << " MB"
                  << ", alloc_time=" << alloc_time << " us\n";
    }

    std::cout << "\n--- Allocation/Free Throughput ---\n";

    // Measure rapid alloc/free cycles
    size_t test_seq_len = 1000;
    int num_cycles = 1000;

    std::vector<double> times;
    for (int i = 0; i < num_cycles; ++i) {
        double t = benchmark_single([&]() {
            auto result = allocator->allocate(test_seq_len);
            if (result.ok()) {
                auto handle = std::move(result.value());
                allocator->free(handle);
            }
        });
        times.push_back(t);
    }

    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << "Alloc/free cycle (seq_len=" << test_seq_len << "): "
              << mean << " us mean, "
              << 1000000.0 / mean << " ops/sec\n";

    std::cout << "\n--- Concurrent Allocations ---\n";

    // Allocate multiple caches simultaneously
    std::vector<kvcache::KVCacheHandle> handles;
    size_t concurrent_seq_len = 1000;
    int max_concurrent = 0;

    while (allocator->can_allocate(concurrent_seq_len)) {
        auto result = allocator->allocate(concurrent_seq_len);
        if (result.is_error()) break;
        handles.push_back(std::move(result.value()));
        max_concurrent++;
    }

    std::cout << "Max concurrent allocations (seq_len=" << concurrent_seq_len << "): "
              << max_concurrent << "\n";
    std::cout << "Memory used: " << allocator->stats().used_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";

    // Cleanup
    for (auto& handle : handles) {
        allocator->free(handle);
    }

    std::cout << "\n=== Benchmarks Complete ===\n";
    return 0;
}
