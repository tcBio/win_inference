#include "scheduler/scheduler.hpp"
#include "backend/null_backend.hpp"
#include "tokenizer/tokenizer.hpp"
#include "kvcache/kv_allocator.hpp"

#include <chrono>
#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <latch>

using namespace qwen;

int main() {
    std::cout << "=== Scheduler Benchmarks ===\n\n";

    // Create backend
    auto backend = std::make_shared<backend::NullBackend>();
    backend::ModelConfig model_config;
    model_config.vocab_size = 152064;
    backend->initialize(model_config);
    backend->load_model();

    // Create tokenizer
    auto tokenizer = tokenizer::create_qwen_tokenizer();
    tokenizer::TokenizerConfig tok_config;
    tokenizer->initialize(tok_config);

    // Create KV allocator
    auto kv_allocator = kvcache::create_contiguous_allocator();
    kvcache::KVAllocatorConfig kv_config;
    kv_config.max_total_memory = 4ULL * 1024 * 1024 * 1024;
    kv_allocator->initialize(kv_config);

    // Create scheduler
    scheduler::SchedulerConfig sched_config;
    sched_config.max_queue_size = 100;
    sched_config.max_active_requests = 4;

    auto scheduler = std::make_shared<scheduler::Scheduler>(
        backend,
        kv_allocator,
        tokenizer,
        sched_config
    );

    scheduler->start();

    std::cout << "--- Single Request Latency ---\n";

    auto make_request = [](int prompt_len, int max_tokens) {
        api::ChatCompletionRequest api_req;
        api_req.model = "test";
        api_req.messages = {{api::Role::User, "Hello"}};
        api_req.max_tokens = max_tokens;

        auto request = scheduler::make_request(api_req);
        request->input_tokens.resize(prompt_len, 1);  // Fake tokens
        return request;
    };

    std::vector<std::pair<int, int>> test_cases = {
        {100, 10},
        {1000, 50},
        {5000, 100}
    };

    for (const auto& [prompt_len, max_tokens] : test_cases) {
        auto request = make_request(prompt_len, max_tokens);

        std::atomic<bool> done{false};
        auto start = std::chrono::high_resolution_clock::now();

        request->on_complete = [&](api::FinishReason) {
            done.store(true);
        };
        request->on_error = [&](const Error&) {
            done.store(true);
        };

        scheduler->submit(request);

        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "prompt_len=" << prompt_len
                  << ", max_tokens=" << max_tokens
                  << ", latency=" << duration << " ms\n";
    }

    std::cout << "\n--- Concurrent Request Throughput ---\n";

    std::vector<int> concurrency_levels = {1, 2, 4};

    for (int concurrency : concurrency_levels) {
        std::atomic<int> completed{0};
        int total_requests = 20;

        auto start = std::chrono::high_resolution_clock::now();

        // Submit requests
        for (int i = 0; i < total_requests; ++i) {
            auto request = make_request(100, 10);
            request->on_complete = [&](api::FinishReason) {
                completed.fetch_add(1);
            };
            request->on_error = [&](const Error&) {
                completed.fetch_add(1);
            };
            scheduler->submit(request);

            // Pace submissions
            if ((i + 1) % concurrency == 0) {
                while (completed.load() < i + 1 - concurrency) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        }

        // Wait for all to complete
        while (completed.load() < total_requests) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start).count();

        double requests_per_sec = total_requests / duration;

        std::cout << "concurrency=" << concurrency
                  << ", requests=" << total_requests
                  << ", duration=" << duration << " s"
                  << ", throughput=" << requests_per_sec << " req/s\n";
    }

    std::cout << "\n--- Queue Latency ---\n";

    // Measure time spent in queue
    std::vector<double> queue_times;

    for (int i = 0; i < 10; ++i) {
        auto request = make_request(100, 5);

        std::atomic<bool> done{false};

        request->on_complete = [&](api::FinishReason) {
            queue_times.push_back(request->timing.queue_time_ms());
            done.store(true);
        };

        scheduler->submit(request);

        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    double mean_queue = std::accumulate(queue_times.begin(), queue_times.end(), 0.0) / queue_times.size();
    std::cout << "Mean queue time: " << mean_queue << " ms\n";

    scheduler->stop();

    std::cout << "\n=== Benchmarks Complete ===\n";
    return 0;
}
