#include "api/server.hpp"
#include "scheduler/scheduler.hpp"
#include "backend/null_backend.hpp"
#include "tokenizer/tokenizer.hpp"
#include "tokenizer/prompt_builder.hpp"
#include "kvcache/kv_allocator.hpp"

#include <httplib.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <future>

using namespace qwen;

struct LatencyStats {
    double p50_ms;
    double p95_ms;
    double p99_ms;
    double mean_ms;
    double min_ms;
    double max_ms;
};

LatencyStats calculate_stats(std::vector<double>& latencies) {
    std::sort(latencies.begin(), latencies.end());

    size_t n = latencies.size();
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);

    return LatencyStats{
        .p50_ms = latencies[n * 50 / 100],
        .p95_ms = latencies[n * 95 / 100],
        .p99_ms = latencies[n * 99 / 100],
        .mean_ms = sum / n,
        .min_ms = latencies.front(),
        .max_ms = latencies.back()
    };
}

int main() {
    std::cout << "=== End-to-End Benchmarks ===\n\n";

    // Create backend
    auto backend = std::make_shared<backend::NullBackend>();
    backend::ModelConfig model_config;
    model_config.vocab_size = 152064;
    backend->initialize(model_config);
    backend->load_model();
    backend->set_seed(42);

    // Create tokenizer
    auto tokenizer = tokenizer::create_qwen_tokenizer();
    tokenizer::TokenizerConfig tok_config;
    tokenizer->initialize(tok_config);

    auto prompt_builder = std::make_shared<tokenizer::PromptBuilder>(
        tokenizer,
        tokenizer::PromptBuilderConfig{}
    );

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

    // Create server
    api::ServerConfig server_config;
    server_config.port = 18081;
    server_config.model_name = "qwen2.5-32b-instruct";

    api::Server server(scheduler, prompt_builder, server_config);
    server.start();

    // Wait for server to be ready
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::cout << "Server started on port " << server_config.port << "\n\n";

    // Benchmark function
    auto run_benchmark = [](int port, int num_requests, int concurrency,
                           const std::string& description) {
        std::cout << "--- " << description << " ---\n";
        std::cout << "Requests: " << num_requests << ", Concurrency: " << concurrency << "\n";

        std::vector<double> latencies;
        std::atomic<int> completed{0};
        std::mutex latency_mutex;

        auto worker = [&](int start_idx, int count) {
            httplib::Client client("localhost", port);
            client.set_connection_timeout(30);
            client.set_read_timeout(30);

            nlohmann::json req_body = {
                {"model", "qwen2.5-32b-instruct"},
                {"messages", {{{"role", "user"}, {"content", "Hello, how are you?"}}}},
                {"max_tokens", 10},
                {"stream", false}
            };

            for (int i = 0; i < count; ++i) {
                auto start = std::chrono::high_resolution_clock::now();

                auto res = client.Post("/v1/chat/completions",
                                       req_body.dump(), "application/json");

                auto end = std::chrono::high_resolution_clock::now();
                double latency = std::chrono::duration<double, std::milli>(end - start).count();

                if (res && res->status == 200) {
                    std::lock_guard<std::mutex> lock(latency_mutex);
                    latencies.push_back(latency);
                }

                completed.fetch_add(1);
            }
        };

        auto start = std::chrono::high_resolution_clock::now();

        // Launch workers
        std::vector<std::thread> workers;
        int per_worker = num_requests / concurrency;

        for (int i = 0; i < concurrency; ++i) {
            int count = (i == concurrency - 1) ?
                        (num_requests - per_worker * i) : per_worker;
            workers.emplace_back(worker, i * per_worker, count);
        }

        // Wait for completion
        for (auto& w : workers) {
            w.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end - start).count();

        if (!latencies.empty()) {
            auto stats = calculate_stats(latencies);

            std::cout << "Successful requests: " << latencies.size() << "/" << num_requests << "\n";
            std::cout << "Total time: " << total_time << " s\n";
            std::cout << "Throughput: " << latencies.size() / total_time << " req/s\n";
            std::cout << "Latency p50: " << stats.p50_ms << " ms\n";
            std::cout << "Latency p95: " << stats.p95_ms << " ms\n";
            std::cout << "Latency p99: " << stats.p99_ms << " ms\n";
            std::cout << "Latency mean: " << stats.mean_ms << " ms\n";
            std::cout << "Latency min: " << stats.min_ms << " ms\n";
            std::cout << "Latency max: " << stats.max_ms << " ms\n";
        } else {
            std::cout << "No successful requests!\n";
        }

        std::cout << "\n";
    };

    // Run benchmarks
    run_benchmark(18081, 10, 1, "Warmup");
    run_benchmark(18081, 50, 1, "Single Client");
    run_benchmark(18081, 100, 4, "4 Concurrent Clients");
    run_benchmark(18081, 100, 8, "8 Concurrent Clients");

    // Cleanup
    server.stop();
    scheduler->stop();

    std::cout << "=== Benchmarks Complete ===\n";
    return 0;
}
