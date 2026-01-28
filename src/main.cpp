#include "api/server.hpp"
#include "scheduler/scheduler.hpp"
#include "tokenizer/tokenizer.hpp"
#include "tokenizer/prompt_builder.hpp"
#include "kvcache/kv_allocator.hpp"
#include "gpu/device_manager.hpp"
#include "backend/null_backend.hpp"
#include "utils/logger.hpp"
#include "utils/metrics.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <csignal>
#include <atomic>

namespace {
std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_shutdown_requested.store(true);
    }
}
}  // namespace

struct ServerConfiguration {
    qwen::api::ServerConfig api;
    qwen::scheduler::SchedulerConfig scheduler;
    qwen::backend::ModelConfig model;
    qwen::kvcache::KVAllocatorConfig kv_cache;
    qwen::gpu::DeviceManagerConfig gpu;
    qwen::LoggerConfig logging;
};

ServerConfiguration load_config(const std::string& config_path) {
    ServerConfiguration config;

    if (!config_path.empty()) {
        std::ifstream file(config_path);
        if (file.is_open()) {
            try {
                nlohmann::json j;
                file >> j;

                // Parse API config
                if (j.contains("api")) {
                    auto& api = j["api"];
                    config.api.host = api.value("host", config.api.host);
                    config.api.port = api.value("port", config.api.port);
                    config.api.model_name = api.value("model_name", config.api.model_name);
                }

                // Parse scheduler config
                if (j.contains("scheduler")) {
                    auto& sched = j["scheduler"];
                    config.scheduler.max_queue_size = sched.value("max_queue_size",
                        config.scheduler.max_queue_size);
                    config.scheduler.max_active_requests = sched.value("max_active_requests",
                        config.scheduler.max_active_requests);
                }

                // Parse model config
                if (j.contains("model")) {
                    auto& model = j["model"];
                    config.model.model_path = model.value("path", config.model.model_path);
                    config.model.max_seq_len = model.value("max_seq_len", config.model.max_seq_len);
                    config.model.tensor_parallel = model.value("tensor_parallel",
                        config.model.tensor_parallel);
                }

                // Parse GPU config
                if (j.contains("gpu")) {
                    auto& gpu = j["gpu"];
                    if (gpu.contains("device_ids")) {
                        config.gpu.device_ids = gpu["device_ids"].get<std::vector<int32_t>>();
                    }
                }

                // Parse logging config
                if (j.contains("logging")) {
                    auto& log = j["logging"];
                    config.logging.level = log.value("level", config.logging.level);
                    config.logging.output = log.value("output", config.logging.output);
                    config.logging.json_format = log.value("json_format", config.logging.json_format);
                }

            } catch (const std::exception& e) {
                std::cerr << "Failed to parse config file: " << e.what() << std::endl;
            }
        }
    }

    return config;
}

int main(int argc, char* argv[]) {
    // Parse command line
    std::string config_path;
    if (argc > 1) {
        config_path = argv[1];
    }

    // Load configuration
    auto config = load_config(config_path);

    // Initialize logging
    qwen::init_logger(config.logging);
    qwen::log_info("main", "starting", {{"version", "0.1.0"}});

    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        // Initialize GPU devices
        auto device_manager = std::make_shared<qwen::gpu::DeviceManager>(config.gpu);
        auto init_result = device_manager->initialize();
        if (init_result.is_error()) {
            qwen::log_error("main", "gpu_init_failed", {
                {"error", init_result.error().message}
            });
            return 1;
        }

        // Initialize tokenizer
        auto tokenizer = qwen::tokenizer::create_qwen_tokenizer();
        qwen::tokenizer::TokenizerConfig tokenizer_config;
        auto tokenizer_result = tokenizer->initialize(tokenizer_config);
        if (tokenizer_result.is_error()) {
            qwen::log_error("main", "tokenizer_init_failed", {
                {"error", tokenizer_result.error().message}
            });
            return 1;
        }

        // Create prompt builder
        auto prompt_builder = std::make_shared<qwen::tokenizer::PromptBuilder>(
            tokenizer,
            qwen::tokenizer::PromptBuilderConfig{}
        );

        // Initialize KV cache allocator
        auto kv_allocator = qwen::kvcache::create_contiguous_allocator();
        auto kv_result = kv_allocator->initialize(config.kv_cache);
        if (kv_result.is_error()) {
            qwen::log_error("main", "kv_cache_init_failed", {
                {"error", kv_result.error().message}
            });
            return 1;
        }

        // Initialize backend (use null backend for now)
        auto backend = std::make_shared<qwen::backend::NullBackend>();
        auto backend_result = backend->initialize(config.model);
        if (backend_result.is_error()) {
            qwen::log_error("main", "backend_init_failed", {
                {"error", backend_result.error().message}
            });
            return 1;
        }
        backend->load_model();

        // Create scheduler
        auto scheduler = std::make_shared<qwen::scheduler::Scheduler>(
            backend,
            kv_allocator,
            tokenizer,
            config.scheduler
        );

        auto sched_result = scheduler->start();
        if (sched_result.is_error()) {
            qwen::log_error("main", "scheduler_start_failed", {
                {"error", sched_result.error().message}
            });
            return 1;
        }

        // Create and start HTTP server
        qwen::api::Server server(scheduler, prompt_builder, config.api);
        auto server_result = server.start();
        if (server_result.is_error()) {
            qwen::log_error("main", "server_start_failed", {
                {"error", server_result.error().message}
            });
            return 1;
        }

        qwen::log_info("main", "ready", {
            {"host", config.api.host},
            {"port", config.api.port},
            {"model", config.api.model_name}
        });

        // Wait for shutdown signal
        while (!g_shutdown_requested.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        qwen::log_info("main", "shutting_down", {});

        // Graceful shutdown
        server.stop();
        scheduler->stop();
        backend->unload_model();

        qwen::log_info("main", "shutdown_complete", {});

    } catch (const std::exception& e) {
        qwen::log_error("main", "fatal_error", {{"error", e.what()}});
        return 1;
    }

    return 0;
}
