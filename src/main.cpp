#include "api/server.hpp"
#include "scheduler/scheduler.hpp"
#include "tokenizer/tokenizer.hpp"
#include "tokenizer/prompt_builder.hpp"
#include "kvcache/kv_allocator.hpp"
#include "gpu/device_manager.hpp"
#include "gpu/gpu_router.hpp"
#include "backend/null_backend.hpp"
#include "utils/logger.hpp"
#include "utils/metrics.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <atomic>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#endif

namespace {
std::atomic<bool> g_shutdown_requested{false};

#ifdef _WIN32
// Windows console control handler
BOOL WINAPI console_handler(DWORD ctrl_type) {
    switch (ctrl_type) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            g_shutdown_requested.store(true);
            return TRUE;
        default:
            return FALSE;
    }
}
#else
// POSIX signal handler
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_shutdown_requested.store(true);
    }
}
#endif

void setup_signal_handlers() {
#ifdef _WIN32
    SetConsoleCtrlHandler(console_handler, TRUE);
#else
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#endif
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

/// Validate configuration values
/// Returns error message if invalid, empty string if valid
std::string validate_config(const ServerConfiguration& config) {
    // Validate port
    if (config.api.port <= 0 || config.api.port > 65535) {
        return "Invalid port: " + std::to_string(config.api.port) + " (must be 1-65535)";
    }

    // Validate scheduler settings
    if (config.scheduler.max_queue_size == 0) {
        return "max_queue_size must be > 0";
    }
    if (config.scheduler.max_active_requests == 0) {
        return "max_active_requests must be > 0";
    }

    // Validate model settings
    if (config.model.max_seq_len <= 0) {
        return "max_seq_len must be > 0";
    }
    if (config.model.vocab_size <= 0) {
        return "vocab_size must be > 0";
    }

    // Validate GPU settings
    for (int32_t device_id : config.gpu.device_ids) {
        if (device_id < 0) {
            return "Invalid device_id: " + std::to_string(device_id);
        }
    }

    return "";  // Valid
}

ServerConfiguration load_config(const std::string& config_path, bool& success) {
    ServerConfiguration config;
    success = true;

    if (config_path.empty()) {
        // No config file specified, use defaults
        return config;
    }

    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file: " << config_path
                  << ". Using default configuration." << std::endl;
        return config;
    }

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

        // Parse KV cache config
        if (j.contains("kv_cache")) {
            auto& kv = j["kv_cache"];
            config.kv_cache.max_total_memory = kv.value("max_memory_bytes",
                config.kv_cache.max_total_memory);
            if (kv.contains("device_ids")) {
                config.kv_cache.device_ids = kv["device_ids"].get<std::vector<int32_t>>();
            }
        }

        // Parse logging config
        if (j.contains("logging")) {
            auto& log = j["logging"];
            config.logging.level = log.value("level", config.logging.level);
            config.logging.output = log.value("output", config.logging.output);
            config.logging.json_format = log.value("json_format", config.logging.json_format);
        }

    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "Error: Invalid JSON in config file: " << e.what() << std::endl;
        success = false;
        return config;
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to parse config file: " << e.what() << std::endl;
        success = false;
        return config;
    }

    return config;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS] [config_file]\n"
              << "\n"
              << "Windows-native inference server for Qwen2.5-32B-Instruct\n"
              << "\n"
              << "Options:\n"
              << "  --help, -h     Show this help message\n"
              << "  --version, -v  Show version information\n"
              << "\n"
              << "Arguments:\n"
              << "  config_file    Path to JSON configuration file (optional)\n"
              << "\n"
              << "Environment variables:\n"
              << "  QWEN_CONFIG    Path to configuration file\n"
              << "  QWEN_HOST      Server host (default: 0.0.0.0)\n"
              << "  QWEN_PORT      Server port (default: 8080)\n"
              << std::endl;
}

void print_version() {
    std::cout << "qwen-inference-server version 0.1.0\n"
              << "Target model: Qwen2.5-32B-Instruct\n"
              << "Built for Windows with CUDA support\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string config_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--version" || arg == "-v") {
            print_version();
            return 0;
        } else if (arg[0] != '-') {
            config_path = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Check environment variable for config path
    if (config_path.empty()) {
        const char* env_config = std::getenv("QWEN_CONFIG");
        if (env_config) {
            config_path = env_config;
        }
    }

    // Load configuration
    bool config_loaded;
    auto config = load_config(config_path, config_loaded);
    if (!config_loaded) {
        return 1;
    }

    // Override with environment variables
    const char* env_host = std::getenv("QWEN_HOST");
    if (env_host) {
        config.api.host = env_host;
    }
    const char* env_port = std::getenv("QWEN_PORT");
    if (env_port) {
        try {
            config.api.port = std::stoi(env_port);
        } catch (...) {
            std::cerr << "Invalid QWEN_PORT value: " << env_port << std::endl;
            return 1;
        }
    }

    // Validate configuration
    std::string validation_error = validate_config(config);
    if (!validation_error.empty()) {
        std::cerr << "Configuration error: " << validation_error << std::endl;
        return 1;
    }

    // Initialize logging first
    qwen::init_logger(config.logging);
    qwen::log_info("main", "starting", {
        {"version", "0.1.0"},
        {"config_file", config_path.empty() ? "(defaults)" : config_path}
    });

    // Set up signal handlers for graceful shutdown
    setup_signal_handlers();

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

        qwen::log_info("main", "gpu_initialized", {
            {"num_devices", device_manager->num_devices()},
            {"has_nvlink", device_manager->has_nvlink()}
        });

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

        // Initialize KV cache allocator with device IDs from GPU manager
        if (config.kv_cache.device_ids.empty()) {
            for (const auto& device : device_manager->devices()) {
                config.kv_cache.device_ids.push_back(device.device_id);
            }
        }
        auto kv_allocator = qwen::kvcache::create_contiguous_allocator();
        auto kv_result = kv_allocator->initialize(config.kv_cache);
        if (kv_result.is_error()) {
            qwen::log_error("main", "kv_cache_init_failed", {
                {"error", kv_result.error().message}
            });
            return 1;
        }

        qwen::log_info("main", "kv_cache_initialized", {
            {"total_bytes", kv_allocator->stats().total_bytes},
            {"num_devices", config.kv_cache.device_ids.size()}
        });

        // Initialize GPU router for multi-GPU scheduling
        auto gpu_router = std::make_shared<qwen::gpu::GPURouter>(device_manager);
        // Calculate KV budget per GPU (total memory / num devices for TP)
        size_t kv_budget_per_gpu = kv_allocator->stats().total_bytes / device_manager->num_devices();
        gpu_router->initialize_budgets(kv_budget_per_gpu);

        // Initialize backend (use null backend for now - replace with TensorRT in production)
        auto backend = std::make_shared<qwen::backend::NullBackend>();
        auto backend_result = backend->initialize(config.model);
        if (backend_result.is_error()) {
            qwen::log_error("main", "backend_init_failed", {
                {"error", backend_result.error().message}
            });
            return 1;
        }
        backend->load_model();

        qwen::log_info("main", "backend_initialized", {
            {"model_path", config.model.model_path},
            {"max_seq_len", config.model.max_seq_len},
            {"tensor_parallel", config.model.tensor_parallel}
        });

        // Create scheduler
        auto scheduler = std::make_shared<qwen::scheduler::Scheduler>(
            backend,
            kv_allocator,
            tokenizer,
            config.scheduler
        );

        // Wire in GPU router
        scheduler->set_gpu_router(gpu_router);

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
            {"model", config.api.model_name},
            {"endpoints", std::vector<std::string>{
                "/v1/chat/completions",
                "/v1/models",
                "/health",
                "/metrics"
            }}
        });

        std::cout << "Server ready at http://" << config.api.host << ":"
                  << config.api.port << std::endl;
        std::cout << "Press Ctrl+C to shutdown" << std::endl;

        // Wait for shutdown signal
        while (!g_shutdown_requested.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        qwen::log_info("main", "shutting_down", {});
        std::cout << "\nShutting down..." << std::endl;

        // Graceful shutdown
        server.stop();
        scheduler->stop();
        backend->unload_model();

        qwen::log_info("main", "shutdown_complete", {});

    } catch (const std::exception& e) {
        qwen::log_error("main", "fatal_error", {{"error", e.what()}});
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
