# Core utilities library
add_library(qwen_utils STATIC
    utils/logger.cpp
    utils/metrics.cpp
)
target_link_libraries(qwen_utils PUBLIC spdlog::spdlog nlohmann_json::nlohmann_json)
target_include_directories(qwen_utils PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Tokenizer library
add_library(qwen_tokenizer STATIC
    tokenizer/tokenizer.cpp
    tokenizer/bpe.cpp
    tokenizer/prompt_builder.cpp
)
target_link_libraries(qwen_tokenizer PUBLIC qwen_utils)

# KV Cache library
add_library(qwen_kvcache STATIC
    kvcache/kv_allocator.cpp
    kvcache/memory_budget.cpp
)
target_link_libraries(qwen_kvcache PUBLIC qwen_utils CUDA::cudart)

# GPU management library
add_library(qwen_gpu STATIC
    gpu/device_manager.cpp
    gpu/gpu_router.cpp
)
target_link_libraries(qwen_gpu PUBLIC qwen_utils CUDA::cudart)

# Backend library
add_library(qwen_backend STATIC
    backend/null_backend.cpp
    backend/cuda_utils.cpp
)
target_link_libraries(qwen_backend PUBLIC qwen_utils qwen_kvcache CUDA::cudart)
if(ENABLE_TENSORRT)
    target_sources(qwen_backend PRIVATE backend/tensorrt_backend.cpp)
    target_link_libraries(qwen_backend PUBLIC TensorRT::nvinfer)
endif()

# Scheduler library
add_library(qwen_scheduler STATIC
    scheduler/request.cpp
    scheduler/request_registry.cpp
    scheduler/scheduler.cpp
    scheduler/decode_loop.cpp
)
target_link_libraries(qwen_scheduler PUBLIC qwen_utils qwen_backend qwen_kvcache qwen_gpu)

# Core services library
add_library(qwen_core STATIC
    core/sampler.cpp
    core/stop_checker.cpp
)
target_link_libraries(qwen_core PUBLIC qwen_utils qwen_tokenizer)

# API library
add_library(qwen_api STATIC
    api/server.cpp
    api/routes.cpp
    api/sse_streamer.cpp
    api/request_validator.cpp
)
target_link_libraries(qwen_api PUBLIC
    qwen_utils
    qwen_core
    qwen_tokenizer
    qwen_scheduler
    httplib::httplib
)

# Main executable
add_executable(qwen_server main.cpp)
target_link_libraries(qwen_server PRIVATE qwen_api)

# Install targets
install(TARGETS qwen_server DESTINATION bin)
