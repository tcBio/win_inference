cmake_minimum_required(VERSION 3.20)
project(qwen_inference_server VERSION 0.1.0 LANGUAGES CXX CUDA)

# C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Windows-specific settings
if(WIN32)
    add_definitions(-D_WIN32_WINNT=0x0A00)  # Windows 10+
    add_definitions(-DNOMINMAX)             # Prevent Windows min/max macros
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    # MSVC flags
    if(MSVC)
        add_compile_options(/W4 /WX /permissive- /Zc:__cplusplus)
        add_compile_options(/utf-8)
    endif()
endif()

# Linux settings
if(UNIX AND NOT APPLE)
    add_compile_options(-Wall -Wextra -Werror -Wpedantic)
endif()

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_TENSORRT "Enable TensorRT backend" OFF)
option(ENABLE_METRICS "Enable Prometheus metrics" ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find TensorRT (optional)
if(ENABLE_TENSORRT)
    find_package(TensorRT QUIET)
    if(TensorRT_FOUND)
        add_definitions(-DHAS_TENSORRT)
    else()
        message(WARNING "TensorRT not found, building without TensorRT backend")
        set(ENABLE_TENSORRT OFF)
    endif()
endif()

# Third-party dependencies (header-only)
include(FetchContent)

# nlohmann/json for JSON parsing
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# cpp-httplib for HTTP server
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.15.3
)
FetchContent_MakeAvailable(httplib)

# spdlog for logging
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(spdlog)

# GoogleTest for testing
if(BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # Prevent overriding parent project's compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source subdirectories
add_subdirectory(src)

# Tests
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "=== Qwen Inference Server Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA found: ${CUDAToolkit_FOUND}")
message(STATUS "TensorRT enabled: ${ENABLE_TENSORRT}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "============================================")
message(STATUS "")
