#include "utils/result.hpp"
#include "utils/logger.hpp"

#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace qwen::backend::cuda {

/// CUDA error checking helper
inline void check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        log_error("cuda", "error", {
            {"error", cudaGetErrorString(err)},
            {"file", file},
            {"line", line}
        });
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}

#define CUDA_CHECK(err) check_cuda(err, __FILE__, __LINE__)

/// RAII wrapper for CUDA memory
template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    ~CudaBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Move only
    CudaBuffer(CudaBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

    void copy_from_host(const T* src, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* dst, size_t count) const {
        CUDA_CHECK(cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

/// RAII wrapper for CUDA stream
class CudaStream {
public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    explicit CudaStream(unsigned int flags) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }

    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() { return stream_; }

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    cudaStream_t stream_ = nullptr;
};

/// RAII wrapper for CUDA event
class CudaEvent {
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    explicit CudaEvent(unsigned int flags) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }

    ~CudaEvent() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) cudaEventDestroy(event_);
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() { return event_; }

    void record(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    float elapsed_ms(const CudaEvent& start) const {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }

private:
    cudaEvent_t event_ = nullptr;
};

/// Get current device ID
inline int get_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

/// Set current device
inline void set_device(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

/// RAII device setter
class DeviceGuard {
public:
    explicit DeviceGuard(int device) : prev_device_(get_device()) {
        set_device(device);
    }

    ~DeviceGuard() {
        cudaSetDevice(prev_device_);
    }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int prev_device_;
};

}  // namespace qwen::backend::cuda
