# Performance Optimization Plan

## Context

- **Hardware**: 2× NVIDIA L40S (48GB each)
- **Model**: Qwen2.5-32B-Instruct
- **Target Context**: 50-70k tokens
- **Current Implementation**: MVP (contiguous KV, basic scheduling)

---

## 1. Top 10 Expected Bottlenecks

### 1.1 KV Cache Memory Bandwidth
**Measurement**: Nsight Systems memory throughput trace
**Instrumentation**: CUDA events around attention operations
**Impact**: Critical for 70k context

### 1.2 Prefill Computation
**Measurement**: GPU kernel time for prefill
**Instrumentation**: `ScopedTimer` around `runtime->prefill()`
**Impact**: Time to first token (TTFT)

### 1.3 Attention Memory Access Pattern
**Measurement**: L2 cache hit ratio in Nsight Compute
**Instrumentation**: Profile attention kernels
**Impact**: Decode step latency

### 1.4 Host-Device Transfer Overhead
**Measurement**: CUDA stream timeline
**Instrumentation**: Async copy profiling
**Impact**: Token streaming latency

### 1.5 Tokenization Latency
**Measurement**: `bench_tokenizer` results
**Instrumentation**: `ScopedTimer` in `prompt_builder->build()`
**Impact**: Request processing time

### 1.6 Scheduler Queue Contention
**Measurement**: `queue_time_ms` metric
**Instrumentation**: Built-in timing
**Impact**: Request latency at high concurrency

### 1.7 KV Cache Allocation/Deallocation
**Measurement**: `bench_kv_cache` results
**Instrumentation**: Allocation timing
**Impact**: Request startup time

### 1.8 Tensor Parallel Communication
**Measurement**: NCCL profiling
**Instrumentation**: NVLink bandwidth monitoring
**Impact**: Per-token decode time with TP

### 1.9 HTTP/SSE Overhead
**Measurement**: `bench_end_to_end` comparison
**Instrumentation**: Network stack timing
**Impact**: Streaming latency

### 1.10 Python/Native Interop (if applicable)
**Measurement**: FFI call latency
**Instrumentation**: Boundary profiling
**Impact**: End-to-end latency

---

## 2. Steps to Reach vLLM-Class Performance

### 2.1 Continuous Batching

**Current State**: Single-request decode loop
**Target**: Dynamic batching with slot-based scheduling

**Implementation Steps**:
1. Implement batch decode interface in runtime
2. Create BatchScheduler with per-slot tracking
3. Implement dynamic batch resizing
4. Add iteration-level scheduling (add/remove requests mid-batch)
5. Optimize memory layout for batched operations

**Expected Improvement**: 3-5× throughput at high concurrency

### 2.2 Paged KV Cache

**Current State**: Contiguous per-request allocation
**Target**: Block-based allocation with prefix sharing

**Implementation Steps**:
1. Define block size (e.g., 16 tokens)
2. Implement physical block pool
3. Implement logical-to-physical page table
4. Add reference counting for shared prefixes
5. Implement LRU eviction policy
6. Optimize block copy operations

**Data Structures**:
```cpp
struct KVBlock {
    void* data;           // [2, kv_heads/tp, block_size, head_dim]
    uint32_t ref_count;
    uint64_t last_access;
};

struct PageTable {
    std::vector<uint32_t> block_indices;  // Per-sequence mapping
};
```

**Expected Improvement**: 20-30% memory efficiency, enables longer context at same memory

### 2.3 Split Prefill/Decode Lanes

**Current State**: Sequential prefill then decode
**Target**: Parallel prefill and decode on different resources

**Implementation Steps**:
1. Separate prefill and decode CUDA streams
2. Implement chunked prefill (process prefill in chunks)
3. Allow decode iterations between prefill chunks
4. Optimize GPU utilization balance

**Expected Improvement**: 2× throughput for mixed workloads

### 2.4 GPU-Side Sampling/Logits Transforms

**Current State**: Logits copied to CPU for sampling
**Target**: Sampling on GPU, only return token ID

**Implementation Steps**:
1. Implement temperature/top_p/top_k on GPU (curand)
2. Implement argmax on GPU
3. Add repetition penalty kernel
4. Minimize D2H transfer (just token ID)

**Expected Improvement**: 10-20% latency reduction for decode step

### 2.5 Kernel Fusion Opportunities

**Target Fusions**:
1. **RMSNorm + Activation**: Fuse layer norm with activation
2. **Attention Score + Mask + Softmax**: Single kernel
3. **FFN (up + gate + down)**: Fused SwiGLU

**Expected Improvement**: 10-15% overall latency reduction

---

## 3. Benchmarks and Targets

### 3.1 Latency Targets

| Metric | MVP Target | V2 Target | vLLM Reference |
|--------|------------|-----------|----------------|
| TTFT (1k prompt) | < 200ms | < 100ms | ~80ms |
| TTFT (10k prompt) | < 2s | < 500ms | ~400ms |
| TTFT (50k prompt) | < 10s | < 2s | ~1.5s |
| Decode latency/token | < 50ms | < 20ms | ~15ms |

### 3.2 Throughput Targets

| Concurrency | MVP Target (tok/s) | V2 Target (tok/s) | vLLM Reference |
|-------------|--------------------|-------------------|----------------|
| 1 | 50 | 80 | 100 |
| 4 | 100 | 300 | 400 |
| 8 | 150 | 500 | 700 |
| 16 | 200 | 800 | 1000+ |

### 3.3 Context Length Scaling

| Context | Max Concurrent | Memory per Request | TTFT Target |
|---------|----------------|-------------------|-------------|
| 10k | 5-6 | 2.5 GB | < 300ms |
| 20k | 2-3 | 5 GB | < 600ms |
| 50k | 1 | 12.5 GB | < 1.5s |
| 70k | 1 | 17.5 GB | < 2s |

---

## 4. Profiling Workflow on Windows

### 4.1 Nsight Systems

**Trace Collection**:
```powershell
nsys profile --trace=cuda,nvtx,osrt ^
  --output=profile_report ^
  qwen_server.exe --config config.json
```

**Key Views**:
- GPU kernel timeline
- CUDA API calls
- Memory copy operations
- CPU/GPU synchronization points

**What to Look For**:
- Gaps between kernels (CPU bound)
- Long kernels (optimization targets)
- Memory copy bottlenecks
- Stream synchronization waits

### 4.2 Nsight Compute

**Kernel Profiling**:
```powershell
ncu --set detailed ^
  --kernel-name regex:attention ^
  --launch-count 10 ^
  qwen_server.exe
```

**Key Metrics**:
- Memory throughput (% of peak)
- Compute throughput (% of peak)
- L2 cache hit rate
- Occupancy

### 4.3 Custom Instrumentation

**NVTX Markers**:
```cpp
#include <nvtx3/nvtx3.hpp>

void prefill(...) {
    nvtx3::scoped_range range{"prefill"};
    // ...
}
```

**Metrics Collection**:
```cpp
struct PerfCounters {
    std::atomic<uint64_t> prefill_us;
    std::atomic<uint64_t> decode_us;
    std::atomic<uint64_t> kv_alloc_us;
    std::atomic<uint64_t> tokens_generated;
};
```

### 4.4 Recommended Traces

1. **Baseline trace**: Normal operation, 10 requests
2. **High concurrency trace**: 8+ concurrent requests
3. **Long context trace**: 50k+ token request
4. **Memory pressure trace**: Near-OOM conditions
5. **Cold start trace**: First requests after startup

---

## 5. Prioritized Backlog

### P0 - Critical (MVP)

| Issue | Estimate | Risk | Description |
|-------|----------|------|-------------|
| TensorRT engine integration | 2 weeks | High | Replace null backend with real inference |
| Prefill optimization | 1 week | Medium | Chunked prefill, overlap with decode |
| Memory limit enforcement | 3 days | Low | Strict OOM prevention |

### P1 - High (V1 Performance)

| Issue | Estimate | Risk | Description |
|-------|----------|------|-------------|
| GPU-side sampling | 1 week | Medium | Eliminate D2H for logits |
| Continuous batching | 2 weeks | High | Core throughput improvement |
| Tensor parallel optimization | 1 week | Medium | Reduce AllReduce overhead |

### P2 - Medium (V2 Performance)

| Issue | Estimate | Risk | Description |
|-------|----------|------|-------------|
| Paged KV cache | 3 weeks | High | Complex but high value |
| Kernel fusion | 2 weeks | Medium | Custom CUDA kernels |
| Speculative decoding | 2 weeks | Medium | Latency optimization |

### P3 - Low (Nice to Have)

| Issue | Estimate | Risk | Description |
|-------|----------|------|-------------|
| Prefix caching | 1 week | Low | RAG optimization |
| Multi-LoRA support | 2 weeks | Medium | Fine-tuned variants |
| INT8/FP8 quantization | 3 weeks | High | Memory/throughput trade-off |

---

## 6. Performance Testing Checklist

### Before Each Release

- [ ] Run `bench_tokenizer` - verify no regressions
- [ ] Run `bench_kv_cache` - verify allocation performance
- [ ] Run `bench_scheduler` - verify concurrency handling
- [ ] Run `bench_end_to_end` - verify latency targets
- [ ] Collect Nsight Systems trace for 100 requests
- [ ] Verify memory usage stays within bounds
- [ ] Test 70k context end-to-end

### Weekly Performance Check

- [ ] Compare latency p95/p99 against baseline
- [ ] Compare throughput against baseline
- [ ] Review memory high-water mark
- [ ] Check for memory leaks (24h soak test)
- [ ] Profile any new code paths

---

## 7. Performance Monitoring in Production

### Key Metrics to Export

```
# Latency
inference_time_to_first_token_seconds{quantile="0.5"}
inference_time_to_first_token_seconds{quantile="0.95"}
inference_time_to_first_token_seconds{quantile="0.99"}
inference_decode_latency_per_token_seconds

# Throughput
inference_tokens_per_second
inference_requests_per_second

# Resource Usage
inference_kv_cache_utilization_ratio
inference_gpu_memory_used_bytes
inference_gpu_compute_utilization

# Queue Health
inference_queue_depth
inference_queue_wait_seconds
inference_rejected_requests_total
```

### Alerting Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| TTFT p99 | > 2× baseline | > 5× baseline |
| Decode latency p99 | > 2× baseline | > 5× baseline |
| Queue depth | > 50 | > 100 |
| Memory usage | > 80% | > 95% |
| Error rate | > 1% | > 5% |
