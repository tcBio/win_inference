# Qwen2.5-32B-Instruct Windows Inference Server - Engineering Specification

## Executive Summary

This document specifies a Windows-native, high-performance inference server for Qwen2.5-32B-Instruct targeting 2× NVIDIA L40S GPUs (48GB each) with 50-70k token context for RAG workloads.

**Priority Order:**
1. Correctness/Accuracy (prompt formatting, logits, stop behavior, deterministic outputs)
2. Stability/Observability (structured logging, metrics, tracing)
3. Performance (throughput and tail latency)

---

## 1. System Architecture

### 1.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           HTTP Server Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Route       │  │ SSE         │  │ Request     │  │ Metrics     │    │
│  │ Handlers    │  │ Streamer    │  │ Validator   │  │ Endpoint    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Core Services Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Prompt      │  │ Tokenizer   │  │ Sampler     │  │ Stop        │    │
│  │ Builder     │  │             │  │             │  │ Checker     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Scheduler Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Request     │  │ Admission   │  │ Decode      │  │ Cancellation│    │
│  │ Registry    │  │ Controller  │  │ Loop        │  │ Manager     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Memory Management Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │ KV Cache    │  │ Memory      │  │ GPU Memory  │                      │
│  │ Allocator   │  │ Budget      │  │ Pool        │                      │
│  └─────────────┘  └─────────────┘  └─────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           GPU Management Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │ Device      │  │ GPU Router  │  │ Tensor      │                      │
│  │ Manager     │  │             │  │ Parallel    │                      │
│  └─────────────┘  └─────────────┘  └─────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Backend Adapter Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │ Runtime     │  │ TensorRT    │  │ Null        │                      │
│  │ Interface   │  │ Backend     │  │ Backend     │                      │
│  └─────────────┘  └─────────────┘  └─────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Module Responsibilities

| Module | Responsibility | Key Files |
|--------|---------------|-----------|
| `api/` | HTTP server, SSE streaming, request validation | `server.cpp`, `routes.cpp`, `sse_streamer.cpp` |
| `core/` | Prompt building, sampling, stop checking | `prompt_builder.cpp`, `sampler.cpp`, `stop_checker.cpp` |
| `tokenizer/` | BPE tokenization, Qwen vocab | `tokenizer.cpp`, `bpe.cpp`, `vocab.cpp` |
| `scheduler/` | Request lifecycle, decode loop, cancellation | `scheduler.cpp`, `request_registry.cpp`, `decode_loop.cpp` |
| `kvcache/` | KV cache allocation, memory budgeting | `kv_allocator.cpp`, `memory_budget.cpp` |
| `gpu/` | Device management, multi-GPU routing | `device_manager.cpp`, `gpu_router.cpp` |
| `backend/` | Model runtime abstraction, TensorRT integration | `runtime_interface.cpp`, `tensorrt_backend.cpp` |
| `utils/` | Logging, metrics, error handling | `logger.cpp`, `metrics.cpp`, `result.cpp` |

---

## 2. Request Lifecycle

### 2.1 State Machine

```
                    ┌──────────────┐
                    │   RECEIVED   │
                    └──────┬───────┘
                           │ validate
                           ▼
                    ┌──────────────┐
              ┌─────│    QUEUED    │─────┐
              │     └──────┬───────┘     │
              │            │ admit       │ reject (memory)
              │            ▼             ▼
              │     ┌──────────────┐  ┌──────────────┐
              │     │  PREFILLING  │  │   REJECTED   │
              │     └──────┬───────┘  └──────────────┘
              │            │ complete
              │            ▼
              │     ┌──────────────┐
    cancel ───┼─────│   DECODING   │─────┐
              │     └──────┬───────┘     │
              │            │             │ error
              │            │ finish      ▼
              │            │      ┌──────────────┐
              │            │      │   ERRORED    │
              │            ▼      └──────────────┘
              │     ┌──────────────┐
              └────▶│  CANCELLED   │
                    └──────────────┘
                    ┌──────────────┐
                    │  COMPLETED   │
                    └──────────────┘
```

### 2.2 Execution Flow

1. **Receive**: HTTP layer receives `/v1/chat/completions` request
2. **Validate**: Schema validation, model check, parameter bounds
3. **Build Prompt**: Convert messages to Qwen chat template, tokenize
4. **Admit**: Check memory budget, allocate KV cache or reject
5. **Queue**: Add to scheduler queue with priority
6. **Route GPU**: Select GPU(s) based on context length and availability
7. **Prefill**: Run prefill pass, populate KV cache
8. **Decode Loop**:
   - Run decode step
   - Sample next token
   - Check stop conditions
   - Stream token delta via SSE
   - Repeat until done
9. **Complete**: Send final SSE event, free KV cache, update metrics

### 2.3 Streaming Protocol

```
Client Request (POST /v1/chat/completions with stream=true)
    │
    ▼
HTTP 200 OK
Content-Type: text/event-stream
    │
    ├──▶ data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}
    │
    ├──▶ data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}
    │
    ├──▶ data: {"id":"...","choices":[{"delta":{"content":" world"}}]}
    │
    ├──▶ data: {"id":"...","choices":[{"delta":{},"finish_reason":"stop"}]}
    │
    └──▶ data: [DONE]
```

### 2.4 Cancellation Flow

- Client disconnect detected via socket monitoring
- CancellationToken signaled atomically
- Decode loop checks token each iteration
- KV cache freed immediately on cancellation
- Metrics record cancellation reason and tokens generated

---

## 3. OpenAI-Compatible API

### 3.1 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming and non-streaming) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### 3.2 Request Schema (Chat Completions)

```json
{
  "model": "qwen2.5-32b-instruct",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 2048,
  "stop": ["<|endoftext|>"],
  "stream": true,
  "seed": 42,
  "tools": []  // passthrough only
}
```

### 3.3 Response Schema (Non-Streaming)

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "qwen2.5-32b-instruct",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

---

## 4. KV Cache Strategy

### 4.1 Memory Math for Qwen2.5-32B

**Model Parameters:**
- Layers: 64
- Hidden dimension: 5120
- Num attention heads: 40
- Num KV heads (GQA): 8
- Head dimension: 128 (5120 / 40)
- Dtype: FP16 (2 bytes)

**KV Cache per Token:**
```
KV_per_token = 2 × layers × kv_heads × head_dim × dtype_bytes
             = 2 × 64 × 8 × 128 × 2
             = 262,144 bytes
             = 256 KB per token
```

**KV Cache for 70k Context:**
```
KV_70k = 70,000 × 256 KB = 17.5 GB per request
```

**Model Weights (FP16):**
```
~32B params × 2 bytes = ~64 GB (split across 2 GPUs = 32 GB each)
```

**Memory Budget per L40S (48 GB):**
```
Available = 48 GB
- Weights = 32 GB
- Workspace = 2 GB
- Available for KV = 14 GB
```

**Max Concurrent Requests at 70k:**
```
With tensor parallel (KV split across 2 GPUs):
  Per-GPU KV = 8.75 GB
  Concurrent requests = floor(14 GB / 8.75 GB) = 1 request at 70k

With tensor parallel, shorter contexts:
  10k context = 2.5 GB KV → ~5 concurrent
  20k context = 5 GB KV → ~2-3 concurrent
```

### 4.2 MVP: Contiguous KV Allocation

```cpp
struct KVCacheAllocation {
    void* k_cache;      // [layers, kv_heads, max_seq_len, head_dim]
    void* v_cache;      // [layers, kv_heads, max_seq_len, head_dim]
    size_t max_seq_len;
    size_t current_len;
    int device_id;
};
```

**MVP Characteristics:**
- Pre-allocate for max_tokens at admission time
- No fragmentation, simple addressing
- Memory waste for variable-length sequences

### 4.3 V2: Paged KV Cache (Future)

**Block Size:** 16 tokens per block
**Page Table:** Maps logical positions to physical blocks
**Reuse Policy:** LRU eviction, prefix sharing for common prompts

```cpp
struct KVBlock {
    void* data;         // [2, kv_heads, block_size, head_dim]
    uint32_t ref_count;
    uint64_t last_access;
};

struct PagedKVCache {
    std::vector<KVBlock*> physical_blocks;
    std::unordered_map<uint64_t, std::vector<uint32_t>> page_tables;
};
```

---

## 5. Multi-GPU Strategy

### 5.1 Recommendation: Tensor Parallel

**Justification:**
- 70k context requires 17.5 GB KV cache
- Single GPU cannot fit weights (32 GB) + KV (17.5 GB) for long context
- Tensor parallel splits both weights and KV across GPUs
- Enables single-request handling for full 70k context

### 5.2 Implementation

```
GPU 0                           GPU 1
┌─────────────────────┐        ┌─────────────────────┐
│ Weights (shard 0)   │        │ Weights (shard 1)   │
│ KV Cache (heads 0-3)│◄──────▶│ KV Cache (heads 4-7)│
│                     │ NVLink │                     │
└─────────────────────┘        └─────────────────────┘
```

**Communication Pattern:**
- All-reduce for attention output aggregation
- All-reduce for FFN intermediate
- NVLink bandwidth: 600 GB/s (sufficient for 70k)

### 5.3 Fallback Strategy

For shorter requests where memory allows:
- Requests < 20k: Can run on single GPU
- Route to least-loaded GPU
- Config flag: `gpu_strategy = "tensor_parallel" | "per_request"`

---

## 6. Build & Packaging

### 6.1 CMake Structure

```
CMakeLists.txt
├── cmake/
│   ├── FindTensorRT.cmake
│   ├── CUDASetup.cmake
│   └── WindowsMSVC.cmake
├── src/
│   └── CMakeLists.txt (per-module)
├── tests/
│   └── CMakeLists.txt
└── benchmarks/
    └── CMakeLists.txt
```

### 6.2 Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| CUDA | 12.x | GPU compute |
| TensorRT | 10.x | Inference backend |
| cuDNN | 9.x | Deep learning primitives |
| httplib | header-only | HTTP server |
| nlohmann/json | header-only | JSON parsing |
| spdlog | 1.12+ | Logging |
| prometheus-cpp | optional | Metrics export |

### 6.3 Build Commands

```powershell
# Configure
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"

# Build
cmake --build build --config Release --parallel

# Test
ctest --test-dir build -C Release --output-on-failure
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

| Component | Test Focus |
|-----------|------------|
| Tokenizer | Encode/decode roundtrip, special tokens, edge cases |
| PromptBuilder | Chat template correctness, role handling |
| Scheduler | State transitions, cancellation, queue ordering |
| KVAllocator | Alloc/free, bounds checking, memory limits |
| Sampler | Temperature, top_p, determinism |
| StopChecker | Stop sequences, max_tokens, EOS |

### 7.2 Golden Tests

- **Chat Template:** Fixed messages → exact prompt string
- **Tokenization:** Fixed prompts → exact token IDs
- **Stop Behavior:** Verify stop sequence matching
- **Determinism:** Same seed → same output tokens

### 7.3 Integration Tests

- End-to-end request flow with mock backend
- SSE streaming correctness
- Cancellation cleanup
- Memory pressure handling

### 7.4 Load Tests

- Concurrent request handling
- Memory limit enforcement
- Queue backpressure behavior

---

## 8. Benchmark Plan

### 8.1 Metrics

| Metric | Target (MVP) | Target (V2) |
|--------|--------------|-------------|
| p50 latency (first token) | < 500ms | < 200ms |
| p95 latency (first token) | < 1s | < 400ms |
| p99 latency (first token) | < 2s | < 800ms |
| Throughput (tokens/sec) | 50 @ concurrency 1 | 200 @ concurrency 4 |
| Max context length | 70k | 70k |

### 8.2 Test Scenarios

1. **Single Request Latency:** 1k/10k/50k/70k prompt lengths
2. **Throughput:** Concurrent requests at 1/4/8/16 levels
3. **Long Generation:** 4k output tokens sustained
4. **Mixed Load:** Variable prompt/output lengths

### 8.3 Measurement Points

```cpp
struct RequestMetrics {
    Duration queue_time;
    Duration prefill_time;
    Duration total_decode_time;
    Duration time_to_first_token;
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    float tokens_per_second;
};
```

---

## 9. Observability

### 9.1 Structured Logging

```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "component": "scheduler",
  "request_id": "req_abc123",
  "event": "request_completed",
  "prompt_tokens": 1024,
  "completion_tokens": 256,
  "latency_ms": 1500,
  "gpu_id": 0
}
```

### 9.2 Metrics (Prometheus Format)

```
# Request metrics
inference_requests_total{status="completed"} 1234
inference_requests_active 5
inference_queue_depth 3

# Latency histograms
inference_time_to_first_token_seconds_bucket{le="0.1"} 100
inference_decode_time_seconds_bucket{le="1.0"} 500

# Resource metrics
inference_kv_cache_used_bytes 1073741824
inference_kv_cache_total_bytes 15032385536
inference_gpu_memory_used_bytes{device="0"} 40000000000
```

### 9.3 Tracing Fields

Every log/metric includes: `request_id`, `trace_id`, `span_id`

---

## 10. Implementation Phases

### Phase 0: Foundation (Week 1-2)

**Deliverables:**
- Project structure and build system
- Core interfaces defined
- Tokenizer with golden tests
- Prompt builder with golden tests
- Null backend for testing

**Acceptance Criteria:**
- [ ] CMake builds on Windows with MSVC
- [ ] Tokenizer encodes/decodes Qwen vocab correctly
- [ ] Prompt builder passes all golden tests
- [ ] Unit tests pass in CI

### Phase 1: MVP Server (Week 3-4)

**Deliverables:**
- HTTP server with SSE streaming
- Scheduler with basic queue
- Contiguous KV cache allocator
- Single-GPU execution path
- TensorRT backend stub

**Acceptance Criteria:**
- [ ] `/v1/chat/completions` works (non-streaming)
- [ ] SSE streaming works
- [ ] Cancellation frees resources
- [ ] Memory limits enforced
- [ ] Integration tests pass

### Phase 2: Production Ready (Week 5-6)

**Deliverables:**
- Tensor parallel for 2 GPUs
- TensorRT backend integration
- Full metrics and logging
- Load testing passed
- Performance benchmarks documented

**Acceptance Criteria:**
- [ ] 70k context works with tensor parallel
- [ ] p95 latency targets met
- [ ] Throughput targets met
- [ ] 24-hour stability test passes
- [ ] Golden test suite passes

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TensorRT Windows compatibility issues | Medium | High | Early prototype, fallback to CUDA kernels |
| 70k context OOM | High | High | Strict admission control, dynamic max context |
| Chat template mismatch | Medium | High | Extensive golden tests, reference comparison |
| NVLink bandwidth insufficient | Low | Medium | Profile early, optimize communication |
| Tokenizer correctness | Medium | Critical | Use reference tokenizer for validation |
| SSE connection handling bugs | Medium | Medium | Comprehensive disconnect testing |
| MSVC C++20 compatibility | Low | Medium | Test early, fallback to C++17 if needed |

---

## 12. Assumptions and Unknowns

### Assumptions
- TensorRT 10.x supports Windows deployment
- Model weights available in TensorRT engine format
- NVLink available between L40S GPUs
- CUDA 12.x driver installed

### Unknowns (To Investigate)
- [ ] Exact TensorRT-LLM Windows support status
- [ ] Optimal block size for paged KV cache
- [ ] Best tensor parallel communication pattern
- [ ] Flash attention kernel availability on Windows

---

## Appendix A: Directory Structure

```
win_inference/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── ENGINEERING_SPEC.md
│   ├── API_SPEC.md
│   └── PERFORMANCE.md
├── configs/
│   └── server_config.json
├── include/
│   ├── api/
│   │   ├── server.hpp
│   │   ├── routes.hpp
│   │   └── sse_streamer.hpp
│   ├── core/
│   │   ├── prompt_builder.hpp
│   │   ├── sampler.hpp
│   │   └── stop_checker.hpp
│   ├── tokenizer/
│   │   ├── tokenizer.hpp
│   │   └── bpe.hpp
│   ├── scheduler/
│   │   ├── scheduler.hpp
│   │   ├── request.hpp
│   │   └── decode_loop.hpp
│   ├── kvcache/
│   │   ├── kv_allocator.hpp
│   │   └── memory_budget.hpp
│   ├── gpu/
│   │   ├── device_manager.hpp
│   │   └── gpu_router.hpp
│   ├── backend/
│   │   ├── runtime_interface.hpp
│   │   ├── tensorrt_backend.hpp
│   │   └── null_backend.hpp
│   └── utils/
│       ├── logger.hpp
│       ├── metrics.hpp
│       ├── result.hpp
│       └── cancellation.hpp
├── src/
│   └── [mirrors include structure]
├── tests/
│   ├── unit/
│   ├── integration/
│   └── golden/
├── benchmarks/
└── scripts/
    ├── build.ps1
    └── run_tests.ps1
```
