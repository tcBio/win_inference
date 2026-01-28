# Qwen Inference Server - Implementation Prompt Pack

This document preserves the full prompt pack used to guide implementation of the Windows-native inference server.

---

## Prompt 0 — Global Constraints + Architecture Stance

```
You are a senior systems engineer building a Windows-native LLM inference server. We will implement an OpenAI-compatible API for Qwen2.5-32B-Instruct with streaming. Hardware: 2x NVIDIA L40S (48GB each). Target context length: 50k–70k tokens for RAG. Priority order: (1) correctness/accuracy (prompt formatting, logits, stop behavior, deterministic outputs), (2) stability/observability, (3) performance (throughput and tail latency).

Critical requirements:
- Windows-native build with MSVC + CMake.
- NVIDIA GPUs. Use CUDA. For the execution backend, assume TensorRT(-LLM) or a TensorRT/CUDA backend is used. If a component cannot be realistically Windows-native, explicitly design an interface so it can be replaced later, but keep the core server native.
- All code must be production quality: minimal dependencies, strict error handling, structured logging, metrics, clean interfaces, unit tests, and benchmarks.
- Provide code in small modules (<= 400 LOC per file) and propose a directory structure with clear ownership boundaries.
- Always include: (a) interfaces first, (b) tests, (c) performance considerations, (d) thread-safety notes.
- Never hand-wave: if a step is complex (e.g., paged KV cache), outline MVP then V2.
- We need streaming via SSE for /v1/chat/completions.
- We may not need embeddings; treat embeddings as optional and design to add later.

When you propose a design, provide:
1) Module list and responsibilities
2) Key data structures
3) Execution flow for a request (queue → prefill → decode → stream)
4) Backpressure/cancellation design
5) Memory budgeting approach for 70k context
6) Multi-GPU plan (tensor parallel vs pipeline vs split by requests)

Output must be concrete and actionable.
```

---

## Prompt 1 — Full Spec + Work Breakdown + Risk Register

```
Create a full engineering spec for a Windows-native inference server for Qwen2.5-32B-Instruct using 2x L40S GPUs, targeting 50–70k context. Accuracy first, streaming required.

Deliverables:
- System architecture (components, boundaries, interfaces)
- Detailed request lifecycle (including streaming and cancellation)
- Proposed OpenAI-compatible endpoints (minimal set)
- KV cache strategy: MVP (contiguous), V2 (block/paged). Include memory math for 70k context.
- Multi-GPU strategy: recommend one and justify (tensor parallel vs per-request GPU assignment vs pipeline parallel). Include constraints with 70k context.
- Build & packaging plan for Windows (MSVC+CMake)
- Testing strategy: unit tests + golden tests for chat templates + load tests
- Benchmark plan: p50/p95/p99 latency + throughput at concurrency levels
- Observability: structured logs + metrics + tracing fields
- A staged implementation plan with milestones (Phase 0/1/2)
- Risk register with mitigations

Be explicit about assumptions and unknowns. Provide acceptance criteria for each phase.
```

---

## Prompt 2 — API Contract + Streaming SSE + Payload Schemas

```
Design the OpenAI-compatible HTTP API layer for this server.

Requirements:
- Implement /v1/chat/completions with streaming (SSE) and non-streaming.
- Support minimal request fields: model, messages, temperature, top_p, max_tokens, stop, stream, seed.
- Support tool/function calling only as passthrough for now (don't implement full schema inference, but preserve correctness).
- Implement cancellation: if client disconnects, stop decoding quickly and free KV.

Output:
1) Precise JSON schemas for request/response (documented)
2) SSE event format used (tokens/deltas, finish_reason)
3) Error model (HTTP status + JSON error body)
4) A C++ interface for the model runtime that the HTTP layer will call (pure virtual).
5) A concurrency model for the server: threads, async IO, and how you avoid blocking.
6) A minimal code skeleton: CMake + main server + route handlers + unit tests for the API layer.

Keep code modular (<400 LOC per file). Use modern C++ (C++20).
```

---

## Prompt 3 — Tokenization + Qwen Chat Template + Golden Tests

```
Implement tokenization and prompt formatting for Qwen2.5-32B-Instruct.

Requirements:
- Provide a PromptBuilder that converts OpenAI-style messages into the exact Qwen chat template.
- Include stop/EOS handling behavior and how stop sequences are applied.
- Tokenizer must be correct and fast. If you rely on an external tokenizer lib, justify it and keep dependencies minimal. Otherwise implement a minimal BPE loader with tests.
- Provide golden tests: given fixed messages and known expected token IDs (or expected serialized prompt), verify deterministic output.
- Provide a test harness to validate that the prompt string matches the reference format exactly.

Output:
- Directory/module layout
- C++ code + tests
- Explanation of how this avoids "double templating" when clients already format prompts
- Performance notes and thread safety
```

---

## Prompt 4 — Runtime Interface + Scheduler State Machine (MVP)

```
Implement the core request scheduler for LLM inference.

Constraints:
- Accuracy > performance for MVP, but architecture must allow V2 continuous batching.
- Define request states: QUEUED, PREFILLING, DECODING, COMPLETED, CANCELLED, ERRORED.
- Provide a thread-safe RequestRegistry with cancellation tokens.
- Provide a DecodeLoop thread that processes active requests and emits token deltas to streaming clients.
- Provide backpressure: prevent unbounded queued requests; implement admission control.
- Provide metrics counters/timers for queue time, prefill time, decode time, tokens/sec, active requests.

Output:
1) C++ interfaces for ModelRuntime::Prefill and ModelRuntime::DecodeStep (or DecodeBatch)
2) Scheduler implementation with clear ownership boundaries
3) Unit tests for state transitions and cancellation correctness
4) A fake/mock runtime for tests (deterministic logits)
5) Notes on how to extend to continuous batching in V2
```

---

## Prompt 5 — Memory Budgeting for 70k Context + KV Cache MVP

```
Design and implement KV cache management.

Requirements:
- Provide explicit memory math for Qwen2.5-32B and 70k context.
- MVP KV cache: contiguous per-request KV allocation with safe bounds.
- Admission control: reject requests that would exceed GPU memory budget.
- Provide an interface KVCacheAllocator that can be swapped for a paged/block allocator later.
- Include a plan for V2 paged KV cache (data structures + page table + reuse policy).

Output:
- C++ code for KV cache allocator (MVP)
- Unit tests for allocation/free, cancellation cleanup, memory limits
- Benchmarks to measure allocation overhead
- A written spec for paged KV cache V2
```

---

## Prompt 6 — Multi-GPU Plan for 2× L40S

```
Given 2x L40S GPUs and Qwen2.5-32B-Instruct with 70k context, propose and implement a multi-GPU strategy.

Options to evaluate:
A) Tensor parallel across 2 GPUs for a single request (preferred for long context)
B) Assign whole requests to one GPU (better for many shorter requests)
C) Pipeline parallel

Requirements:
- Choose one primary strategy and justify based on long context (70k) and KV cache size.
- Provide code-level design: device manager, per-request GPU affinity, and how scheduler routes work.
- Include a fallback mode (if a request is too large for single GPU, route to TP).
- Expose config flags to control GPU strategy.

Output:
- Detailed design + module interfaces
- Pseudocode for request routing
- C++ skeleton code for GPU manager and runtime wrapper
- Testing approach (mock devices)
```

---

## Prompt 7 — TensorRT(-LLM) Backend Adapter

```
Implement a backend adapter layer that can run the model via TensorRT(-LLM) or a placeholder backend.

Requirements:
- Define a minimal runtime interface used by the scheduler: LoadModel, Prefill, DecodeBatch, GetVocabSize, GetEosTokenId, etc.
- Implement a "NullBackend" for tests.
- Implement a TensorRTBackend stub with correct resource lifecycle, but if full engine build is too involved, clearly mark TODOs and provide the exact integration points.
- Include CUDA stream management, workspace allocation, and error handling patterns on Windows.

Output:
- Code layout + interfaces
- C++ code stubs with real patterns (RAII wrappers for CUDA, TRT objects)
- Unit tests for resource lifecycle (at least compile/link tests)
- Notes on engine build pipeline assumptions and where it plugs in
```

---

## Prompt 8 — Streaming Implementation End-to-End (SSE)

```
Implement end-to-end streaming via SSE for /v1/chat/completions.

Requirements:
- Each token delta must stream promptly.
- Handle client disconnect: cancel request, free KV.
- Backpressure: if client is slow, prevent unlimited buffering (bounded queue, drop policy or slow down decode per-request).
- Ensure thread-safe write path and no blocking on scheduler threads.
- Include integration tests: simulate a client receiving SSE and disconnecting mid-stream.

Output:
- C++ server code + SSE streamer module
- Request → scheduler → streamer glue
- Tests + a small CLI test client
```

---

## Prompt 9 — Correctness Harness (Golden Outputs + Determinism)

```
Create a correctness test harness to validate the server's outputs.

Requirements:
- Deterministic mode: fixed seed, deterministic sampling, stable results.
- Golden tests for:
  - chat template serialization
  - stop sequence behavior
  - max_tokens enforcement
  - streaming vs non-streaming equivalence (same final text)
- Provide a way to run a "shadow mode" comparison against a reference implementation (can be offline) by logging prompts and comparing outputs.

Output:
- Test harness code
- Test vectors format
- CI-friendly command set
- Guidance on how to extend vectors for RAG scenarios
```

---

## Prompt 10 — Performance & Profiling Plan

```
Now produce a performance optimization plan for this engine.

Context:
- We have 2x L40S.
- Qwen2.5-32B-Instruct.
- 50–70k context; KV cache dominates.
- Current implementation is MVP (contiguous KV, basic scheduling).

Deliver:
1) Top 10 expected bottlenecks with measurement strategy (what to measure, where to instrument)
2) Steps to reach vLLM-class performance:
   - continuous batching
   - paged KV cache
   - split prefill/decode lanes
   - GPU-side sampling/logits transforms
   - kernel fusion opportunities
3) Concrete benchmarks and targets (tokens/sec at concurrency 1/4/16, p95 latency)
4) Profiling workflow on Windows (Nsight Systems/Compute), what traces to collect
5) A prioritized backlog (issues) with estimates and risk

Output should be explicit and actionable.
```

---

## Meta Prompts

### Meta Prompt A — Enforce Modularity

```
Refactor your output into modules <= 400 LOC per file. Provide a directory structure and list files with responsibilities. For each module, include unit tests or explain precisely why not. Keep dependencies minimal and Windows/MSVC friendly.
```

### Meta Prompt B — Force Memory Math Realism

```
Stop and compute KV cache memory requirements for 70k context for Qwen2.5-32B. Show the formula with heads/kv heads/hidden size/layers/dtype. Then derive max concurrency given 2x 48GB and leave room for weights/workspace. Use that to justify admission control and GPU routing.
```

---

## Key Technical Notes

### Memory Math for Qwen2.5-32B + 70k Context

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
             = 262,144 bytes = 256 KB per token
```

**KV Cache for 70k Context:**
```
KV_70k = 70,000 × 256 KB = 17.5 GB per request
```

**GPU Memory Budget (per L40S, 48GB):**
```
Model weights (FP16): ~64 GB total → 32 GB per GPU with tensor parallel
Workspace: ~2 GB
Available for KV: ~14 GB per GPU
```

**Max Concurrency at 70k (with Tensor Parallel):**
```
Per-GPU KV = 8.75 GB (split across 2)
Max concurrent = floor(14 GB / 8.75 GB) = 1 request at 70k
```

### Tensor Parallel Justification

70k context produces 17.5 GB KV cache per request. A single 48GB GPU cannot fit:
- 32B weights (would need 64GB for FP16)
- 17.5GB KV cache

Therefore tensor parallel is required for 70k context. This splits:
- Weights: 50% on each GPU
- KV cache: 50% on each GPU (by KV heads)
- Communication: All-reduce via NVLink (600 GB/s)
