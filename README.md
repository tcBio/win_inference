# Qwen2.5-32B-Instruct Windows Inference Server

A Windows-native, high-performance inference server for Qwen2.5-32B-Instruct with OpenAI-compatible API.

## Features

- **OpenAI-compatible API** with `/v1/chat/completions` endpoint
- **SSE streaming** for real-time token delivery
- **50-70k context support** optimized for RAG workloads
- **Multi-GPU support** with tensor parallel for 2× L40S
- **Production quality** with structured logging, metrics, and observability
- **Windows-native** with MSVC + CMake build

## Hardware Requirements

- 2× NVIDIA L40S (48GB each) or equivalent
- NVLink for tensor parallel communication
- CUDA 12.x compatible driver
- Windows 10/11 or Windows Server 2019+

## Quick Start

### Build

```powershell
# Configure
cmake -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release --parallel

# Test
ctest --test-dir build -C Release --output-on-failure
```

### Run

```powershell
.\build\Release\qwen_server.exe configs\server_config.json
```

### Test with curl

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-32b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-32b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming/non-streaming) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

## Configuration

See `configs/server_config.json` for all options:

```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8080,
    "model_name": "qwen2.5-32b-instruct"
  },
  "scheduler": {
    "max_queue_size": 100,
    "max_active_requests": 4
  },
  "model": {
    "path": "/models/qwen2.5-32b-instruct",
    "max_seq_len": 70000,
    "tensor_parallel": true
  }
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      HTTP Server (httplib)                   │
│                 /v1/chat/completions, /health                │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                      Request Validator                       │
│              Schema validation, parameter bounds             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                      Prompt Builder                          │
│           Qwen chat template, tokenization                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                        Scheduler                             │
│         Request registry, decode loop, cancellation          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                      KV Cache Allocator                      │
│            Memory budgeting, admission control               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                    GPU Device Manager                        │
│              Tensor parallel, device routing                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                     Model Runtime                            │
│             TensorRT-LLM / Null backend (testing)            │
└─────────────────────────────────────────────────────────────┘
```

## Memory Requirements

For Qwen2.5-32B with 70k context:

| Component | Memory |
|-----------|--------|
| Model weights (FP16) | ~64 GB (32 GB per GPU with TP) |
| KV cache per request | 17.5 GB (8.75 GB per GPU with TP) |
| Workspace | ~2 GB per GPU |
| **Available for KV** | **~14 GB per GPU** |

**Max concurrent requests at 70k context: 1**

For shorter contexts:
- 10k context: ~5 concurrent requests
- 20k context: ~2-3 concurrent requests

## Documentation

- [Engineering Specification](docs/ENGINEERING_SPEC.md)
- [Implementation Prompts](docs/PROMPT_PACK.md)
- [Performance Optimization Plan](docs/PERFORMANCE.md)

## Project Structure

```
win_inference/
├── include/           # Header files
│   ├── api/          # HTTP server, routes, SSE
│   ├── core/         # Sampler, stop checker
│   ├── tokenizer/    # Tokenizer, prompt builder
│   ├── scheduler/    # Request lifecycle, decode loop
│   ├── kvcache/      # Memory management
│   ├── gpu/          # Multi-GPU coordination
│   ├── backend/      # Model runtime interface
│   └── utils/        # Logging, metrics, errors
├── src/              # Implementation files
├── tests/            # Unit, integration, golden tests
├── benchmarks/       # Performance benchmarks
├── configs/          # Configuration files
└── docs/             # Documentation
```

## Testing

```powershell
# Run all tests
ctest --test-dir build -C Release --output-on-failure

# Run specific test
.\build\tests\Release\test_tokenizer.exe
.\build\tests\Release\test_prompt_builder.exe
.\build\tests\Release\test_golden.exe
```

## Benchmarks

```powershell
# Tokenizer performance
.\build\benchmarks\Release\bench_tokenizer.exe

# KV cache allocation
.\build\benchmarks\Release\bench_kv_cache.exe

# Scheduler throughput
.\build\benchmarks\Release\bench_scheduler.exe

# End-to-end latency
.\build\benchmarks\Release\bench_e2e.exe
```

## Development

### Priority Order

1. **Correctness/Accuracy**: Prompt formatting, logits, stop behavior
2. **Stability/Observability**: Logging, metrics, error handling
3. **Performance**: Throughput, latency optimization

### Code Guidelines

- Module size: ≤400 LOC per file
- Modern C++20
- RAII for resource management
- Structured logging with request IDs
- Unit tests for all modules

## License

[Add license here]

## Acknowledgments

Built following best practices from:
- vLLM
- TensorRT-LLM
- OpenAI API specification
