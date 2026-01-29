#include "scheduler/scheduler.hpp"
#include "utils/logger.hpp"
#include "utils/metrics.hpp"

#include <chrono>

namespace qwen::scheduler {

Scheduler::Scheduler(
    std::shared_ptr<backend::IModelRuntime> runtime,
    std::shared_ptr<kvcache::IKVCacheAllocator> kv_allocator,
    std::shared_ptr<tokenizer::ITokenizer> tokenizer,
    SchedulerConfig config
)
    : runtime_(std::move(runtime))
    , kv_allocator_(std::move(kv_allocator))
    , tokenizer_(std::move(tokenizer))
    , config_(config) {

    sampler_ = std::make_unique<core::Sampler>();
    stop_checker_ = std::make_unique<core::StopChecker>(tokenizer_);
}

Scheduler::~Scheduler() {
    stop();
}

Result<void> Scheduler::start() {
    if (running_.load()) {
        return Error::internal("Scheduler already running");
    }

    running_.store(true);
    decode_thread_ = std::thread([this] { decode_loop(); });

    log_info("scheduler", "started", {
        {"max_queue_size", config_.max_queue_size},
        {"max_active_requests", config_.max_active_requests}
    });

    return Result<void>::success();
}

void Scheduler::stop() {
    if (!running_.load()) return;

    running_.store(false);
    queue_cv_.notify_all();

    if (decode_thread_.joinable()) {
        decode_thread_.join();
    }

    // Cancel all pending requests
    registry_.cancel_all();

    log_info("scheduler", "stopped", {});
}

Result<void> Scheduler::submit(RequestPtr request) {
    if (!running_.load()) {
        return Error::internal("Scheduler not running");
    }

    // Admission control
    auto admit_result = admit(request);
    if (admit_result.is_error()) {
        request->set_state(RequestState::Rejected);
        request->error = admit_result.error();
        if (request->on_error) {
            request->on_error(admit_result.error());
        }
        return admit_result.error();
    }

    // Register request
    registry_.add(request);
    request->set_state(RequestState::Queued);
    request->timing.queued_at = std::chrono::steady_clock::now();

    // Set timeout deadlines
    auto now = std::chrono::steady_clock::now();
    if (config_.queue_timeout_sec > 0) {
        request->timing.queue_deadline = now + std::chrono::seconds(config_.queue_timeout_sec);
    } else {
        request->timing.queue_deadline = std::chrono::steady_clock::time_point::max();
    }
    if (config_.request_timeout_sec > 0) {
        request->timing.request_deadline = now + std::chrono::seconds(config_.request_timeout_sec);
    } else {
        request->timing.request_deadline = std::chrono::steady_clock::time_point::max();
    }
    // decode_deadline is set when prefill completes
    request->timing.decode_deadline = std::chrono::steady_clock::time_point::max();

    // Add to queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_queue_.push(request);
    }
    queue_cv_.notify_one();

    counter(metrics::REQUESTS_TOTAL).increment();
    gauge(metrics::QUEUE_DEPTH).increment();

    log_info("scheduler", "request_submitted", {
        {"request_id", request->id},
        {"streaming", request->is_streaming}
    }, {request->id, request->trace_id, ""});

    return Result<void>::success();
}

void Scheduler::cancel(const std::string& request_id) {
    registry_.cancel(request_id);
}

RequestPtr Scheduler::get_request(const std::string& request_id) const {
    return registry_.get(request_id);
}

Scheduler::Stats Scheduler::stats() const {
    auto counts = registry_.state_counts();
    return Stats{
        .queued_count = counts[RequestState::Queued],
        .active_count = counts[RequestState::Prefilling] + counts[RequestState::Decoding],
        .completed_count = completed_count_.load(),
        .cancelled_count = cancelled_count_.load(),
        .errored_count = errored_count_.load()
    };
}

bool Scheduler::is_running() const {
    return running_.load();
}

void Scheduler::set_gpu_router(std::shared_ptr<gpu::GPURouter> router) {
    gpu_router_ = std::move(router);
}

size_t Scheduler::estimate_kv_memory(size_t seq_len) {
    return seq_len * KV_BYTES_PER_TOKEN;
}

Result<void> Scheduler::admit(RequestPtr request) {
    // Check queue size
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (pending_queue_.size() >= config_.max_queue_size) {
            return Error::resource_exhausted("Queue full");
        }
    }

    // Estimate memory requirements
    size_t estimated_seq_len = request->input_tokens.size() + request->max_tokens;
    size_t estimated_bytes = estimate_kv_memory(estimated_seq_len);

    // Check GPU router budgets if available
    if (gpu_router_) {
        if (!gpu_router_->can_place(request, estimated_bytes)) {
            return Error::resource_exhausted("Insufficient GPU memory for request");
        }
    }

    // Also check KV allocator (may have different limits)
    if (!kv_allocator_->can_allocate(estimated_seq_len)) {
        return Error::resource_exhausted("Insufficient memory for request");
    }

    return Result<void>::success();
}

Result<void> Scheduler::run_prefill(RequestPtr request) {
    request->set_state(RequestState::Prefilling);
    request->timing.prefill_start = std::chrono::steady_clock::now();

    gauge(metrics::REQUESTS_ACTIVE).increment();

    // Calculate memory requirements
    size_t max_seq_len = request->input_tokens.size() + request->max_tokens;
    size_t estimated_bytes = estimate_kv_memory(max_seq_len);

    // Route request to GPU(s) if router available
    gpu::RoutingDecision routing;
    if (gpu_router_) {
        routing = gpu_router_->route(request);
        request->assigned_devices = routing.device_ids;

        // Reserve memory in GPU budgets
        gpu_router_->reserve(routing, estimated_bytes);
    }

    // Allocate KV cache (tensor-parallel if routing decision requires it)
    Result<kvcache::KVCacheHandle> alloc_result;
    if (gpu_router_ && routing.strategy == gpu::GPUStrategy::TensorParallel
            && routing.device_ids.size() > 1) {
        alloc_result = kv_allocator_->allocate_tensor_parallel(max_seq_len, routing.device_ids);
    } else {
        alloc_result = kv_allocator_->allocate(max_seq_len);
    }

    if (alloc_result.is_error()) {
        // Release GPU budget reservation on failure
        if (gpu_router_) {
            gpu_router_->unreserve(routing, estimated_bytes);
        }
        gauge(metrics::REQUESTS_ACTIVE).decrement();
        return alloc_result.error();
    }
    request->kv_cache = std::move(alloc_result.value());

    // Commit GPU budget (move from reserved to used)
    if (gpu_router_) {
        gpu_router_->commit(routing, estimated_bytes);
    }

    // Run prefill
    backend::PrefillInput input{
        .input_ids = request->input_tokens,
        .kv_cache = &request->kv_cache,
        .seq_len = static_cast<int32_t>(request->input_tokens.size())
    };

    auto prefill_result = runtime_->prefill(input, request->cancel_token);
    if (prefill_result.is_error()) {
        free_resources(request);
        gauge(metrics::REQUESTS_ACTIVE).decrement();
        return prefill_result.error();
    }

    // Store prefill logits for first token sampling
    const auto& output = prefill_result.value();
    request->prefill_logits = output.logits.data;
    request->prefill_logits_vocab_size = output.logits.vocab_size;

    request->kv_cache.current_len = request->input_tokens.size();
    request->timing.prefill_end = std::chrono::steady_clock::now();

    histogram(metrics::PREFILL_TIME).observe(
        request->timing.prefill_time_ms() / 1000.0);

    log_debug("scheduler", "prefill_complete", {
        {"request_id", request->id},
        {"prefill_ms", request->timing.prefill_time_ms()},
        {"prompt_tokens", request->input_tokens.size()},
        {"assigned_devices", request->assigned_devices.size()}
    });

    return Result<void>::success();
}

Result<bool> Scheduler::run_decode_step(RequestPtr request) {
    if (request->is_cancelled()) {
        return false;
    }

    backend::Logits logits_to_sample;
    int32_t input_token;

    if (request->output_tokens.empty()) {
        // First decode step - sample from prefill logits
        if (request->prefill_logits.empty()) {
            return Error::internal("No prefill logits available for first token");
        }

        logits_to_sample.data = std::move(request->prefill_logits);
        logits_to_sample.vocab_size = request->prefill_logits_vocab_size;
        request->prefill_logits.clear();  // Free memory

        // Sample first token from prefill logits
        core::SamplingParams params{
            .temperature = request->temperature,
            .top_p = request->top_p,
            .seed = request->seed
        };

        auto sample_result = sampler_->sample(logits_to_sample, params, {});
        if (sample_result.is_error()) {
            return sample_result.error();
        }

        int32_t first_token = sample_result.value().token_id;
        request->output_tokens.push_back(first_token);
        request->kv_cache.current_len++;

        // Record first token time
        request->timing.first_token_at = std::chrono::steady_clock::now();
        histogram(metrics::TIME_TO_FIRST_TOKEN).observe(
            request->timing.time_to_first_token_ms() / 1000.0);

        // Decode and emit first token
        auto text_result = tokenizer_->decode_token(first_token);
        std::string token_text = text_result.ok() ? text_result.value() : "";
        request->output_text.reserve(request->max_tokens * 4);  // Pre-allocate
        request->output_text += token_text;

        // Stream via token queue (thread-safe) or callback
        emit_token(request, first_token, token_text);

        counter(metrics::TOKENS_GENERATED).increment();

        // Check stop after first token
        return check_stop_and_continue(request);
    }

    // Subsequent decode steps
    input_token = request->output_tokens.back();

    backend::DecodeInput input{
        .input_token = input_token,
        .kv_cache = &request->kv_cache,
        .current_pos = static_cast<int32_t>(request->kv_cache.current_len)
    };

    auto decode_result = runtime_->decode_step(input, request->cancel_token);
    if (decode_result.is_error()) {
        return decode_result.error();
    }

    // Sample next token
    core::SamplingParams params{
        .temperature = request->temperature,
        .top_p = request->top_p,
        .seed = request->seed
    };

    auto sample_result = sampler_->sample(
        decode_result.value().logits, params, request->output_tokens);
    if (sample_result.is_error()) {
        return sample_result.error();
    }

    int32_t new_token = sample_result.value().token_id;
    request->output_tokens.push_back(new_token);
    request->kv_cache.current_len++;

    // Decode token to text
    auto text_result = tokenizer_->decode_token(new_token);
    std::string token_text = text_result.ok() ? text_result.value() : "";
    request->output_text += token_text;

    // Stream token
    emit_token(request, new_token, token_text);

    counter(metrics::TOKENS_GENERATED).increment();

    return check_stop_and_continue(request);
}

void Scheduler::emit_token(RequestPtr request, int32_t token_id, const std::string& text) {
    // Prefer thread-safe token queue for streaming
    if (request->token_queue) {
        bool pushed = request->token_queue->push(TokenEvent{
            .type = TokenEvent::Type::Token,
            .token_id = token_id,
            .text = text
        });
        if (!pushed) {
            // Token dropped due to backpressure - log warning (once per request)
            size_t dropped = request->token_queue->dropped_count();
            if (dropped == 1) {
                log_warn("scheduler", "token_queue_backpressure", {
                    {"request_id", request->id},
                    {"queue_full", true}
                });
            }
            counter(metrics::TOKENS_DROPPED).increment();
        }
    }
    // Fall back to callback (deprecated, not thread-safe for SSE)
    else if (request->on_token) {
        request->on_token(token_id, text);
    }
}

Result<bool> Scheduler::check_stop_and_continue(RequestPtr request) {
    core::StopConfig stop_config{
        .stop_sequences = request->stop_sequences,
        .max_tokens = request->max_tokens
    };

    auto stop_result = stop_checker_->check(
        request->output_tokens, request->output_text, stop_config);

    if (stop_result.should_stop) {
        request->finish_reason = stop_result.reason;

        // Trim stop sequence from output if present
        if (!stop_result.matched_stop_sequence.empty()) {
            request->output_text = stop_checker_->trim_stop_sequence(
                request->output_text, stop_result.matched_stop_sequence);
        }

        return false;  // Done
    }

    return true;  // Continue
}

void Scheduler::decode_loop() {
    while (running_.load()) {
        // Process queue - try to start new requests
        process_queue();

        // Get active requests
        std::vector<RequestPtr> active;
        {
            std::lock_guard<std::mutex> lock(active_mutex_);
            active = active_requests_;
        }

        if (active.empty()) {
            // Wait for new requests
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock,
                std::chrono::microseconds(config_.decode_loop_interval_us),
                [this] { return !pending_queue_.empty() || !running_.load(); });
            continue;
        }

        // Run decode step for each active request
        for (auto& request : active) {
            if (request->is_cancelled()) {
                complete_request(request, api::FinishReason::Stop);
                continue;
            }

            // Check for request timeout
            if (request->timing.is_request_timed_out()) {
                fail_request(request, Error::timeout("Request timed out"));
                continue;
            }

            // Check for decode timeout
            if (request->timing.is_decode_timed_out()) {
                fail_request(request, Error::timeout("Decode phase timed out"));
                continue;
            }

            auto result = run_decode_step(request);
            if (result.is_error()) {
                fail_request(request, result.error());
            } else if (!result.value()) {
                // Decode complete
                complete_request(request, request->finish_reason);
            }
        }
    }
}

void Scheduler::process_queue() {
    std::unique_lock<std::mutex> queue_lock(queue_mutex_);

    while (!pending_queue_.empty()) {
        // Check if we can start more requests
        {
            std::lock_guard<std::mutex> active_lock(active_mutex_);
            if (active_requests_.size() >= config_.max_active_requests) {
                break;
            }
        }

        auto request = pending_queue_.front();
        pending_queue_.pop();
        gauge(metrics::QUEUE_DEPTH).decrement();

        queue_lock.unlock();

        // Check for queue timeout
        if (request->timing.is_queue_timed_out()) {
            fail_request(request, Error::timeout("Request timed out waiting in queue"));
            queue_lock.lock();
            continue;
        }

        // Check for overall request timeout
        if (request->timing.is_request_timed_out()) {
            fail_request(request, Error::timeout("Request timed out"));
            queue_lock.lock();
            continue;
        }

        // Run prefill
        auto prefill_result = run_prefill(request);
        if (prefill_result.is_error()) {
            fail_request(request, prefill_result.error());
        } else {
            request->set_state(RequestState::Decoding);

            // Set decode deadline if configured
            if (config_.decode_timeout_sec > 0) {
                request->timing.decode_deadline =
                    std::chrono::steady_clock::now() + std::chrono::seconds(config_.decode_timeout_sec);
            }

            std::lock_guard<std::mutex> active_lock(active_mutex_);
            active_requests_.push_back(request);
        }

        queue_lock.lock();
    }
}

void Scheduler::complete_request(RequestPtr request, api::FinishReason reason) {
    request->finish_reason = reason;
    request->set_state(RequestState::Completed);
    request->timing.completed_at = std::chrono::steady_clock::now();

    // Emit completion via token queue (thread-safe)
    if (request->token_queue) {
        request->token_queue->push(TokenEvent{
            .type = TokenEvent::Type::Complete,
            .finish_reason = reason
        });
        request->token_queue->close();
    }
    // Fall back to callback
    else if (request->on_complete) {
        request->on_complete(reason);
    }

    free_resources(request);

    {
        std::lock_guard<std::mutex> lock(active_mutex_);
        active_requests_.erase(
            std::remove(active_requests_.begin(), active_requests_.end(), request),
            active_requests_.end());
    }

    // Remove from registry to prevent memory leak
    registry_.remove(request->id);

    gauge(metrics::REQUESTS_ACTIVE).decrement();
    completed_count_.fetch_add(1);

    histogram(metrics::DECODE_TIME).observe(
        request->timing.total_time_ms() / 1000.0);

    log_info("scheduler", "request_completed", {
        {"request_id", request->id},
        {"finish_reason", api::finish_reason_to_string(reason)},
        {"prompt_tokens", request->prompt_tokens()},
        {"completion_tokens", request->completion_tokens()},
        {"total_ms", request->timing.total_time_ms()}
    }, {request->id, request->trace_id, ""});
}

void Scheduler::fail_request(RequestPtr request, Error error) {
    request->error = error;
    request->set_state(RequestState::Errored);
    request->timing.completed_at = std::chrono::steady_clock::now();

    // Emit error via token queue (thread-safe)
    if (request->token_queue) {
        request->token_queue->push(TokenEvent{
            .type = TokenEvent::Type::Error,
            .error = error
        });
        request->token_queue->close();
    }
    // Fall back to callback
    else if (request->on_error) {
        request->on_error(error);
    }

    free_resources(request);

    {
        std::lock_guard<std::mutex> lock(active_mutex_);
        active_requests_.erase(
            std::remove(active_requests_.begin(), active_requests_.end(), request),
            active_requests_.end());
    }

    // Remove from registry to prevent memory leak
    registry_.remove(request->id);

    gauge(metrics::REQUESTS_ACTIVE).decrement();
    errored_count_.fetch_add(1);

    log_error("scheduler", "request_failed", {
        {"request_id", request->id},
        {"error_code", error.code},
        {"error_message", error.message}
    }, {request->id, request->trace_id, ""});
}

void Scheduler::free_resources(RequestPtr request) {
    if (request->kv_cache.is_valid()) {
        // Release GPU budget
        if (gpu_router_ && !request->assigned_devices.empty()) {
            size_t max_seq_len = request->input_tokens.size() + request->max_tokens;
            size_t estimated_bytes = estimate_kv_memory(max_seq_len);
            gpu_router_->free(request->assigned_devices, estimated_bytes);
        }

        kv_allocator_->free(request->kv_cache);
    }
}

}  // namespace qwen::scheduler
