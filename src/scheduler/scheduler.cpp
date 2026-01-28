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
    if (!kv_allocator_->can_allocate(estimated_seq_len)) {
        return Error::resource_exhausted("Insufficient memory for request");
    }

    return Result<void>::success();
}

Result<void> Scheduler::run_prefill(RequestPtr request) {
    request->set_state(RequestState::Prefilling);
    request->timing.prefill_start = std::chrono::steady_clock::now();

    gauge(metrics::REQUESTS_ACTIVE).increment();

    // Allocate KV cache
    size_t max_seq_len = request->input_tokens.size() + request->max_tokens;
    auto alloc_result = kv_allocator_->allocate(max_seq_len);
    if (alloc_result.is_error()) {
        gauge(metrics::REQUESTS_ACTIVE).decrement();
        return alloc_result.error();
    }
    request->kv_cache = std::move(alloc_result.value());

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

    request->kv_cache.current_len = request->input_tokens.size();
    request->timing.prefill_end = std::chrono::steady_clock::now();

    histogram(metrics::PREFILL_TIME).observe(
        request->timing.prefill_time_ms() / 1000.0);

    log_debug("scheduler", "prefill_complete", {
        {"request_id", request->id},
        {"prefill_ms", request->timing.prefill_time_ms()},
        {"prompt_tokens", request->input_tokens.size()}
    });

    return Result<void>::success();
}

Result<bool> Scheduler::run_decode_step(RequestPtr request) {
    if (request->is_cancelled()) {
        return false;
    }

    // Determine input token
    int32_t input_token;
    if (request->output_tokens.empty()) {
        // First decode step - use last prefill logits
        // In real implementation, would sample from prefill output
        input_token = tokenizer_->special_tokens().eos_token_id;  // Placeholder
    } else {
        input_token = request->output_tokens.back();
    }

    // Run decode
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

    // Record first token time
    if (request->output_tokens.size() == 1) {
        request->timing.first_token_at = std::chrono::steady_clock::now();
        histogram(metrics::TIME_TO_FIRST_TOKEN).observe(
            request->timing.time_to_first_token_ms() / 1000.0);
    }

    // Decode token to text
    auto text_result = tokenizer_->decode_token(new_token);
    std::string token_text = text_result.ok() ? text_result.value() : "";
    request->output_text += token_text;

    // Stream token if callback is set
    if (request->on_token) {
        request->on_token(new_token, token_text);
    }

    counter(metrics::TOKENS_GENERATED).increment();

    // Check stop conditions
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

        // Run prefill
        auto prefill_result = run_prefill(request);
        if (prefill_result.is_error()) {
            fail_request(request, prefill_result.error());
        } else {
            request->set_state(RequestState::Decoding);
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

    if (request->on_complete) {
        request->on_complete(reason);
    }

    free_resources(request);

    {
        std::lock_guard<std::mutex> lock(active_mutex_);
        active_requests_.erase(
            std::remove(active_requests_.begin(), active_requests_.end(), request),
            active_requests_.end());
    }

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

    if (request->on_error) {
        request->on_error(error);
    }

    free_resources(request);

    {
        std::lock_guard<std::mutex> lock(active_mutex_);
        active_requests_.erase(
            std::remove(active_requests_.begin(), active_requests_.end(), request),
            active_requests_.end());
    }

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
        kv_allocator_->free(request->kv_cache);
    }
}

}  // namespace qwen::scheduler
