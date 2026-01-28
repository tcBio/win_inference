# Benchmarks for performance measurement

add_executable(bench_tokenizer
    bench_tokenizer.cpp
)
target_link_libraries(bench_tokenizer PRIVATE
    qwen_tokenizer
)

add_executable(bench_kv_cache
    bench_kv_cache.cpp
)
target_link_libraries(bench_kv_cache PRIVATE
    qwen_kvcache
)

add_executable(bench_scheduler
    bench_scheduler.cpp
)
target_link_libraries(bench_scheduler PRIVATE
    qwen_scheduler
    qwen_backend
)

add_executable(bench_e2e
    bench_end_to_end.cpp
)
target_link_libraries(bench_e2e PRIVATE
    qwen_api
    qwen_scheduler
    qwen_backend
)
