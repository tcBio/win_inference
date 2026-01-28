# Unit tests
add_executable(test_tokenizer
    unit/test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    qwen_tokenizer
    GTest::gtest_main
)
add_test(NAME test_tokenizer COMMAND test_tokenizer)

add_executable(test_prompt_builder
    unit/test_prompt_builder.cpp
)
target_link_libraries(test_prompt_builder PRIVATE
    qwen_tokenizer
    GTest::gtest_main
)
add_test(NAME test_prompt_builder COMMAND test_prompt_builder)

add_executable(test_sampler
    unit/test_sampler.cpp
)
target_link_libraries(test_sampler PRIVATE
    qwen_core
    qwen_backend
    GTest::gtest_main
)
add_test(NAME test_sampler COMMAND test_sampler)

add_executable(test_scheduler
    unit/test_scheduler.cpp
)
target_link_libraries(test_scheduler PRIVATE
    qwen_scheduler
    qwen_backend
    GTest::gtest_main
)
add_test(NAME test_scheduler COMMAND test_scheduler)

add_executable(test_kv_allocator
    unit/test_kv_allocator.cpp
)
target_link_libraries(test_kv_allocator PRIVATE
    qwen_kvcache
    GTest::gtest_main
)
add_test(NAME test_kv_allocator COMMAND test_kv_allocator)

add_executable(test_api_types
    unit/test_api_types.cpp
)
target_link_libraries(test_api_types PRIVATE
    qwen_api
    GTest::gtest_main
)
add_test(NAME test_api_types COMMAND test_api_types)

# Golden tests
add_executable(test_golden
    golden/test_golden_outputs.cpp
)
target_link_libraries(test_golden PRIVATE
    qwen_tokenizer
    qwen_core
    GTest::gtest_main
)
add_test(NAME test_golden COMMAND test_golden)

# Integration tests
add_executable(test_integration
    integration/test_server_integration.cpp
)
target_link_libraries(test_integration PRIVATE
    qwen_api
    qwen_scheduler
    qwen_backend
    GTest::gtest_main
)
add_test(NAME test_integration COMMAND test_integration)
