add_executable(gpt_train train.cpp)
target_link_libraries(gpt_train PRIVATE neuralnet::neuralnet)
if(MSVC)
    target_compile_options(gpt_train PRIVATE /UNDEBUG)
else()
    target_compile_options(gpt_train PRIVATE -UNDEBUG)
endif()

add_executable(gpt_infer infer.cpp)
target_link_libraries(gpt_infer PRIVATE neuralnet::neuralnet)

add_executable(gpt_tokenizer_train tokenizer_train.cpp)
target_link_libraries(gpt_tokenizer_train PRIVATE neuralnet::neuralnet)

add_executable(gpt_tokenizer_infer tokenizer_infer.cpp)
target_link_libraries(gpt_tokenizer_infer PRIVATE neuralnet::neuralnet)
