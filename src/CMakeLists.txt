# ── 主程序 ────────────────────────────────────────────────────────────────────
add_executable(mnist_train train.cpp)
add_executable(mnist_infer infer.cpp)

target_link_libraries(mnist_train PRIVATE neuralnet::neuralnet)
target_link_libraries(mnist_infer PRIVATE neuralnet::neuralnet)

target_include_directories(mnist_train PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(mnist_infer PRIVATE ${PROJECT_SOURCE_DIR}/include)
