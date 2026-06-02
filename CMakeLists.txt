cmake_minimum_required(VERSION 3.20)
project(neuralnet.cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 为所有目标添加编译标志
add_compile_options(-Wall -Wextra -O2)

# 启用 CTest
enable_testing()

# 链接 TBB（GCC libstdc++ 的 <execution> 并行策略后端）
find_package(TBB REQUIRED)

# ── 主程序 ────────────────────────────────────────────────────────────────────
add_executable(mnist_train src/train.cpp)
add_executable(mnist_infer src/infer.cpp)

target_link_libraries(mnist_train PRIVATE TBB::tbb)
target_link_libraries(mnist_infer PRIVATE TBB::tbb)

target_include_directories(mnist_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mnist_infer PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ── 测试 ──────────────────────────────────────────────────────────────────────
add_executable(test_nn src/test_nn.cpp)
target_link_libraries(test_nn PRIVATE TBB::tbb)
target_include_directories(test_nn PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 每个 test_entry 注册为独立 CTest，方便 ctest 单独运行/过滤
set(nn_test_names
    counting_iterator
    matrix_construction
    matrix_arithmetic
    matrix_transpose
    matrix_multiplication
    relu
    linear_forward_backward
    mse_loss
    cross_entropy_loss
    sgd
    sgd_momentum
    model_forward
    model_io
    one_hot
    e2e_train
)

foreach(name ${nn_test_names})
    add_test(NAME ${name} COMMAND test_nn ${name})
endforeach()
