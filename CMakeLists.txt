cmake_minimum_required(VERSION 3.20)
project(neuralnet.cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 为所有目标添加编译标志
add_compile_options(-Wall -Wextra -O2)

# 链接 TBB（GCC libstdc++ 的 <execution> 并行策略后端）
find_package(TBB REQUIRED)

add_executable(mnist_train src/train.cpp)
add_executable(mnist_infer src/infer.cpp)

target_link_libraries(mnist_train PRIVATE TBB::tbb)
target_link_libraries(mnist_infer PRIVATE TBB::tbb)

target_include_directories(mnist_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mnist_infer PRIVATE ${CMAKE_SOURCE_DIR}/include)
