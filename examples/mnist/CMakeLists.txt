# ── MNIST 示例程序 ────────────────────────────────────────────────────────────
# 这些是 demo / example，链接到 neuralnet::neuralnet 库。
# 不安装（install 阶段不包含这些可执行文件）。

add_executable(mnist_train train.cpp)
add_executable(mnist_infer infer.cpp)

target_link_libraries(mnist_train PRIVATE neuralnet::neuralnet)
target_link_libraries(mnist_infer PRIVATE neuralnet::neuralnet)
