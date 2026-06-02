# ── 测试 ──────────────────────────────────────────────────────────────────────
add_executable(test_nn test_nn.cpp)
target_link_libraries(test_nn PRIVATE TBB::tbb)
target_include_directories(test_nn PRIVATE ${PROJECT_SOURCE_DIR}/include)

# 每个 test_entry 注册为独立 CTest，方便 ctest 单独运行/过滤
set(nn_test_names
    counting_iterator
    matrix_construction
    matrix_arithmetic
    matrix_transpose
    matrix_multiplication
    relu
    leaky_relu
    sigmoid
    tanh
    gelu
    dropout
    linear_forward_backward
    mse_loss
    cross_entropy_loss
    sgd
    sgd_momentum
    adam
    step_lr
    cosine_lr
    exp_lr
    model_forward
    model_io
    one_hot
    e2e_train
)

foreach(name ${nn_test_names})
    add_test(NAME ${name} COMMAND test_nn ${name})
endforeach()
