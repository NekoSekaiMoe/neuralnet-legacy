# ── 主程序 ────────────────────────────────────────────────────────────────────
add_executable(mnist_train train.cpp)
add_executable(mnist_infer infer.cpp)

target_link_libraries(mnist_train PRIVATE TBB::tbb)
target_link_libraries(mnist_infer PRIVATE TBB::tbb)

target_include_directories(mnist_train PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(mnist_infer PRIVATE ${PROJECT_SOURCE_DIR}/include)
