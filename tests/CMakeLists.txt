# ── 测试 ──────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    add_executable(test_nn test_nn.cpp)
    target_link_libraries(test_nn PRIVATE neuralnet::neuralnet)
    target_include_directories(test_nn PRIVATE ${PROJECT_SOURCE_DIR}/include)

    # 每个 test_entry 注册为独立 CTest，方便 ctest 单独运行/过滤
    set(nn_test_names
        counting_iterator
        matrix_construction
        matrix_arithmetic
        matrix_transpose
        matrix_multiplication
        matrix_norm
        matrix_colwise_mean
        relu
        leaky_relu
        sigmoid
        tanh
        gelu
        dropout
        linear_forward_backward
        linear_backward_blocked_matmul
        mse_loss
        cross_entropy_loss
        sgd
        sgd_momentum
        adam
        step_lr
        cosine_lr
        exp_lr
        model_forward
        model_default_is_training
        model_train_eval
        on_mode_change
        model_io
        model_io_member
        model_save_load_v1
        model_save_load_v2
        model_load_state_count_mismatch
        one_hot
        e2e_train
        set_data_match
        set_data_resize_grow
        set_data_resize_shrink
        set_data_from_empty
        set_data_empty_throws
        set_data_inconsistent_throws
        set_data_exception_safety
        batchnorm1d_forward_shape
        batchnorm1d_forward_normalize
        batchnorm1d_backward_shape
        batchnorm1d_eval_mode
        batchnorm1d_running_stats_update
        batchnorm1d_no_affine
        batchnorm1d_save_load_v2
        batchnorm1d_gradient_check
        batchnorm1d_input_gradient_check
        batchnorm1d_batch_size_one
        batchnorm2d_forward_shape
        batchnorm2d_normalize
        batchnorm2d_save_load_v2
        dataloader_basic
        dataloader_drop_last
        dataloader_shuffle
        dataloader_shuffle_reproducible_with_seed
        grad_clip_no_op_below_threshold
        grad_clip_scales_above_threshold
        grad_clip_nan_throws
        grad_clip_inf_throws
        summary_contains_total_params
        summary_contains_all_layer_names
        summary_output_stream
    )

    foreach(name ${nn_test_names})
        add_test(NAME ${name} COMMAND test_nn ${name})
    endforeach()
endif()
