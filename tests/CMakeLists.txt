# ── 测试 ──────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    add_executable(test_nn test_nn.cpp)
    target_link_libraries(test_nn PRIVATE neuralnet::neuralnet)
    target_include_directories(test_nn PRIVATE ${PROJECT_SOURCE_DIR}/include)

    add_executable(test_tensor test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE neuralnet::neuralnet)
    target_include_directories(test_tensor PRIVATE ${PROJECT_SOURCE_DIR}/include)
    if(MSVC)
        target_compile_options(test_tensor PRIVATE /UNDEBUG)
    else()
        target_compile_options(test_tensor PRIVATE -UNDEBUG)
    endif()

    set(tensor_test_names
        tensor_add
        tensor_matmul
        tensor_relu
        tensor_linear
    )
    foreach(name ${tensor_test_names})
        add_test(NAME ${name} COMMAND test_tensor ${name})
    endforeach()

    add_executable(test_transformer test_transformer.cpp)
    target_link_libraries(test_transformer PRIVATE neuralnet::neuralnet)
    target_include_directories(test_transformer PRIVATE ${PROJECT_SOURCE_DIR}/include)

    # 测试断言基于 <cassert>。Release 构建（CMake 默认行为）会注入 -DNDEBUG，
    # 使 assert(...) 变为空表达式，整个用例无论真假都会打印 "PASSED" — 测试形同虚设。
    # 这里在编译标志末尾追加 -UNDEBUG，强制无视上游 NDEBUG，让 assert 始终生效。
    # 用 -UNDEBUG 而不是 PRIVATE_OPTIONS 控制构建类型，是因为它独立于 BUILD_TYPE，
    # CI 跑 Release 也能保留断言。
    if(MSVC)
        target_compile_options(test_nn PRIVATE /UNDEBUG)
        target_compile_options(test_transformer PRIVATE /UNDEBUG)
    else()
        target_compile_options(test_nn PRIVATE -UNDEBUG)
        target_compile_options(test_transformer PRIVATE -UNDEBUG)
    endif()

    # 每个 test_entry 注册为独立 CTest，方便 ctest 单独运行/过滤
    set(nn_test_names
        counting_iterator
        matrix_construction
        matrix_arithmetic
        matrix_transpose
        matrix_multiplication
        matrix_matmul_nt_tn
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
        linear_backward_grad_b_nonsquare
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

    set(transformer_test_names
        matrix_resize
        matrix_row_slice
        matrix_set_row_slice
        layernorm_forward_shape
        layernorm_normalize
        layernorm_backward_shape
        layernorm_gradient_check
        softmax_forward
        softmax_backward_shape
        positional_encoding_shape
        positional_encoding_adds_values
        positional_encoding_backward_passthrough
        mha_forward_shape
        mha_backward_shape
        mha_params_count
        feedforward_shape
        feedforward_params
        encoder_layer_shape
        encoder_forward_shape
        encoder_backward_shape
        patch_embedding_shape
        patch_embedding_backward_shape
        vit_e2e_forward
        vit_e2e_backward
    )

    foreach(name ${transformer_test_names})
        add_test(NAME transformer_${name} COMMAND test_transformer ${name})
    endforeach()
    add_executable(test_gpt test_gpt.cpp)
    target_link_libraries(test_gpt PRIVATE neuralnet::neuralnet)
    target_include_directories(test_gpt PRIVATE ${PROJECT_SOURCE_DIR}/include)
    if(MSVC)
        target_compile_options(test_gpt PRIVATE /UNDEBUG)
    else()
        target_compile_options(test_gpt PRIVATE -UNDEBUG)
    endif()

    set(gpt_test_names
        causal_self_attention_shape
        causal_self_attention_masking
        gpt_block_shape
        gpt_model_shape
        gpt_model_numeric_grad
        gpt_model_generate
    )
    foreach(name ${gpt_test_names})
        add_test(NAME gpt_${name} COMMAND test_gpt ${name})
    endforeach()

    add_executable(test_tokenizer test_tokenizer.cpp)
    target_link_libraries(test_tokenizer PRIVATE neuralnet::neuralnet)
    target_include_directories(test_tokenizer PRIVATE ${PROJECT_SOURCE_DIR}/include)
    if(MSVC)
        target_compile_options(test_tokenizer PRIVATE /UNDEBUG)
    else()
        target_compile_options(test_tokenizer PRIVATE -UNDEBUG)
    endif()

    set(tokenizer_test_names
        char_tokenizer
        byte_zip_tokenizer_train
        bpe_tokenizer_load_vocab
    )
    foreach(name ${tokenizer_test_names})
        add_test(NAME tokenizer_${name} COMMAND test_tokenizer ${name})
    endforeach()
endif()
