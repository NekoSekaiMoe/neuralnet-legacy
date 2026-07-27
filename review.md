Inline comments:
In `@include/neuralnet/module.h`:
- Around line 1523-1570: Update the decoding conditions in the generation logic
so temperature scaling applies only when temperature is positive and differs
from 1.0, while token selection samples for every temperature greater than 0.0,
including the default 1.0; reserve argmax decoding exclusively for temperature
== 0.0.
- Around line 1020-1056: 在 TransformerEncoderModule 构造函数中复用同文件既有的
validated_num_patches 校验逻辑，确保传入的 num_patches 为有效非零值后再初始化 num_patches_；不要让
forward 中的除法和取模接收到零值。
- Around line 551-574: Update the backward-gradient block guarded by
`in->requires_grad` to validate the `gamma_node` weak-pointer result before
accessing `gamma_node->data`. Preserve the existing gradient computation when
`gamma_node` is available, and follow the null-handling behavior already used at
L532 and L547 for an expired node.
- Line 1387: 在 forward() 中将 req_grad 从硬编码 true 改为根据输入 requires_grad
与模型参数实际梯度需求推导；更新 generate() 使用 requires_grad=false 的 Tensor
input，并在生成期间临时禁用参数梯度、完成后恢复原状态，以确保推理路径不构建或保留反向图。

In `@ref`:
- Line 1: 修复仓库中的 ref gitlink 配置：在 .gitmodules 中补充 ref
的子模块路径和对应元数据，并确保引用的提交对象可获取；如果 ref 不应作为子模块，则移除该 gitlink，改为提交源码或声明为普通依赖。

In `@refactor_norms_layers.py`:
- Around line 88-130: Update each regex replacement in the migration logic,
including bn1d_regex, bn2d_regex, and ln_regex, to use re.subn and capture the
replacement count. Raise an error when any count is not exactly one, and only
write layer.h after all three substitutions succeed.

In `@review.md`:
- Around line 188-190: 在 review.md 对应说明中，将索引表达式 `merges[best_priority][2]`
用反引号标记为行内代码，确保 Markdown 将其作为代码而不是引用链接解析。

In `@scripts/gui.py`:
- Around line 17-22: Update BUILD_DIR to derive from the resolved absolute path
of __file__ before traversing to the project root and appending "build"; keep
the executable path constants TRAIN_EXE, INFER_EXE, TEXT_TRAIN_EXE, and
TEXT_INFER_EXE based on BUILD_DIR.

In `@scripts/inspect_tokenizer.py`:
- Around line 6-8: Update the vocab_path definition in inspect_tokenizer.py to
reference the existing data/gpt_bpe.json artifact instead of the repository-root
bpe_compatible.json, ensuring the subsequent open call loads the actual
vocabulary file.

In `@scripts/test_tokenizer.py`:
- Line 1: 删除 scripts/test_tokenizer.py，或将其真正重命名为
inspect_tokenizer.py，确保仓库中不再存在会被 pytest 默认收集的 test_tokenizer.py 占位文件。

In `@test_topo.cpp`:
- Around line 6-47: 将 test_topo.cpp 中自定义的 TensorNode、手写 stack/visited/expanded
拓扑遍历替换为生产代码中的 nn::TensorNode 和实际 Tensor 图构造；通过 Tensor::backward()
作为反向传播入口验证拓扑处理结果。删除测试内复制的遍历逻辑，并复用生产类型、API 及其可观察结果，确保测试能覆盖
include/neuralnet/tensor.h 中的真实实现。
- Around line 1-8: 在 test_topo.cpp 的 TensorNode 定义所在包含区域显式添加 <string>，确保
std::string 直接依赖对应的标准头文件，不依赖 <iostream> 的传递包含。
- Around line 49-52: 在遍历 topo
的测试逻辑中加入对预期顺序（t1、t2、t3）的显式断言或比较，顺序不匹配时返回非零结果而不是仅打印输出；同时将该测试可执行文件注册到 CTest，确保 CI
会执行它。

---

Outside diff comments:
In `@AGENTS.md`:
- Around line 57-68: Update the AGENTS.md “Test structure” and “CMake Options”
documentation to reflect the current test and example targets introduced by the
PR, including test_tensor, test_gpt, test_tokenizer, and the GPT examples.
Remove or replace stale references to test_nn, mnist_train, and mnist_infer,
preferably using the same general target wording as README.

In `@include/neuralnet/tensor.h`:
- Around line 585-595: Update the normalization loop’s feature-index calculation
to match Matrix row-major storage: derive the feature from the flattened index
using batch_size, not num_features. Use this corrected feature index
consistently for mean, inverse standard deviation, gamma, and beta lookups in
the affine and non-affine paths.
- Around line 111-116: 修改 TensorNode 的 backward() 起始梯度逻辑：仅在调用方尚未设置梯度时初始化全 1，保留
TransformerEncoderModule 和 GPTModelModule 通过 out_t.grad() 写入的
grad_sample（包括缩放和实际 logits 梯度）。如采用 seed_ones 参数，默认保持现有行为，并让这些模块调用 backward()
时显式禁用单位梯度初始化。

In `@scripts/token_bench.py`:
- Around line 42-61: 更新合并逻辑：在构建 merge_priority 时使用 merges 的列表顺序作为 rank，而不是用
new_id 比较优先级；选择 pair 时按最低 rank 合并，但仍使用对应的 new_id 替换。同步修改 chunks 的初始 ID 转换，避免
byte_to_id.get(b, b) 将缺失字节回退为可能与 token ID 冲突的裸字节值，改为使用词表中明确且不冲突的字节 ID，或在缺失时显式报错。

In `@tests/test_transformer.cpp`:
- Around line 424-454: Update test_vit_e2e_backward to invoke Tensor::backward()
on the output after assigning loss.backward() to out.grad(), rather than calling
out.node()->backward_op() directly. Keep the existing input gradient shape
assertions so the test verifies gradients propagate through
TransformerEncoderModule and PatchEmbeddingModule to input.

---

Nitpick comments:
In `@include/neuralnet/module.h`:
- Around line 971-981: 为四个复合模块补充 on_mode_change 覆写并向所有子模块传播模式变化：在
include/neuralnet/module.h:971-981 的 TransformerEncoderLayerModule 中依次转发给
norm1_、mha_、norm2_、ffn_；在 include/neuralnet/module.h:1021-1028 的
TransformerEncoderModule 中遍历 layers_ 转发；在 include/neuralnet/module.h:1278-1283 的
GPTBlockModule 中转发给 self_attn_、norm1_、ff_、norm2_；在
include/neuralnet/module.h:1329-1347 的 GPTModelModule 中遍历 blocks_，并转发给 ln_f_ 与
lm_head_，沿用 Sequential 的模式传播行为。

In `@scripts/csv_png.py`:
- Around line 19-23: 在生成图像的流程中，更新 img_array 的缩放逻辑，在转换为 np.uint8 前先将像素值限制到 [0, 1]
范围，确保超出范围的 CSV 值不会发生静默取模溢出；保持 Image.fromarray 和保存流程不变。

In `@scripts/train_ByteZip.py`:
- Line 291: 将最终词表写入逻辑改为原子写入：在训练结果保存处先写入同目录临时文件，完成并关闭后使用 os.replace
替换目标路径，避免中断时留下损坏文件；复用 download_book 中现有的临时文件命名、清理和替换模式。
- Around line 180-189: Update the sampling logic in the total_bytes >
MAX_V2_SCAN_BYTES branch to preserve original span boundaries instead of
concatenating sampled spans into one bytearray. Limit the sampled data to
MAX_V2_SCAN_BYTES while keeping each selected span separate, and ensure the
downstream n-gram scan processes spans independently so no cross-boundary
substrings are generated.

In `@tests/test_tensor.cpp`:
- Around line 114-145: 抽取 tests/test_tensor.cpp 中的 TestEntry、测试表遍历及 main
分发逻辑为公共测试运行器头文件，并让 tensor 测试复用该实现；同时更新 tests/test_transformer.cpp
使用相同的共享符号，保持批量执行、单测试执行、异常报告及未知测试返回值行为不变。
- Line 1: 将 tests/test_tensor.cpp 和 tests/test_transformer.cpp 中重复的 TestEntry
定义及 main 测试运行/分发逻辑抽取到公共头文件（如 tests/test_runner.h）。两个测试文件仅保留各自的 tests[]
表并包含该头文件，确保共享逻辑可供后续 test_gpt.cpp 和 test_tokenizer.cpp 复用。

In `@tests/test_transformer.cpp`:
- Around line 309-318: 为 TransformerEncoderLayerModule 补充与
test_encoder_layer_shape 对应的
test_encoder_layer_backward_shape，用与文件中其他模块一致的方式执行反向传播并断言梯度形状；保持现有前向形状测试不变，并沿用已有反向测试使用的
API 和断言风格。
- Around line 488-514: Refactor the test dispatch logic in main and the tests
collection handling into a shared header reused by test_tensor.cpp, avoiding
duplicated runner code. Ensure test failures from assertions are reported by the
shared runner, and verify the build configuration does not define NDEBUG so
assertions in this test file remain active.
