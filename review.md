Inline comments:
In `@examples/gpt/infer.cpp`:
- Line 39: Validate the parsed max_tokens value before passing it to generate,
ensuring negative values are rejected rather than implicitly converted to
std::size_t. Update the argument-parsing paths around max_tokens, including the
default declaration and the --max-tokens handling, to require a positive value
and terminate with a clear input error when validation fails.
- Line 20: 统一更新 infer.cpp 中相关模型格式说明文字，将“V2 格式”改为“V3 格式”，包括指定架构参数和自动读取规格的说明，确保与
train.cpp 及 io.h::read_model_spec 的版本判定一致。

In `@examples/gpt/tokenizer_infer.cpp`:
- Around line 337-367: 在 main 中参照 examples/gpt/infer.cpp 的异常处理惯例，用 try/catch
包裹参数解析、tokenizer.load 及后续模式分发流程，捕获运行时异常并输出错误信息后以失败状态退出；确保 tokenizer.load 或
parse_ids 中 std::stoul 的异常不会导致未捕获异常终止程序。

In `@examples/gpt/tokenizer_train.cpp`:
- Around line 126-200: 在 main 中采用项目现有惯例，用 try/catch 包裹 tokenizer.train、编码解码流程及
tokenizer.save 等可能抛出异常的执行逻辑；捕获 std::exception 后输出“错误: ”及
e.what()，并以非零状态退出，避免异常传播至入口导致 terminate。

In `@examples/gpt/train.cpp`:
- Around line 79-104: 在参数解析逻辑中为
cfg.epochs、cfg.lr、cfg.batch_size、cfg.seq_len、cfg.d_model、cfg.num_heads、cfg.num_layers
和 cfg.d_ff 增加数值校验：解析失败或结果不大于 0 时输出明确的参数错误并终止，避免负值转换为异常大的
size_t；保留现有有效参数的赋值行为，并确保 --optimizer 的处理不受影响。
- Around line 223-231: 在训练初始化逻辑中修正 max_start 的边界处理：先验证 all_tokens.size() 大于
cfg.seq_len + 1，再执行无符号减法；文本不足时立即沿用现有错误返回。随后确保 steps_per_epoch 在 cfg.batch_size
大于 max_start 时不会为 0，避免训练循环空转及后续除零，并保持有效数据范围内的随机采样。

In `@include/neuralnet/layer.h`:
- Around line 1887-1918: 修正该批处理反向流程：在每个样本重建缓存时补充调用 lm_head_.forward(x)，并为
blocks_、ln_f_ 与 lm_head_ 的参数及输入梯度建立独立的 batch 累加缓冲区；每次 backward
后累加对应梯度，循环结束后统一写回现有梯度存储，避免赋值语义覆盖前序样本，同时保持 embedding 梯度的累加行为一致。
- Around line 1829-1859: Update Layer::forward to validate that input.rows()
(sl) does not exceed seq_len_ before accessing pos_emb_.at_unchecked in the
token-processing loop. Add explicit handling consistent with the existing
out-of-range token_id fallback, while preserving valid-position behavior and
preventing positional embedding out-of-bounds reads.
- Around line 1913-1916: 在嵌入梯度累加逻辑中修正索引布局：将 grad_token_emb_ 的索引改为 tid * d_model_
+ d，将 grad_pos_emb_ 的索引改为 t * d_model_ + d；保持现有梯度值 g 和累加行为不变。

In `@include/neuralnet/tokenizer.h`:
- Around line 32-66: Update CharTokenizer::decode_one to accept only token IDs
in the actual encodable byte range (0–255), returning "?" for IDs 256 through
GPT_VOCAB_SIZE - 1 as well as other invalid IDs. Keep vocab_size() unchanged at
GPT_VOCAB_SIZE for the model output vocabulary.

In `@tests/CMakeLists.txt`:
- Around line 149-150: 为 tests/CMakeLists.txt 中的 test_tokenizer 添加与其他测试目标一致的
target_link_libraries 配置，链接 neuralnet::neuralnet，使其继承该 INTERFACE
库的编译特性和依赖；保持现有可执行文件及 include 配置不变。

---

Nitpick comments:
In `@examples/gpt/CMakeLists.txt`:
- Around line 3-7: Update the MSVC and non-MSVC compile-option blocks in
examples/gpt/CMakeLists.txt so the UNDEBUG option is either removed entirely or
applied consistently to all four example targets, rather than only gpt_train;
preserve the existing platform-specific option forms.

In `@examples/gpt/train.cpp`:
- Line 159: Update the tokenizer initialization in train.cpp to accept a --vocab
<path> command-line option while retaining data/gpt_bpe.json as the default when
omitted. Replace the hardcoded path passed to tokenizer.load_vocab, and apply
the same option and default behavior in infer.cpp.
- Around line 248-251: 将训练采样逻辑中的 std::uniform_int_distribution 对象移到 batch
元素循环外，在进入以 b 遍历 cfg.batch_size 之前完成一次构造；循环内继续复用该分布生成 start，保持现有采样范围和行为不变。

In `@include/neuralnet/layer.h`:
- Line 1943: 修改 generate 中通过 forward 计算 logits 的路径，避免覆盖训练反向传播所需的
stored_inputs_、stored_tokens_ 和 batch_size_；优先使用不写入训练缓存的推理前向路径，或在调用 forward
前后完整保存并恢复这些成员，确保 generate 在 forward 与 backward 之间执行不会影响反传。
- Around line 1664-1665: 移除类成员 residual1_cache_ 和 residual2_cache_，并在 forward
方法中将所需值改为局部变量；同步更新 forward 内对这两个缓存的赋值和引用，确保不再保留每个 block 的常驻状态。
- Around line 1225-1408: 将 CausalSelfAttention 与既有 MultiHeadAttention
的重复实现合并：复用共享的初始化、softmax、维度校验及 forward/backward 逻辑，并通过构造参数或共享基类控制是否启用因果掩码。确保
MultiHeadAttention 保持非因果行为，而 CausalSelfAttention 仅额外启用上三角屏蔽，避免两套实现后续分叉。

In `@include/neuralnet/model/model.h`:
- Around line 266-282: 统一 Model::load 中 version == 3 头部的解析逻辑，避免与 save() 和 io.h 的
read_model_spec 各自维护 ModelType 及后续字段顺序；抽取并复用共享的头部读取/跳过辅助函数，或直接复用 read_model_spec
已解析的结果，确保字段增删时只需更新一个格式定义且保持现有读取顺序。

In `@include/neuralnet/tokenizer.h`:
- Around line 100-213: 抽取 BPETokenizer::load_vocab 与 ByteZipTokenizer::load
共用的词表 JSON 解析辅助函数，统一处理字符串转义、Unicode 解码、十六进制字节值及 token/id
映射；让两处加载逻辑复用该实现，移除重复的手写解析代码并保持现有词表加载结果不变。

In `@tests/test_gpt.cpp`:
- Around line 77-93: 在 test_gpt_model_shape 附近新增一个小规模 GPTModel
数值梯度校验用例，定义可微损失并通过参数正负 ε 扰动计算有限差分梯度，再与对应的 param_gradients()
项比较且使用适当容差；保留现有形状断言，并覆盖至少一个参数以检测 backward 的索引和累加错误。

In `@tests/test_tokenizer.cpp`:
- Around line 1-41: Add unit-test coverage for BPETokenizer::load_vocab in the
tokenizer test suite, including vocabulary entries with escaped characters,
\uXXXX sequences, and hexadecimal byte restoration. Exercise encode/decode round
trips with the loaded vocabulary and register the new test in the existing tests
array.
