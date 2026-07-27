Verify each finding against current code. Fix only still-valid issues, skip the
rest with a brief reason, keep changes minimal, and validate.

Inline comments:
In `@include/neuralnet/module.h`:
- Around line 100-106: Update the LinearModule weight initialization around the
local rng so instances do not all produce identical matrices: replace the
per-instance fixed seed with a shared random engine, while preserving the
existing uniform distribution range and optionally allowing an externally
configured seed for reproducibility.

In `@include/neuralnet/tensor.h`:
- Around line 296-305: Update include/neuralnet/tensor.h lines 296-305 in linear
to use column-major indexing out_span[r + c * rows] and short-circuit when bias
has no node (node_ == nullptr). Also update include/neuralnet/tensor.h lines
572-582 in batchnorm1d so the feature index uses idx % num_features, preserving
column-major layout throughout.
- Around line 1-9: 添加缺失的标准库直接依赖：在 include/neuralnet/tensor.h 的头文件区域加入 <cmath> 和
<random>，并删除使用 std::cout 的调试输出（无需为其新增 <iostream>）；在 include/neuralnet/module.h
的头文件区域加入 <cmath> 和 <random>，确保各自使用的数学与随机数设施独立可编译。
- Around line 166-173: Remove both std::cout debug statements from the backward
propagation block in add, including the out->grad and in1->grad diagnostics.
This eliminates the undeclared iostream dependency and prevents dereferencing
in1->grad or accessing (0,0) for empty tensors; leave the existing gradient
accumulation behavior unchanged.

In `@scripts/gui.py`:
- Around line 910-911: 在 scripts/gui.py#L910-L911 的异常处理回调中，将 lambda
注册时的错误消息通过默认参数固化，避免延迟执行时访问已被删除的 e；同样修改 scripts/gui.py#L1055-L1056 的 log_fn
回调，确保两处都按捕获时的消息记录错误。
- Around line 17-25: 修正脚本顶部的 BUILD_DIR 和可执行文件常量：从仓库根目录定位 build，并将 MNIST 目标指向
build/examples/mnist 下的 mnist_train、mnist_infer，将文本训练/推理目标改为 build/examples/gpt
下的 gpt_train、gpt_infer；保留现有 Windows 后缀逻辑，使
_start_training、_start_infer、_recognize_drawing、_start_text_training 和
_start_text_infer 能找到实际产物。

In `@scripts/token_bench.py`:
- Around line 38-63: Update the BPE encoding path around encode_bpe and
encode_auto so initial byte values are converted to their corresponding
single-byte vocabulary IDs before merge lookup. Build byte_to_id from the
tokenizer vocabulary returned by load_tokenizer by mapping each length-one token
to its ID, pass it into encode_auto/encode_bpe, and preserve the existing
merge-priority behavior using vocabulary IDs throughout.

In `@scripts/train_bpe.py`:
- Around line 57-60: 调整临时文件处理流程：在 tokenizer.save 调用前关闭
NamedTemporaryFile，避免文件句柄仍占用路径；围绕保存和后续使用加入 try/finally，并在 finally 中通过 os.unlink
清理临时文件。更新 scripts/train_bpe.py 的导入以包含 os，并删除原先位置的 unlink，确保成功和异常路径都能完成清理。
- Around line 40-43: 更新 scripts/train_bpe.py 中构建 vocab 的逻辑，避免在遍历 vocab_str_to_id
时直接对所有 token_str 使用 UTF-8 编码。对 ByteLevel token 使用其反向字节映射还原原始字节（如
Ġ→0x20、Ċ→0x0a），对普通特殊 token 保留原样 UTF-8 编码，并继续写入 vocab[tid]。

---

Outside diff comments:
In `@examples/gpt/infer.cpp`:
- Around line 77-86: Validate that the values for --d-model, --num-heads,
--num-layers, --d-ff, and --seq-len are positive before converting them to
std::size_t and assigning them to cfg. Reuse the existing --max-tokens
validation behavior and reject invalid or negative values before they can reach
build_gpt_model.

---

Minor comments:
In `@AGENTS.md`:
- Line 16: 更新 AGENTS.md 中与本 PR 不一致的指导内容：将测试数量和构建目标补充
test_gpt、test_tokenizer、test_tensor；在组件清单加入
LayerNorm、MultiHeadAttention、GPTBlock、GPTModel；在测试文件和模块清单加入对应
GPT、tokenizer、tensor 文件；删除 GPT/decoder 尚未实现的过时描述，并补充 V3 ModelSpec 序列化说明，使其与
README 及当前实现一致。

In `@include/neuralnet/module.h`:
- Around line 296-330: 修复 BatchNorm eval 模式下 `normalized_tensor` 反向传播被静默截断的问题。更新
`normalized_tensor.node()->backward_op` 的设置逻辑，使其在 `input.requires_grad()`
且非训练模式时也注册反向函数；训练模式保留现有批统计公式，eval 模式使用 `running_var` 计算 `inv_std` 并按 `dx = g *
inv_std` 累积输入梯度，与 `tensor.h::batchnorm1d` 保持一致。

In `@README.md`:
- Around line 284-296: 更新 README 的神经网络组件表，移除已删除的 nn::CausalSelfAttention
条目，并改为记录使用 causal=true 构造的 nn::MultiHeadAttention。保持其余 GPT 组件和说明不变。
- Around line 44-71: 更新 README.md 中的项目结构和“构建产物”列表，在 tests 目录及 build/tests
目录分别补充新增的 test_tensor 和既有的 test_transformer；同时为新增的 build/代码围栏添加 text 语言标识，消除
MD040 检查问题。

In `@scripts/download_books.py`:
- Around line 48-58: Update the download/write flow around strip_gutenberg and
filepath so content is written to a temporary file in the same directory, then
atomically renamed to filepath only after the write succeeds. Ensure failed
downloads clean up the temporary file and never leave a partial cache file that
later runs can reuse.

In `@scripts/extract_digits.py`:
- Around line 21-26: 在处理 row[0] 的 extract_digits 主流程中，先校验并转换 label
为纯数字值，再使用其规范化后的字符串创建 label_dir；遇到非数字标签时跳过或按现有无效行处理逻辑处理，确保任何标签都不能通过路径遍历写入
output_dir 之外，并保持后续 key=int 流程可正常执行。

In `@scripts/test_tokenizer.py`:
- Around line 1-10: Update the vocabulary inspection script to load the
repository’s provided BPE vocabulary via a with-managed file handle instead of
hardcoding the current working directory file. Read ascii_start defensively
because the vocabulary JSON may contain only vocab, and prevent execution during
pytest collection by renaming the script from the test naming pattern to an
inspection-oriented name. In the parts handling near the reported line, split
the single-line compound statement into separate statements to satisfy Ruff
E701.

In `@scripts/token_bench.py`:
- Around line 218-221: Remove the unnecessary try/except around
token_bytes.decode in the token string conversion flow. Decode directly with
UTF-8 and errors='replace', preserving the existing token_str result while
eliminating the unreachable bare-except branch and its exception swallowing.

In `@scripts/train_bpe.py`:
- Around line 31-49: Update the return annotation of extract_vocab_hex to
describe its actual two-element return value: the vocabulary hex mapping and the
vocabulary length. Keep the existing tuple return and its element types
unchanged.

In `@scripts/train_ByteZip.py`:
- Line 268: 更新训练输出流程中 Path.write_text 的调用，显式指定 encoding="utf-8"，使
scripts/train_ByteZip.py 与 train_vocab.py 的写入行为一致，并确保生成的 JSON 始终采用 UTF-8 编码。
- Line 21: 同步更新 MAX_V2_SCAN_BYTES 及其行内注释，使常量实际值与注释描述一致；保留预期的扫描字节数，不要留下相互矛盾的容量说明。

In `@scripts/train_vocab.py`:
- Around line 1-12: Update the module docstring to match the current
implementation: remove the unsupported --vocab-size option and describe that
high-frequency tokens are assigned before ASCII characters fill remaining
vocabulary slots. Keep the documented output behavior and valid --output usage
accurate without changing implementation.
- Around line 47-51: Update the vocabulary statistics in the relevant training
flow: remove the unnecessary f-string prefix from the 100% coverage print, and
calculate word_count from the words actually inserted into vocab after excluding
special-token collisions such as <unk>, <pad>, and <num>, so ascii_count cannot
become negative or misrepresent the vocabulary breakdown.

In `@tests/test_tokenizer.cpp`:
- Around line 62-73: 修正 tests/test_tokenizer.cpp 中 decode 预期输出逻辑的注释，使各步骤与最终断言
"hello worldA 中 test" 一致：第 3 步及后续步骤均不得错误地添加初始前导空格，同时保留单词之间实际需要的空格说明。

---

Duplicate comments:
In `@include/neuralnet/layer.h`:
- Around line 1683-1685: 在该位置嵌入计算逻辑中，避免序列长度超过 seq_len_ 时静默使用零向量；参照
PositionalEncoding::forward 的行为，在 t 越界时显式抛出 std::invalid_argument。完成边界校验后，将 pe
的三元表达式改为直接调用 pos_emb_.at_unchecked(t, d)，保持正常范围内的计算不变。

---

Nitpick comments:
In `@include/neuralnet/tensor.h`:
- Around line 119-134: Replace the recursive build_topo traversal in the Tensor
graph construction with an explicit-stack iterative traversal. Preserve the
existing visited-set behavior, child traversal order, null-node handling, and
postorder insertion into topo so the resulting topological order remains
unchanged without risking call-stack overflow.

In `@scripts/csv_png.py`:
- Around line 5-24: Refactor the script’s top-level CSV-to-image flow to use
argparse-provided input and output paths instead of hardcoded
"./build/digit.csv" and "./build/output.png" values. Add a main-entry guard so
argument parsing and file processing run only when the script is executed
directly, while preserving the existing pixel validation, reshaping, conversion,
and image-saving behavior.

In `@scripts/extract_digits.py`:
- Around line 16-17: Update both CSV open calls in the extraction flow to
explicitly specify newline='' and a consistent encoding, including the reader
around csv.reader and the corresponding writer-side call. Preserve the existing
CSV read/write behavior while ensuring embedded newlines and cross-platform text
encoding are handled consistently.

In `@scripts/gui.py`:
- Around line 384-398: Remove the unused dist calculation from _paint_at and
move the sigma-derived constant setup outside the nested pixel loops, while
preserving the existing intensity calculation and drawing behavior.

In `@scripts/save_dataset.py`:
- Around line 7-13: Update export_to_csv so pixel values are rounded to four
decimal places before writer.writerow emits them, while preserving the existing
0–1 floating-point representation and label ordering.

In `@scripts/token_bench.py`:
- Line 132: Update the max_len parameter annotation in evaluate to explicitly
allow None by using int | None, while preserving its default value and existing
function behavior.
- Around line 44-45: 更新合并优先级表构建逻辑：在遍历 merges 时让 merge_priority 直接将每个 (a, b)
映射到对应的 new_id，避免未使用的循环变量。随后在使用 best_priority 的逻辑中直接取得该映射值，删除通过
merges[best_priority][2] 的回查。

In `@scripts/train_ByteZip.py`:
- Around line 158-166: 在 build_v2_from_spans 的采样逻辑中避免直接修改调用方传入的 spans：先创建 spans
的副本，再对该副本执行 random.shuffle，并使用副本完成采样；保持 MAX_V2_SCAN_BYTES 限制和最终 spans 赋值行为不变。
- Around line 32-57: Reduce scan_v1’s memory usage for large corpora by adding
bounded processing before populating freq, left_ctx, and right_ctx: process
oversized tokenized input in chunks or sample a limited subset, and prune
low-frequency substrings before retaining their context sets. Preserve existing
substring enumeration and context semantics for the processed data while
ensuring memory scales within a defined bound.

In `@tests/test_tensor.cpp`:
- Around line 117-133: Update the test_tensor main entry point to match the
exception-handling behavior of test_gpt.cpp and test_tokenizer.cpp: wrap test
execution in try/catch for std::exception, print the failure reason, and return
a failure status instead of terminating. Preserve the existing named-test
dispatch while also supporting no-argument execution of all tests, and add the
required standard headers including string and stdexcept.
