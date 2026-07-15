# GPT / Transformer 全套移植指南

本文档描述如何将上游 `EthanPeng-2048/neuralnet.cpp` 的 GPT / 完整 Transformer 推理管线移植到本 Fork（`neuralnet-legacy`，C++17 / GCC）。

> 当前 Fork 已移植的基础组件：LayerNorm、Softmax、PositionalEncoding、MultiHeadAttention、FeedForward、TransformerEncoderLayer、TransformerEncoder、PatchEmbedding。以下内容基于这些已有组件展开。

---

## 一、待移植组件清单

| 组件 | 上游文件 | 行数 (估) | 依赖 | 移植难度 |
|------|---------|-----------|------|---------|
| `CausalSelfAttention` | `layer.hpp` | ~200 | MultiHeadAttention (已有) | 中 |
| `GPTBlock` | `layer.hpp` | ~80 | CausalSelfAttention, LayerNorm, FeedForward | 低 |
| `GPTModel` | `layer.hpp` | ~300 | GPTBlock, Linear, LayerNorm, Softmax | 中 |
| `CharTokenizer` | `gpt_common.hpp` | ~40 | 无 | 低 |
| `BPETokenizer` | `gpt_common.hpp` | ~180 | 无 | 低 |
| `ByteZipTokenizer` | `tokenizer.hpp` | ~500 | 无 | 中 |
| `ModelSpec + V2 IO` | `model_io.hpp` | ~350 | Model | 中 |
| `text_train.cpp` | `src/text_train.cpp` | ~250 | GPTModel, BPETokenizer, ModelSpec IO | 低 |
| `text_infer.cpp` | `src/text_infer.cpp` | ~200 | GPTModel, BPETokenizer, ModelSpec IO | 低 |
| `tokenizer_train.cpp` | `src/tokenizer_train.cpp` | ~150 | ByteZipTokenizer | 低 |
| `tokenizer_infer.cpp` | `src/tokenizer_infer.cpp` | ~200 | ByteZipTokenizer | 低 |

**总工作量估计**：~2300 行新代码 + 测试

---

## 二、移植步骤

### 步骤 1：CausalSelfAttention

**与 MultiHeadAttention 的区别**：在 softmax 之前，对 attention scores 施加上三角因果 mask（将未来位置设为 `-inf`），使得位置 `p` 只能关注 `≤ p` 的位置。

**C++17 适配要点**：
- 上游在构造时用 `std::views::iota` 生成 mask 矩阵 → 改用 `counting_iterator` 或手动循环
- 返回值 `Result<Matrix>` → `Matrix` + throw

**实现建议**：
```cpp
// 复用已有的 MultiHeadAttention，仅在 forward 中 scores 计算后、softmax 之前
// 加上 causal mask：
for (std::size_t i = 0; i < sl; ++i)
    for (std::size_t j = i + 1; j < sl; ++j)
        scores.set_value_unchecked(i, j, -1e9);
```

可以直接继承 MultiHeadAttention 或新建类，推荐新建类以保持关注点分离。

### 步骤 2：GPTBlock

Pre-norm decoder block，结构类似 TransformerEncoderLayer：

```
input → LayerNorm → CausalSelfAttention → + residual
      → LayerNorm → FeedForward → + residual → output
```

**实现建议**：直接参照已有的 `TransformerEncoderLayer`，将 `MultiHeadAttention` 替换为 `CausalSelfAttention`。

### 步骤 3：GPTModel

完整的 decoder-only Transformer：

```
token IDs → Token Embedding (lookup table) → + Positional Encoding
          → N × GPTBlock
          → LayerNorm
          → LM Head (Linear: d_model → vocab_size)
          → output logits
```

**关键实现细节**：

1. **Token Embedding**：上游用一个 `(d_model, vocab_size)` 的权重矩阵，forward 时做 one-hot → matmul 或直接列索引。C++17 中直接用列索引更高效：
   ```cpp
   // 输入: (seq_len, batch_size)，每个元素是 token ID
   // 输出: (d_model, seq_len * batch_size)
   for (std::size_t col = 0; col < total; ++col) {
       std::size_t token_id = static_cast<std::size_t>(input.at_unchecked(col / batch, col % batch));
       for (std::size_t i = 0; i < d_model; ++i)
           output.set_value_unchecked(i, col, embedding_.at_unchecked(i, token_id));
   }
   ```

2. **generate() 方法**：自回归生成，支持温度采样和贪心解码。需要 `<random>` 做 temperature sampling。

3. **形状约定**：
   - 输入: `(seq_len, batch_size)` — 每个元素是 token ID（double 编码的整数）
   - 内部: `(d_model, seq_len)` — 单样本在 Transformer 层间传递
   - 输出: `(vocab_size, seq_len * batch_size)` — logits

### 步骤 4：分词器

三个分词器均为独立组件，不依赖 nn 命名空间的核心层：

| 分词器 | 特点 | C++17 适配 |
|--------|------|-----------|
| `CharTokenizer` | 最简单，字符级 | 无需适配 |
| `BPETokenizer` | 加载 JSON 词表，word-level fallback to char | 手写 JSON 解析器，C++17 兼容 |
| `ByteZipTokenizer` | 自研频率分词器，V1+V2 两阶段 | `std::ranges::sort` → `std::sort`；`std::ranges::shuffle` → `std::shuffle`；`std::span` → 自定义或指针 |

**建议**：创建 `include/neuralnet/tokenizer.h`，放置所有分词器。

**C++17 关键替换**：
```cpp
// 上游 (C++26):
std::ranges::sort(sorted, comparator);
std::span<const std::size_t> ids

// Fork (C++17):
std::sort(sorted.begin(), sorted.end(), comparator);
const std::vector<std::size_t> &ids  // 或 const std::size_t*, std::size_t
```

### 步骤 5：ModelSpec 序列化

上游的 V2 格式在文件头嵌入 `ModelSpec`（模型架构描述），加载时可先读规格再构建模型。

**与 Fork 现有 V2 格式的冲突**：
- Fork V2：`[magic][version=2][n_params][matrices...][n_state_layers][states...]`
- 上游 V2：`[magic][version=2][model_type][spec_data...][matrices...]`

**建议方案**：引入 V3 格式，合并两种设计：
```
[magic][version=3][model_type][spec_data...][n_params][matrices...][n_state_layers][states...]
```
保留 V1/V2 读取兼容，新文件统一用 V3。

### 步骤 6：训练与推理程序

`text_train.cpp` 和 `text_infer.cpp` 是应用层代码，移植相对简单：

- 命令行解析：上游用手写 `parse_args()`，无 C++26 依赖，可直接复用
- `std::string::starts_with()` (C++20) → 手动检查：`arg.substr(0, 2) == "--"`
- `std::from_chars` 用于快速解析 — C++17 已有

**建议位置**：`examples/gpt/train.cpp` 和 `examples/gpt/infer.cpp`，配合 `examples/gpt/CMakeLists.txt`。

---

## 三、C++26 → C++17 通用替换表

| C++26 / C++20 特性 | C++17 替代方案 |
|--------------------|---------------|
| `std::expected<T, E>` | 抛异常 / 返回裸值 |
| `Result<T>` 返回值 | `T` + `throw std::runtime_error(...)` |
| `std::views::iota(0, n)` | `counting_iterator<size_t>(0), counting_iterator<size_t>(n)` |
| `std::views::zip(a, b, c)` | 索引循环 `for (size_t i = 0; ...)` |
| `std::ranges::sort(v, cmp)` | `std::sort(v.begin(), v.end(), cmp)` |
| `std::ranges::shuffle(v, rng)` | `std::shuffle(v.begin(), v.end(), rng)` |
| `std::span<T>` | `const std::vector<T>&` 或 `T*, size_t` |
| `std::string::starts_with(s)` | `s.substr(0, prefix.size()) == prefix` |
| `set.contains(key)` | `set.find(key) != set.end()` |
| `std::format(...)` | `std::to_string()` / `std::ostringstream` |

---

## 四、测试策略

每个新组件至少需要：

1. **形状测试**：forward/backward 输入输出维度验证
2. **数值测试**：已知输入 → 已知输出 的确定性验证
3. **梯度校验**：有限差分 vs 解析梯度（参见 `test_nn.cpp` 中的 `finite_diff_grad`）
4. **端到端测试**：GPTModel + 分词器 + 训练循环跑几步，验证 loss 下降

**测试文件建议**：
- `tests/test_gpt.cpp` — CausalSelfAttention, GPTBlock, GPTModel
- `tests/test_tokenizer.cpp` — CharTokenizer, BPETokenizer, ByteZipTokenizer

---

## 五、移植顺序建议

```
Phase 1 (基础):
  1. CausalSelfAttention  ← 在 MHA 基础上加 causal mask
  2. GPTBlock              ← 组合已有组件
  3. GPTModel              ← 组合 + Token Embedding + generate()

Phase 2 (分词器):
  4. CharTokenizer         ← 最简单，先跑通
  5. BPETokenizer          ← JSON 词表加载
  6. ByteZipTokenizer      ← 最复杂，可选

Phase 3 (IO + 应用):
  7. ModelSpec V3 IO       ← 设计新格式
  8. text_train/infer      ← 应用层
  9. tokenizer_train/infer ← 应用层
```

每个 Phase 完成后应独立可测试、可提交。

---

## 六、风险与注意事项

1. **上游代码未经测试**：GPT 相关代码没有任何单元测试。移植时必须自行编写测试验证正确性，不能假设上游实现是正确的。

2. **性能差异**：上游用 `SmartPolicy` + 自定义 `ThreadPool` 做并行，Fork 用 `std::execution::par_unseq` + TBB。Transformer 的注意力计算中大量小矩阵运算可能在两种策略下性能表现不同。

3. **Token Embedding 的反向传播**：需要实现 embedding lookup 的梯度散射（scatter-add），这在上游可能用了 one-hot + matmul 的方式（低效但简单）。

4. **内存占用**：GPTModel 的 `generate()` 方法逐 token 生成，每步需要完整的 forward pass。对于长序列，需要考虑 KV-cache 优化（上游未实现）。

5. **gpt_bpe.json 词表文件**：228KB，是运行 BPETokenizer 的必要数据文件。需要一并迁移并放到合适位置（建议 `data/` 目录）。
