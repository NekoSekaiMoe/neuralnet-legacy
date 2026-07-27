# neuralnet.cpp

一个轻量级 C++17 神经网络库，从头实现（无第三方深度学习框架依赖），**列主序矩阵**（column-major，行=特征维、列=batch）存储，通过 TBB 提供 `std::execution::par_unseq` 并行计算后端。

## 项目结构

```
neuralnet.cpp/
├── CMakeLists.txt
├── README.md
├── cmake/
│   └── neuralnet-config.cmake.in     # find_package 模板
├── include/
│   └── neuralnet/
│       ├── dataloader.h              # F1: Dataset / TensorDataset / DataLoader
│       ├── grad_clip.h               # F3: clip_grad_norm_
│       ├── layer.h                   # Layer 基类 + Linear / ReLU / LeakyReLU /
│       │                             #          Sigmoid / Tanh / GELU / Dropout /
│       │                             #          BatchNorm1d / BatchNorm2d
│       ├── loss.h                    # MSELoss / CrossEntropyLoss
│       ├── lr_scheduler.h            # StepLR / CosineAnnealingLR / ExponentialLR
│       ├── matrix.h                  # 列主序 Matrix（matmul / matmul_TN /
│       │                             #          matmul_NT / norm / rowwise_* /
│       │                             #          colwise_*）
│       ├── optimizer.h               # SGD / SGD_w_Momentum / Adam
│       ├── summary.h                 # F4: summary(model) / Model::summary()
│       ├── model/
│       │   ├── io.h                  # save_model / load_model 自由函数
│       │   └── model.h               # Model（F5/F6） + train()/eval() + v1/v2 I/O
│       └── nn/
│           ├── config.h              # NN_EXEC_POLICY / BLOCK_SIZE / counting_iterator
│           └── nn.h                  # 聚合头：包含上述所有 + one_hot
├── examples/
│   ├── mnist/                      # MNIST 分类示例
│   │   ├── CMakeLists.txt
│   │   ├── infer.cpp                 
│   │   └── train.cpp                 
│   └── gpt/                        # GPT 文本生成示例
│       ├── CMakeLists.txt
│       ├── train.cpp                 # GPT 模型训练
│       ├── infer.cpp                 # GPT 文本生成推理
│       ├── tokenizer_train.cpp       # ByteZip 词表训练
│       └── tokenizer_infer.cpp       # 分词器测试
├── tests/
│   ├── CMakeLists.txt
│   ├── test_gpt.cpp                  # GPT 模型组件测试
│   ├── test_tokenizer.cpp            # 分词器组件测试
│   └── test_nn.cpp                   # 基础网络 CTest 用例
├── scripts/
│   └── *.py                          # 数据下载、词表训练等辅助工具
└── data/
    └── gpt_bpe.json                  # BPE 词表数据
```

## 构建产物

```
build/
├── examples/
│   ├── mnist/
│   │   ├── mnist_train        # MNIST 训练入口
│   │   └── mnist_infer        # MNIST 推理入口
│   └── gpt/
│       ├── gpt_train          # GPT 训练入口
│       ├── gpt_infer          # GPT 交互式推理入口
│       ├── gpt_tokenizer_train # 词表训练工具
│       └── gpt_tokenizer_infer # 分词器调试工具
└── tests/
    ├── test_nn                # 基础测试
    ├── test_gpt               # GPT 测试
    └── test_tokenizer         # 分词器测试
```

## 依赖

- **编译器**: 支持 C++17 的 GCC 11+ / Clang 14+ / MSVC 19.30+（本项目在 Linux + GCC/Clang 上 CI 验证）
- **C++ 标准**: C++17
- **构建工具**: CMake 3.25+ / Ninja（推荐）
- **依赖库**: TBB 2021.5+（提供 libstdc++ `<execution>` 并行策略的后端；老版本有兼容垫片）

## 构建与运行

### 构建

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### 准备数据

```bash
pip install pillow
python save_dataset.py
```

### 运行训练

```bash
# 从头开始训练（默认数据集目录 ./mnist_data，模型保存到 ./mnist_model.bin）
./build/examples/mnist/mnist_train

# 从已有模型恢复训练
./build/examples/mnist/mnist_train --load mnist_model.bin

# 指定模型保存路径
./build/examples/mnist/mnist_train --save my_model.bin

# 指定数据集目录
./build/examples/mnist/mnist_train --dataset /path/to/mnist_data
```

### 运行推理

`mnist_infer` 接受与训练数据同格式的 CSV（一行 784 个归一化到 [0,1] 的像素值）：

```bash
./build/examples/mnist/mnist_infer mnist_model.bin image.csv
# 输出: Predicted digit: <0-9>
```

### 验证安装

```bash
ctest --test-dir build --output-on-failure       # 跑全部 67 个测试
cmake --install build --prefix /tmp/nn-test      # 装到临时目录验证 install 流程
ls /tmp/nn-test/bin 2>&1                         # 应当 "No such file or directory"（demos 不安装）
```

## 作为库消费 (find_package)

本项目既可以本地构建（`cmake -B build`），也可以安装后被其它 CMake 项目以 `find_package` 方式消费。

### 安装

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /usr/local      # 或任意 prefix
```

### 在消费者项目中

```cmake
cmake_minimum_required(VERSION 3.16)
project(myapp CXX)
find_package(neuralnet REQUIRED CONFIG)
add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE neuralnet::neuralnet)
```

`neuralnet::neuralnet` 是一个 INTERFACE 目标，会传递地提供：

- 头文件路径（`include/neuralnet/...`）
- C++17 编译特性
- TBB 链接（`TBB::tbb` 作为 INTERFACE 依赖）

### 消费者示例

```cpp
#include <neuralnet/nn/nn.h>
#include <neuralnet/model/io.h>
#include <iostream>

int main() {
    // 构造与 examples/mnist/train.cpp 一致的网络
    nn::Model model;
    model.add<nn::Linear>(784, 64)
         .add<nn::BatchNorm1d>(64)
         .add<nn::ReLU>()
         .add<nn::Linear>(64, 64)
         .add<nn::BatchNorm1d>(64)
         .add<nn::ReLU>()
         .add<nn::Linear>(64, 64)
         .add<nn::BatchNorm1d>(64)
         .add<nn::ReLU>()
         .add<nn::Linear>(64, 10);

    // 单样本推理：(784, 1) → (10, 1) logits
    nn::Matrix img(784, 1);
    // ... 填充 img ...
    model.eval();
    auto out = model.forward(img);

    std::size_t pred = 0;
    for (std::size_t i = 1; i < 10; ++i) {
        if (out.at_unchecked(i, 0) > out.at_unchecked(pred, 0)) pred = i;
    }
    std::cout << "pred = " << pred << std::endl;

    // 训练循环范式：
    //   model.train();
    //   for (auto [x_batch, y_batch] : loader) {
    //       ce_loss.forward(model.forward(x_batch), y_batch);
    //       model.backward(ce_loss.backward());
    //       nn::clip_grad_norm_(model.param_gradients(), 1.0);
    //       optimizer.step();
    //       optimizer.zero_grad();
    //   }
    //   model.eval();
    return 0;
}
```

## 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `BUILD_TESTING` | `${PROJECT_IS_TOP_LEVEL}` | 关闭后不构建 `test_nn` 也不注册 CTest |
| `BUILD_EXAMPLES` | ON | 关闭后不构建 `examples/mnist/` 下的 demo 程序（不影响库本身） |
| `ENABLE_ASAN` | OFF | 启用 AddressSanitizer（必须搭配 Debug 或 RelWithDebInfo，不能与 Release 同用） |
| `ENABLE_UBSAN` | OFF | 启用 UndefinedBehaviorSanitizer（同上） |

## 网络结构

`examples/mnist/train.cpp` 与 `infer.cpp` 共用以下网络（输入 784、输出 10 logits）：

```
Linear(784, 64) → BatchNorm1d(64) → ReLU
  → Linear(64, 64) → BatchNorm1d(64) → ReLU
  → Linear(64, 64) → BatchNorm1d(64) → ReLU
  → Linear(64, 10)
```

训练超参：batch size 64、5 epoch、`nn::Adam(lr=0.001)`、`nn::CrossEntropyLoss`、含 `model.train()` / `model.eval()` 切换、DataLoader 启用 shuffle。

## v2 特性（5 大新功能 + v2 模型格式 + 3 项性能优化）

### 5 大新功能

| 功能 | API | 头文件 |
|------|-----|--------|
| **DataLoader / Dataset** (F1) | `nn::DataLoader(ds, batch_size, shuffle, drop_last, seed)` + `nn::TensorDataset` | `dataloader.h` |
| **BatchNorm 1D+2D** (F2) | `nn::BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=true)` / `nn::BatchNorm2d` | `layer.h` |
| **`clip_grad_norm_`** (F3) | `nn::clip_grad_norm_(grads, max_norm, eps=1e-6)` → 返回 pre-clip 总范数 | `grad_clip.h` |
| **`Model::summary()`** (F4) | `model.summary()` 或 `nn::summary(model, &stream)` | `summary.h` |
| **`Model::train()/eval()`** (F6) | `model.train()` / `model.eval()` / `model.is_training()`（向 `Layer::on_mode_change` 传播） | `model.h` |

### v2/v3 模型格式（F5）

- **V2**: 写入每层的非参数状态（如 BatchNorm 的 running stats）。
- **V3**: 新增了 `ModelSpec` 规范（如 GPT 架构特有的 `d_model, num_heads, num_layers, seq_len` 等），在模型参数前写入序列化的超参，实现按配置自动初始化。
- 文件 magic 不变 (`0x4E4E4E4E`)，版本号递增。
- 向后兼容：V1/V2 格式文件均可正常读取并兼容旧代码逻辑。

### 3 项性能优化（Linear 层）

- **P1**: bias 加法从逐元素 `idx / cols` 整数除法改为外行-内列 stride 循环（`bias_val` 提到内循环外，一次加载）
- **P2**: `Linear::forward` 从 2 次内存分配（product + result）改为 1 次（matmul 结果直接作为输出，bias 原地加）
- **P3**: `Linear::backward` 改用 `Matrix::matmul_TN` / `matmul_NT` 融合转置 matmul（内部用 `BLOCK_SIZE=64` 的 blocked GEMM），省去中间 `W^T` / `input^T` 的内存分配

### 测试覆盖

**67 个 CTest**（矩阵与基础层 18 + BatchNorm 13 + DataLoader 4 + grad_clip 4 + summary 3 + 其余 25）：

```bash
ctest --test-dir build --output-on-failure       # 跑全部 67 个测试
```

## 提供的组件

### 矩阵与工具（`<neuralnet/matrix.h>` / `<neuralnet/nn/nn.h>`）

| 组件 | 说明 |
|------|------|
| `nn::Matrix` | 列主序矩阵（行=特征维，列=batch），`par_unseq` 并行加/减/逐元素乘、转置、matmul |
| `nn::Matrix::matmul_TN` | `this^T * other`；融合转置 matmul（Linear::backward 关键路径） |
| `nn::Matrix::matmul_NT` | `this * other^T`；融合转置 matmul（Linear::backward 关键路径） |
| `nn::Matrix::norm` | Frobenius 范数 `sqrt(sum(x²))` |
| `nn::Matrix::rowwise_sum/mean` | 按行（特征维）聚合，BatchNorm 内核使用 |
| `nn::Matrix::colwise_sum/mean` | 按列（batch 维）聚合 |
| `nn::one_hot(indices, mat_size)` | 由索引向量生成 `(mat_size, N)` one-hot 矩阵 |

### 层（`<neuralnet/layer.h>`）

| 组件 | 说明 |
|------|------|
| `nn::Layer` | 抽象基类（forward / backward / parameters / name / on_mode_change / save_state / param_count） |
| `nn::Linear` | 全连接层，Xavier 均匀初始化；1 次内存分配的 forward（P2） |
| `nn::ReLU` / `LeakyReLU` / `Sigmoid` / `Tanh` / `GELU` | 激活函数 |
| `nn::Dropout` | 训练时缩放倒置 dropout；`on_mode_change` 自动同步 `training_` 标志 |
| `nn::BatchNorm1d` | 输入 `(num_features, batch)`；可选 affine；持久化 running stats（v2） |
| `nn::BatchNorm2d` | 复用 `BatchNorm1d`（视 `(C, N*H*W)` 为 `(C, batch)`）；仅 `name()` 差异 |
| `nn::LayerNorm` | 层归一化，支持 `(features, batch_size)` 输入 |
| `nn::CausalSelfAttention`| 带因果掩码的多头自注意力机制（Q, K, V 映射） |
| `nn::GPTBlock` | 前置归一化架构的 GPT 解码器块 |
| `nn::GPTModel` | 完整的 GPT 语言模型实现，内置 `generate()` 生成接口 |

### 分词器（`<neuralnet/tokenizer.h>`）

| 组件 | 说明 |
|------|------|
| `nn::Tokenizer` | 抽象基类（`encode` / `decode`） |
| `nn::CharTokenizer` | 朴素字符级分词器（基于 UTF-8 映射） |
| `nn::BPETokenizer` | 标准 BPE 分词器，支持解析 HF tokenizers 格式 JSON 词表 |
| `nn::ByteZipTokenizer` | 优化的两阶段高频字节对和块分词器（带原生训练支持） |

### 损失（`<neuralnet/loss.h>`）

| 组件 | 说明 |
|------|------|
| `nn::MSELoss` | 均方误差 |
| `nn::CrossEntropyLoss` | 数值稳定 softmax + NLL；输入 logits + one-hot target |

### 优化器（`<neuralnet/optimizer.h>`）

| 组件 | 说明 |
|------|------|
| `nn::SGD` | 随机梯度下降 |
| `nn::SGD_w_Momentum` | SGD + 动量（β 默认 0.9） |
| `nn::Adam` | Adam (Kingma & Ba 2015)，带偏差校正；默认 `lr=1e-3, β1=0.9, β2=0.999, ε=1e-8` |

### 学习率调度器（`<neuralnet/lr_scheduler.h>`）

| 组件 | 说明 |
|------|------|
| `nn::StepLR` | 每 `step_size` 个 epoch 乘以 `gamma` |
| `nn::CosineAnnealingLR` | 余弦退火 `lr_min + 0.5(lr_max-lr_min)(1+cos(π·t/T_max))` |
| `nn::ExponentialLR` | 每 epoch 乘以 `gamma` |

### 数据（`<neuralnet/dataloader.h>`）

| 组件 | 说明 |
|------|------|
| `nn::Dataset` | 抽象基类（`size()` / `get(idx) → {Matrix, Matrix}`） |
| `nn::TensorDataset` | dense 矩阵版：features `(F, N)` + labels `(L, N)` |
| `nn::DataLoader` | 批迭代器；支持 `shuffle` / `drop_last` / `seed`（`seed=0` 时使用 `std::random_device`） |

### 模型与 I/O（`<neuralnet/model/model.h>` + `model/io.h`）

| 组件 | 说明 |
|------|------|
| `nn::Model` | `add<L>(args)...` 流式构建；`forward / backward / parameters / param_gradients` |
| `nn::Model::train() / eval() / is_training()` | F6：模式切换并向每层 `on_mode_change` 传播 |
| `nn::Model::summary(out?)` | F4：层表 + Total params（返回字符串，可选写入流） |
| `nn::Model::save / load` | 成员函数，v1/v2 自动分发；路径或 `std::iostream` 重载 |
| `nn::save_model / nn::load_model` | 自由函数，委托给 Model 成员函数 |

### 工具（`<neuralnet/grad_clip.h>`、`<neuralnet/summary.h>`）

| 组件 | 说明 |
|------|------|
| `nn::clip_grad_norm_(grads, max_norm, eps)` | F3：总范数超阈值则按比例缩放所有梯度；NaN/Inf 抛错 |
| `nn::summary(model, out?)` | 自由函数，与 `Model::summary()` 等价 |
