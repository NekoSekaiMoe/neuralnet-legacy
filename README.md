# neuralnet.cpp

一个轻量级 C++17 神经网络库，从头实现（无第三方深度学习框架依赖），**列主序矩阵**（column-major）存储，支持并行计算。

## 项目结构

```
neuralnet.cpp/
├── .gitignore
├── CMakeLists.txt
├── README.md
├── cmake/
│   └── neuralnet-config.cmake.in     # find_package 模板
├── include/
│   └── neuralnet/
│       ├── layer.h
│       ├── loss.h
│       ├── lr_scheduler.h
│       ├── matrix.h
│       ├── optimizer.h
│       ├── model/
│       │   ├── io.h
│       │   └── model.h
│       └── nn/
│           ├── config.h
│           └── nn.h
├── examples/
│   └── mnist/
│       ├── CMakeLists.txt
│       ├── infer.cpp
│       └── train.cpp
├── tests/
│   ├── CMakeLists.txt
│   └── test_nn.cpp
├── csv_png.py
├── extract_digits.py
└── save_dataset.py
```

## 构建产物

```
build/
├── examples/
│   └── mnist/
│       ├── mnist_train        # 训练入口
│       └── mnist_infer        # 推理入口
└── tests/
    └── test_nn                # 测试入口（60 个 CTest 用例）
```

## 依赖

- **编译器**: 支持 C++17 的 GCC 11+ / Clang 14+ / MSVC 19.30+（本项目在 Linux + GCC/Clang 上 CI 验证）
- **C++ 标准**: C++17
- **构建工具**: CMake 3.16+ / Ninja（推荐）
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
# 从头开始训练
./build/examples/mnist/mnist_train

# 从已有模型恢复训练
./build/examples/mnist/mnist_train --load mnist_model.bin

# 指定保存路径
./build/examples/mnist/mnist_train --save my_model.bin
```

### 验证安装

```bash
ctest --test-dir build --output-on-failure       # 跑全部 60 个测试
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
#include <iostream>

int main() {
    nn::Matrix input(4, 3);          // 4 行（in_features），3 列（batch）
    nn::Linear layer(4, 2);          // in=4, out=2
    auto out = layer.forward(input);
    std::cout << "out: " << out.rows() << "x" << out.cols() << std::endl;
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

```
输入 (784) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(64) → ReLU → Linear(10)
```

## 新功能（v2 特性）

5 大新功能 + v2 模型格式 + 3 项性能优化：

### 5 大新功能

| 功能 | API | 文件 |
|------|-----|------|
| **DataLoader** (F1) | `nn::DataLoader(ds, batch_size, shuffle, drop_last, seed)` | `dataloader.h` |
| **BatchNorm 1D+2D** (F2) | `nn::BatchNorm1d(num_features, eps, momentum, affine)` | `layer.h` |
| **`clip_grad_norm_`** (F3) | `nn::clip_grad_norm_(grads, max_norm, eps)` | `grad_clip.h` |
| **`Model::summary()`** (F4) | `model.summary()` 或 `nn::summary(model, &stream)` | `summary.h` |
| **`Model::train()/eval()`** (F6) | `model.train()` / `model.eval()` / `model.is_training()` | `model.h` |

### v2 模型格式

- 文件 magic 不变 (`0x4E4E4E4E`)，版本号 2
- 额外写入每层的非参数状态（BatchNorm 的 `running_mean`、`running_var`、`num_batches_tracked`）
- v1 文件向后兼容，可正常读取
- 升级到 v2 后：含 BatchNorm 的模型推理前需调用 `model.eval()` 使用 running stats

### 3 项性能优化（Linear 层）

- **P1**: bias 加法从逐元素 `idx / cols` 整数除法改为外行-内列 stride 循环
- **P2**: `Linear::forward` 从 2 次内存分配（product + result）改为 1 次（matmul 结果直接作为输出，bias 原地加）
- **P3**: `Linear::backward` 从 3 个朴素 O(n³) 三重循环改为 `Matrix::matmul` 调用（已使用 BLOCK_SIZE=64 的 blocked GEMM）

### 测试覆盖

**60 个 CTest**（之前 31 + 4 大新功能测试 28 + 性能测试 1）：

```
ctest --test-dir build --output-on-failure       # 跑全部 60 个测试
```

## 提供的组件

| 组件 | 说明 |
|------|------|
| `nn::Matrix` | 列主序矩阵，支持并行加/减/乘/转置 |
| `nn::Linear` | 全连接层（含 Xavier 初始化） |
| `nn::ReLU` | ReLU 激活函数 |
| `nn::MSELoss` | 均方误差损失 |
| `CrossEntropyLoss` | 交叉熵损失（含数值稳定 Softmax，在 `train.cpp` 中定义） |
| `nn::SGD` | 随机梯度下降优化器 |
| `nn::Model::save` / `nn::Model::load` | 成员函数，二进制模型序列化 |
| `nn::save_model` / `nn::load_model` | 自由函数，委托给 Model 成员函数 |
