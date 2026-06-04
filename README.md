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
├── src/
│   ├── infer.cpp
│   └── train.cpp
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
├── mnist_train        # 训练入口
├── mnist_infer        # 推理入口
└── test_nn            # 测试入口（24 个 CTest 用例）
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
./build/src/mnist_train

# 从已有模型恢复训练
./build/src/mnist_train --load mnist_model.bin

# 指定保存路径
./build/src/mnist_train --save my_model.bin
```

### 验证安装

```bash
ctest --test-dir build --output-on-failure    # 跑全部 24 个测试
cmake --install build --prefix /tmp/nn-test   # 装到临时目录验证 install 流程
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
| `ENABLE_ASAN` | OFF | 启用 AddressSanitizer（必须搭配 Debug 或 RelWithDebInfo，不能与 Release 同用） |
| `ENABLE_UBSAN` | OFF | 启用 UndefinedBehaviorSanitizer（同上） |

## 网络结构

```
输入 (784) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(64) → ReLU → Linear(10)
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
| `save_model` / `load_model` | 二进制模型序列化 |
