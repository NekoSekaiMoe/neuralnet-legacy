# AGENTS.md

This file provides guidance to Gemini for Antigravity when working with code in this repository.

## Build Commands

```bash
# Configure and build (Release)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Debug build with sanitizers
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON -DENABLE_UBSAN=ON
cmake --build build

# Run all tests
ctest --test-dir build --output-on-failure

# Run a single test by name
ctest --test-dir build -R <test_name> --output-on-failure

# Example: ctest --test-dir build -R batchnorm1d_forward_shape --output-on-failure
```

## Dependencies

- C++17 (GCC 11+ / Clang 14+)
- CMake 3.25+, Ninja (recommended)
- TBB 2021.5+ (parallel execution backend for `std::execution::par_unseq`)
- On Ubuntu: `apt install g++ ninja-build libtbb-dev`

## Architecture

This is a header-only C++17 neural network library. All code lives in `include/neuralnet/`.

**Data layout**: Column-major matrices — rows are feature dimensions, columns are batch samples. This convention is consistent throughout the library.

**Execution policy**: Adaptive dispatch ("SmartPolicy", aligned with upstream neuralnet.cpp): element-wise ops call `nn::for_each / nn::transform / nn::transform_reduce / nn::for_range` (`nn/config.h`) — serial loop below `PARALLEL_THRESHOLD` (512K work units, overridable via `-DNN_PARALLEL_THRESHOLD=...`), `std::execution::par_unseq` (TBB) at or above it. GEMM/transpose block loops use `nn::for_blocks` (always parallel — each iteration computes a 64×64 output block). Compile-time escape hatch `-DNN_EXEC_POLICY=std::execution::seq` still forces the parallel branches serial.

**Key design patterns**:
- `nn::Model` uses a type-erased layer stack (`add<LayerType>(args...)`) with virtual dispatch through the `Layer` base class for forward/backward
- Layers implement `forward()`, `backward()`, `parameters()`, and optionally `on_mode_change()` (for train/eval switching) and `save_state()`/`load_state()` (for v2 serialization)
- Matrix operations use blocked GEMM (`BLOCK_SIZE=64`) with fused-transpose variants (`matmul_TN`, `matmul_NT`) to avoid intermediate allocations
- `counting_iterator` in `nn/config.h` is the C++17 substitute for `std::views::iota`, used throughout for parallel algorithm index ranges

**Module layout**:
- `matrix.h` — core Matrix class, all arithmetic and BLAS-like ops
- `layer.h` — Layer base + all layer implementations (Linear, activations, BatchNorm, Dropout, Conv2D/MaxPool2D)
- `loss.h` — MSELoss, CrossEntropyLoss
- `optimizer.h` — SGD, SGD_w_Momentum, Adam, AdamW (decoupled weight decay)
- `lr_scheduler.h` — StepLR, CosineAnnealingLR, ExponentialLR
- `dataloader.h` — Dataset/TensorDataset/DataLoader
- `model/model.h` — Model container with train/eval mode
- `model/io.h` — Binary serialization (v1/v2 format, v1 backward-compatible)
- `nn/nn.h` — Aggregate header that includes everything

**Transformer components**: LayerNorm, RMSNorm, Softmax, PositionalEncoding, MultiHeadAttention, FeedForward (GELU) / SwiGLUFeedForward, SwiGLU, GPTBlock, GPTModel, PatchEmbedding.

**Test structure**: Tests use `<cassert>` (not a framework). Each test is a named function in test files (like `test_tensor.cpp`, `test_gpt.cpp`, `test_tokenizer.cpp`, `test_transformer.cpp`), dispatched by name via command-line argument. The CMake config forces `-UNDEBUG` so assertions remain active even in Release builds.

## CMake Options

| Option | Default | Notes |
|--------|---------|-------|
| `BUILD_TESTING` | ON (top-level) | Builds tests (test_tensor, test_gpt, test_tokenizer, test_transformer) |
| `BUILD_EXAMPLES` | ON | Builds examples (mnist, gpt) |
| `ENABLE_ASAN` | OFF | Requires Debug or RelWithDebInfo |
| `ENABLE_UBSAN` | OFF | Requires Debug or RelWithDebInfo |

## Model Serialization

Binary format with magic `0x4E4E4E4E`. v2 extends v1 by writing per-layer state (BatchNorm running stats). v1 files load without error (backward compatible). Models with BatchNorm require `model.eval()` before inference to use running stats instead of batch stats.
