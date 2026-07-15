#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <execution>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <neuralnet/nn/config.h>

namespace nn
{
    class Matrix
    {
    private:
        std::vector<double> data_{};
        std::size_t rows_{0};
        std::size_t cols_{0};

        [[nodiscard]] constexpr std::size_t index(std::size_t row, std::size_t col) const noexcept
        {
            return row * cols_ + col;
        }

        static void require_same_shape(const Matrix &lhs, const Matrix &rhs, const char *message)
        {
            if (lhs.rows_ != rhs.rows_ || lhs.cols_ != rhs.cols_)
            {
                throw std::invalid_argument(message);
            }
        }

    public:
        Matrix() = default;

        explicit Matrix(std::size_t rows, std::size_t cols)
            : data_(rows * cols), rows_(rows), cols_(cols) {}

        Matrix(std::vector<double> data, std::size_t rows, std::size_t cols)
            : data_(std::move(data)), rows_(rows), cols_(cols)
        {
            if (data_.size() != rows_ * cols_)
            {
                throw std::invalid_argument("data size mismatch");
            }
        }
        
        // 从标量值初始化矩阵
        Matrix(std::size_t rows, std::size_t cols, double value)
            : data_(rows * cols, value), rows_(rows), cols_(cols) {}
        Matrix(const Matrix &) = default;
        Matrix(Matrix &&) noexcept = default;
        Matrix &operator=(const Matrix &) = default;
        Matrix &operator=(Matrix &&) noexcept = default;
        ~Matrix() = default;

        // 访问器
        [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }
        [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }
        [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
        [[nodiscard]] bool empty() const noexcept { return data_.empty(); }
        [[nodiscard]] double at(std::size_t row, std::size_t col) const
        {
            if (row >= rows_ || col >= cols_)
            {
                throw std::out_of_range("Matrix index out of range");
            }
            return data_[index(row, col)];
        }
        void set_value(std::size_t row, std::size_t col, double value)
        {
            if (row >= rows_ || col >= cols_)
            {
                throw std::out_of_range("Matrix index out of range");
            }
            data_[index(row, col)] = value;
        }
        [[nodiscard]] double at_unchecked(std::size_t row, std::size_t col) const noexcept { return data_[index(row, col)]; }
        void set_value_unchecked(std::size_t row, std::size_t col, double value) noexcept { data_[index(row, col)] = value; }
        [[nodiscard]] const std::vector<double> &data() const noexcept { return data_; }
        [[nodiscard]] std::vector<double> &data() noexcept { return data_; }
        [[nodiscard]] std::vector<std::vector<double>> get_data() const
        {
            std::vector<std::vector<double>> result(rows_, std::vector<double>(cols_, 0.0));
            for (std::size_t row = 0; row < rows_; ++row)
            {
                for (std::size_t col = 0; col < cols_; ++col)
                {
                    result[row][col] = data_[index(row, col)];
                }
            }
            return result;
        }

        void set_data(const std::vector<std::vector<double>> &new_data)
        {
            if (new_data.empty())
            {
                throw std::invalid_argument("set_data: new_data is empty");
            }

            const std::size_t new_rows = new_data.size();
            const std::size_t new_cols = new_data.front().size();
            for (const auto &row : new_data)
            {
                if (row.size() != new_cols)
                {
                    throw std::invalid_argument("set_data: all rows must have the same number of columns");
                }
            }

            if (new_rows != rows_ || new_cols != cols_)
            {
                std::cerr << "warning: Matrix::set_data resizing from "
                          << rows_ << "x" << cols_ << " to "
                          << new_rows << "x" << new_cols << std::endl;
                rows_ = new_rows;
                cols_ = new_cols;
                data_.resize(rows_ * cols_);
            }

            for (std::size_t row = 0; row < rows_; ++row)
            {
                for (std::size_t col = 0; col < cols_; ++col)
                {
                    data_[index(row, col)] = new_data[row][col];
                }
            }
        }

        [[nodiscard]] Matrix transpose() const
        {
            Matrix result(cols_, rows_);

            const std::size_t i_blocks = (rows_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (cols_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(i_blocks * j_blocks),
                          [&](std::size_t block_idx) noexcept
                          {
                              const std::size_t ib = block_idx / j_blocks;
                              const std::size_t jb = block_idx % j_blocks;

                              const std::size_t i0 = ib * BLOCK_SIZE;
                              const std::size_t j0 = jb * BLOCK_SIZE;
                              const std::size_t i1 = std::min(i0 + BLOCK_SIZE, rows_);
                              const std::size_t j1 = std::min(j0 + BLOCK_SIZE, cols_);

                              for (std::size_t i = i0; i < i1; ++i)
                                  for (std::size_t j = j0; j < j1; ++j)
                                      result.data_[j * rows_ + i] = data_[i * cols_ + j];
                          });

            return result;
        }

        [[nodiscard]] Matrix operator+(const Matrix &other) const
        {
            require_same_shape(*this, other, "addition dimension mismatch");
            Matrix result(rows_, cols_);
            std::transform(NN_EXEC_POLICY, data_.begin(), data_.end(), other.data_.begin(),
                           result.data_.begin(), std::plus<>{});
            return result;
        }

        [[nodiscard]] Matrix operator-(const Matrix &other) const
        {
            require_same_shape(*this, other, "subtraction dimension mismatch");
            Matrix result(rows_, cols_);
            std::transform(NN_EXEC_POLICY, data_.begin(), data_.end(), other.data_.begin(),
                           result.data_.begin(), std::minus<>{});
            return result;
        }

        [[nodiscard]] Matrix operator*(double scalar) const
        {
            Matrix result(rows_, cols_);
            std::transform(NN_EXEC_POLICY, data_.begin(), data_.end(), result.data_.begin(),
                           [scalar](double value) noexcept { return value * scalar; });
            return result;
        }

        friend Matrix operator*(double scalar, const Matrix &mat) noexcept
        {
            return mat * scalar;
        }

        [[nodiscard]] Matrix operator*(const Matrix &other) const
        {
            if (cols_ != other.rows_)
            {
                throw std::invalid_argument("matrix multiplication dimension mismatch");
            }

            const std::size_t M = rows_;
            const std::size_t N = other.cols_;
            const std::size_t K = cols_;

            Matrix result(M, N);

            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(i_blocks * j_blocks),
                          [&](std::size_t block_idx)
                          {
                              const std::size_t i_block = block_idx / j_blocks;
                              const std::size_t j_block = block_idx % j_blocks;

                              const std::size_t i_start = i_block * BLOCK_SIZE;
                              const std::size_t i_end = std::min(i_start + BLOCK_SIZE, M);
                              const std::size_t j_start = j_block * BLOCK_SIZE;
                              const std::size_t j_end = std::min(j_start + BLOCK_SIZE, N);

                              for (std::size_t k_start = 0; k_start < K; k_start += BLOCK_SIZE)
                              {
                                  const std::size_t k_end = std::min(k_start + BLOCK_SIZE, K);
                                  const std::size_t k_len = k_end - k_start;
                                  const std::size_t j_len = j_end - j_start;

                                  // 加载 B 的子块到栈数组并转置：b_block[jj * k_len + kk] = B(k_start+kk, j_start+jj)
                                  std::array<double, BLOCK_SIZE * BLOCK_SIZE> b_block{};
                                  for (std::size_t jj = 0; jj < j_len; ++jj)
                                  {
                                      for (std::size_t kk = 0; kk < k_len; ++kk)
                                      {
                                          b_block[jj * k_len + kk] = other.data_[other.index(k_start + kk, j_start + jj)];
                                      }
                                  }

                                  // 累加当前 K 块对 C 块的贡献
                                  for (std::size_t i = i_start; i < i_end; ++i)
                                  {
                                      const std::size_t a_base = i * K + k_start;
                                      for (std::size_t j = j_start; j < j_end; ++j)
                                      {
                                          double sum = 0.0;
                                          const std::size_t b_base = (j - j_start) * k_len;
                                          for (std::size_t kk = 0; kk < k_len; ++kk)
                                              sum += data_[a_base + kk] * b_block[b_base + kk];
                                          result.data_[result.index(i, j)] += sum;
                                      }
                                  }
                              }
                          });

            return result;
        }

        // ── matmul_NT: this * other^T ─────────────────────────────────────────
        // 等价于 (*this) * other.transpose() 但跳过显式 transpose 内存分配。
        // 形状：this = (M, K)，other = (N, K)，result = (M, N)。
        // 访问模式：A 行连续 + B 行连续（B^T 的列就是 B 的行），无需 b_block 转置。
        [[nodiscard]] Matrix matmul_NT(const Matrix &other) const
        {
            if (cols_ != other.cols_)
            {
                throw std::invalid_argument("matmul_NT dimension mismatch (cols)");
            }

            const std::size_t M = rows_;
            const std::size_t N = other.rows_;
            const std::size_t K = cols_;

            Matrix result(M, N);

            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(i_blocks * j_blocks),
                          [&](std::size_t block_idx)
                          {
                              const std::size_t i_block = block_idx / j_blocks;
                              const std::size_t j_block = block_idx % j_blocks;

                              const std::size_t i_start = i_block * BLOCK_SIZE;
                              const std::size_t i_end = std::min(i_start + BLOCK_SIZE, M);
                              const std::size_t j_start = j_block * BLOCK_SIZE;
                              const std::size_t j_end = std::min(j_start + BLOCK_SIZE, N);

                              for (std::size_t k_start = 0; k_start < K; k_start += BLOCK_SIZE)
                              {
                                  const std::size_t k_end = std::min(k_start + BLOCK_SIZE, K);
                                  const std::size_t k_len = k_end - k_start;

                                  // A[i, k_start..k_end) 和 B[j, k_start..k_end) 都是行连续，
                                  // 直接做内积，免去 b_block 转置 staging。
                                  for (std::size_t i = i_start; i < i_end; ++i)
                                  {
                                      const std::size_t a_base = i * K + k_start;
                                      for (std::size_t j = j_start; j < j_end; ++j)
                                      {
                                          const std::size_t b_base = j * K + k_start;
                                          double sum = 0.0;
                                          for (std::size_t kk = 0; kk < k_len; ++kk)
                                              sum += data_[a_base + kk] * other.data_[b_base + kk];
                                          result.data_[result.index(i, j)] += sum;
                                      }
                                  }
                              }
                          });

            return result;
        }

        // ── matmul_TN: this^T * other ─────────────────────────────────────────
        // 等价于 this.transpose() * other 但跳过显式 transpose 内存分配。
        // 形状：this = (K, M)，other = (K, N)，result = (M, N)。
        // 访问模式：A 列连续（this^T 的行）+ B 列连续。把 A 的 (K_block × M_block)
        // 子块和 B 的 (K_block × N_block) 子块加载到栈数组，在内核里按 k 累加。
        [[nodiscard]] Matrix matmul_TN(const Matrix &other) const
        {
            if (rows_ != other.rows_)
            {
                throw std::invalid_argument("matmul_TN dimension mismatch (rows)");
            }

            const std::size_t M = cols_;       // this^T 的行数
            const std::size_t N = other.cols_;
            const std::size_t K = rows_;       // 共同维度

            Matrix result(M, N);

            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(i_blocks * j_blocks),
                          [&](std::size_t block_idx)
                          {
                              const std::size_t i_block = block_idx / j_blocks;
                              const std::size_t j_block = block_idx % j_blocks;

                              const std::size_t i_start = i_block * BLOCK_SIZE;
                              const std::size_t i_end = std::min(i_start + BLOCK_SIZE, M);
                              const std::size_t j_start = j_block * BLOCK_SIZE;
                              const std::size_t j_end = std::min(j_start + BLOCK_SIZE, N);
                              const std::size_t i_len = i_end - i_start;

                              for (std::size_t k_start = 0; k_start < K; k_start += BLOCK_SIZE)
                              {
                                  const std::size_t k_end = std::min(k_start + BLOCK_SIZE, K);
                                  const std::size_t k_len = k_end - k_start;

                                  // 加载 A 子块并转置到栈数组：
                                  // a_block[ii * k_len + kk] = A(k_start+kk, i_start+ii) = this(k, i)
                                  // 这样内层 kk 循环对 a_block 行连续。
                                  std::array<double, BLOCK_SIZE * BLOCK_SIZE> a_block{};
                                  for (std::size_t ii = 0; ii < i_len; ++ii)
                                  {
                                      for (std::size_t kk = 0; kk < k_len; ++kk)
                                      {
                                          a_block[ii * k_len + kk] =
                                              data_[(k_start + kk) * cols_ + (i_start + ii)];
                                      }
                                  }

                                  // 累加：C(i,j) += Σ a_block[ii,kk] * other(k_start+kk, j)
                                  for (std::size_t ii = 0; ii < i_len; ++ii)
                                  {
                                      const std::size_t a_base = ii * k_len;
                                      const std::size_t i = i_start + ii;
                                      for (std::size_t j = j_start; j < j_end; ++j)
                                      {
                                          double sum = 0.0;
                                          for (std::size_t kk = 0; kk < k_len; ++kk)
                                              sum += a_block[a_base + kk] *
                                                     other.data_[(k_start + kk) * other.cols_ + j];
                                          result.data_[result.index(i, j)] += sum;
                                      }
                                  }
                              }
                          });

            return result;
        }

        void scale_inplace(double scalar) noexcept
        {
            std::for_each(NN_EXEC_POLICY, data_.begin(), data_.end(),
                          [scalar](double &value) noexcept { value *= scalar; });
        }

        // 逐元素加法 inplace
        void add_inplace(const Matrix &other) noexcept
        {
            if (rows_ != other.rows_ || cols_ != other.cols_) return;
            std::transform(NN_EXEC_POLICY, data_.begin(), data_.end(), other.data_.begin(),
                           data_.begin(), std::plus<>{});
        }

        // 逐元素减法 inplace
        void subtract_inplace(const Matrix &other) noexcept
        {
            if (rows_ != other.rows_ || cols_ != other.cols_) return;
            std::transform(NN_EXEC_POLICY, data_.begin(), data_.end(), other.data_.begin(),
                           data_.begin(), std::minus<>{});
        }

        // 填充零
        void zero() noexcept
        {
            std::fill(data_.begin(), data_.end(), 0.0);
        }

        void resize(std::size_t rows, std::size_t cols)
        {
            if (rows_ == rows && cols_ == cols) return;
            if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows)
                throw std::overflow_error("Matrix::resize: rows * cols overflow");
            data_.resize(rows * cols);
            rows_ = rows;
            cols_ = cols;
        }

        [[nodiscard]] Matrix row_slice(std::size_t start_row, std::size_t num_rows) const
        {
            if (start_row > rows_ || num_rows > rows_ - start_row)
                throw std::out_of_range("row_slice: slice exceeds matrix bounds");
            Matrix result(num_rows, cols_);
            for (std::size_t i = 0; i < num_rows; ++i)
                std::copy_n(data_.data() + (start_row + i) * cols_, cols_,
                            result.data_.data() + i * cols_);
            return result;
        }

        void set_row_slice(std::size_t start_row, const Matrix &slice)
        {
            if (start_row > rows_ || slice.rows_ > rows_ - start_row)
                throw std::out_of_range("set_row_slice: slice exceeds matrix bounds");
            if (slice.cols_ != cols_)
                throw std::invalid_argument("set_row_slice: column count mismatch");
            for (std::size_t i = 0; i < slice.rows_; ++i)
                std::copy_n(slice.data_.data() + i * slice.cols_, cols_,
                            data_.data() + (start_row + i) * cols_);
        }

        // ── Reduction 规约操作 (F8) ──────────────────────────────────────────────
        // 全部基于 NN_EXEC_POLICY 并行；空矩阵返回值由各方法文档说明。
        // 存储布局：行主序（data_[row * cols_ + col]）。

        // Frobenius 范数：sqrt(sum(x^2))。空矩阵返回 0。
        [[nodiscard]] double norm() const noexcept
        {
            const double sumsq = std::transform_reduce(
                NN_EXEC_POLICY, data_.begin(), data_.end(),
                0.0, std::plus{},
                [](double x) noexcept { return x * x; });
            return std::sqrt(sumsq);
        }

        // 全元素求和。空矩阵返回 0。
        [[nodiscard]] double sum() const noexcept
        {
            return std::reduce(NN_EXEC_POLICY, data_.begin(), data_.end(),
                                0.0, std::plus{});
        }

        // 全元素均值。空矩阵返回 0（避免 0/0）。
        [[nodiscard]] double mean() const noexcept
        {
            if (data_.empty()) return 0.0;
            return sum() / static_cast<double>(data_.size());
        }

        // 每列求和。行主序下，列 j 的元素位于 strided 位置（j, j+cols_, j+2*cols_, ...）。
        // 返回长度 cols_ 的向量。
        [[nodiscard]] std::vector<double> colwise_sum() const
        {
            std::vector<double> result(cols_, 0.0);
            if (cols_ == 0) return result;
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(cols_),
                          [&](std::size_t j)
                          {
                              double s = 0.0;
                              for (std::size_t i = 0; i < rows_; ++i)
                                  s += data_[i * cols_ + j];
                              result[j] = s;
                          });
            return result;
        }

        // 每行求和。返回长度 rows_ 的向量。
        // 用于 Linear::backward 的 grad_b（按 batch 维聚合）以及 BatchNorm 的
        // per-feature 统计校验。
        [[nodiscard]] std::vector<double> rowwise_sum() const
        {
            std::vector<double> result(rows_, 0.0);
            if (rows_ == 0) return result;
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(rows_),
                          [&](std::size_t i)
                          {
                              const double *row_ptr = data_.data() + i * cols_;
                              double s = 0.0;
                              for (std::size_t j = 0; j < cols_; ++j)
                                  s += row_ptr[j];
                              result[i] = s;
                          });
            return result;
        }

        // 每行均值：rowwise_sum() / cols_。cols_ == 0 时返回全 0 向量。
        [[nodiscard]] std::vector<double> rowwise_mean() const
        {
            std::vector<double> result = rowwise_sum();
            if (cols_ == 0) return result;
            const double denom = static_cast<double>(cols_);
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(rows_),
                          [&](std::size_t i) noexcept { result[i] /= denom; });
            return result;
        }

        // 每列均值：colwise_sum() / rows_。rows_ == 0 时返回全 0 向量。
        [[nodiscard]] std::vector<double> colwise_mean() const
        {
            std::vector<double> result = colwise_sum();
            if (rows_ == 0) return result;
            const double denom = static_cast<double>(rows_);
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(cols_),
                          [&](std::size_t j) noexcept { result[j] /= denom; });
            return result;
        }

        // 每列方差（无偏估计，分母 rows_-1）。给 BatchNorm 预留。
        // rows_ <= 1 时无法估计，返回全 0 向量。
        [[nodiscard]] std::vector<double> colwise_var(const std::vector<double> &mean) const
        {
            std::vector<double> result(cols_, 0.0);
            if (cols_ == 0) return result;
            if (rows_ <= 1) return result;
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(cols_),
                          [&](std::size_t j)
                          {
                              const double m = mean[j];
                              double s = 0.0;
                              for (std::size_t i = 0; i < rows_; ++i)
                              {
                                  const double d = data_[i * cols_ + j] - m;
                                  s += d * d;
                              }
                              result[j] = s / static_cast<double>(rows_ - 1);
                          });
            return result;
        }
    };
} // namespace nn

#endif // MATRIX_HPP
