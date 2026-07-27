#ifndef LAYER_HPP
#define LAYER_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <execution>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include <neuralnet/matrix.h>

#include <neuralnet/nn/config.h>

namespace nn
{
    class Layer
    {
    public:
        virtual ~Layer() = default;
        virtual Matrix forward(const Matrix &input) = 0;
        virtual Matrix backward(const Matrix &grad_output) = 0;
        virtual std::vector<std::reference_wrapper<Matrix>> parameters() { return {}; }
        virtual std::vector<std::reference_wrapper<Matrix>> param_gradients() { return {}; }
        
        // 添加参数更新辅助方法，避免虚函数调用开销
        virtual void update_params(double /*lr*/) noexcept {}
        virtual void zero_grad() noexcept {}

        // ── 名称 (F7) ──────────────────────────────────────────────────────────
        // 用于日志/调试/汇总。默认 "Layer"；各具体层覆盖为各自名字。
        virtual const char *name() const { return "Layer"; }

        // ── 模式切换 (F6) ──────────────────────────────────────────────────────
        // 由 Model::train()/eval() 调用以向下传播模式。默认空操作；
        // 含状态/概率行为的层（如 Dropout、BatchNorm）可覆盖以同步内部状态。
        virtual void on_mode_change(bool /*training*/) {}

        // ── 持久化 (F5) ────────────────────────────────────────────────────────
        // 写入/读取非 Matrix 状态（运行均值/方差、batch 计数等）。
        // 默认空操作：基类与无状态层（如 ReLU）不写任何东西。
        // 配合 has_state() 让 Model 知道该层是否参与 v2 状态段。
        virtual void save_state(std::ostream & /*os*/) const {}
        virtual void load_state(std::istream & /*is*/) {}
        virtual bool has_state() const { return false; }

        // ── 参数计数 (F4) ──────────────────────────────────────────────────────
        // 返回本层可训练参数的元素总数（不含 running stats 等非可训练状态）。
        // 用于 Model::summary() 的 "Params" 列。默认 0；各具体层覆盖。
        [[nodiscard]] virtual std::size_t param_count() const noexcept { return 0; }
    };

    class Linear final : public Layer
    {
    private:
        Matrix W_;
        Matrix b_;
        Matrix grad_W_;
        Matrix grad_b_;
        Matrix input_cache_;

        // 修复：使用 thread_local 保证多线程构造 Layer 时的线程安全
        inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

    public:
        Linear(std::size_t in_features, std::size_t out_features)
            : W_(out_features, in_features),
              b_(out_features, 1),
              grad_W_(out_features, in_features),
              grad_b_(out_features, 1),
              input_cache_()
        {
            // Xavier 均匀初始化：适合 tanh/sigmoid，对 ReLU 也可用
            const double limit = std::sqrt(6.0 / static_cast<double>(in_features + out_features));
            std::uniform_real_distribution<double> dist(-limit, limit);
            std::generate(W_.data().begin(), W_.data().end(),
                          [&] { return dist(rng_); });
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            return {std::ref(W_), std::ref(b_)};
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            return {std::ref(grad_W_), std::ref(grad_b_)};
        }

        [[nodiscard]] std::size_t param_count() const noexcept override
        {
            return W_.size() + b_.size();
        }

        const char *name() const override { return "Linear"; }

        Matrix forward(const Matrix &input) override
        {
            if (input.rows() != W_.cols())
            {
                throw std::invalid_argument("linear forward input shape mismatch");
            }

            input_cache_ = input;

            // Single allocation: matmul result is the output; bias is added in-place.
            Matrix result = W_ * input;

            // Row-strided bias add. Outer loop over rows (parallel); inner loop walks
            // the row contiguously in memory, with `bias_val` hoisted out of the
            // inner loop (one load per row instead of one per element).
            const std::size_t cols = result.cols();
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(result.rows()),
                          [&](std::size_t row) noexcept
                          {
                              const double bias_val = b_.at_unchecked(row, 0);
                              for (std::size_t col = 0; col < cols; ++col)
                              {
                                  result.data()[row * cols + col] += bias_val;
                              }
                          });

            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (grad_output.rows() != W_.rows())
            {
                throw std::invalid_argument("linear backward grad_output shape mismatch");
            }
            if (input_cache_.rows() != W_.cols() || input_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("linear backward cache/input shape mismatch");
            }

            const std::size_t out_feat = W_.rows();

            // grad_input = W^T * grad_output       (in_feat × batch)
            // grad_W     = grad_output * input^T   (out_feat × in_feat) — REPLACE not +=
            // grad_b     = rowwise_sum(grad_output) along batch dim → (out_feat × 1)
            // The `=` on grad_W_ is a replace, not accumulate. The optimizer is
            // responsible for zeroing gradients before each backward pass.
            //
            // 用 matmul_TN / matmul_NT 直接消化转置，避免两次中间矩阵分配
            // （旧实现先 W.transpose() / input_cache_.transpose() 后再常规 matmul）。

            Matrix grad_input = W_.matmul_TN(grad_output);
            grad_W_ = grad_output.matmul_NT(input_cache_);

            std::vector<double> b_vec = grad_output.rowwise_sum();
            grad_b_ = Matrix(out_feat, 1);
            for (std::size_t i = 0; i < out_feat; ++i)
            {
                grad_b_.set_value_unchecked(i, 0, b_vec[i]);
            }

            return grad_input;
        }
    };

    class ReLU final : public Layer
    {
    private:
        Matrix input_cache_;

    public:
        const char *name() const override { return "ReLU"; }

        Matrix forward(const Matrix &input) override
        {
            input_cache_ = input;
            Matrix result(input.rows(), input.cols());
            std::transform(NN_EXEC_POLICY, input.data().begin(), input.data().end(),
                           result.data().begin(), [](double value) noexcept
                           { return value > 0.0 ? value : 0.0; });
            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (input_cache_.rows() != grad_output.rows() || input_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("relu backward shape mismatch");
            }

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            std::transform(NN_EXEC_POLICY,
                           input_cache_.data().begin(), input_cache_.data().end(),
                           grad_output.data().begin(),
                           grad_input.data().begin(),
                           [](double input_value, double grad_value) noexcept
                           {
                               return input_value > 0.0 ? grad_value : 0.0;
                           });
            return grad_input;
        }
    };

    // ── LeakyReLU ─────────────────────────────────────────────────────────────
    class LeakyReLU final : public Layer
    {
    private:
        double negative_slope_;
        Matrix input_cache_;

    public:
        explicit LeakyReLU(double negative_slope = 0.01)
            : negative_slope_(negative_slope) {}

        const char *name() const override { return "LeakyReLU"; }

        Matrix forward(const Matrix &input) override
        {
            input_cache_ = input;
            Matrix result(input.rows(), input.cols());
            std::transform(NN_EXEC_POLICY, input.data().begin(), input.data().end(),
                           result.data().begin(),
                           [this](double value) noexcept
                           { return value > 0.0 ? value : negative_slope_ * value; });
            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (input_cache_.rows() != grad_output.rows() || input_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("leaky_relu backward shape mismatch");
            }

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            std::transform(NN_EXEC_POLICY,
                           input_cache_.data().begin(), input_cache_.data().end(),
                           grad_output.data().begin(),
                           grad_input.data().begin(),
                           [this](double input_value, double grad_value) noexcept
                           {
                               return input_value > 0.0 ? grad_value : negative_slope_ * grad_value;
                           });
            return grad_input;
        }
    };

    // ── Sigmoid ───────────────────────────────────────────────────────────────
    class Sigmoid final : public Layer
    {
    private:
        Matrix output_cache_;

    public:
        const char *name() const override { return "Sigmoid"; }

        Matrix forward(const Matrix &input) override
        {
            Matrix result(input.rows(), input.cols());
            std::transform(NN_EXEC_POLICY, input.data().begin(), input.data().end(),
                           result.data().begin(),
                           [](double value) noexcept
                           { return 1.0 / (1.0 + std::exp(-value)); });
            output_cache_ = result;
            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (output_cache_.rows() != grad_output.rows() || output_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("sigmoid backward shape mismatch");
            }

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            std::transform(NN_EXEC_POLICY,
                           output_cache_.data().begin(), output_cache_.data().end(),
                           grad_output.data().begin(),
                           grad_input.data().begin(),
                           [](double s, double grad) noexcept
                           { return grad * s * (1.0 - s); });
            return grad_input;
        }
    };

    // ── Tanh ──────────────────────────────────────────────────────────────────
    class Tanh final : public Layer
    {
    private:
        Matrix output_cache_;

    public:
        const char *name() const override { return "Tanh"; }

        Matrix forward(const Matrix &input) override
        {
            Matrix result(input.rows(), input.cols());
            std::transform(NN_EXEC_POLICY, input.data().begin(), input.data().end(),
                           result.data().begin(),
                           [](double value) noexcept
                           { return std::tanh(value); });
            output_cache_ = result;
            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (output_cache_.rows() != grad_output.rows() || output_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("tanh backward shape mismatch");
            }

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            std::transform(NN_EXEC_POLICY,
                           output_cache_.data().begin(), output_cache_.data().end(),
                           grad_output.data().begin(),
                           grad_input.data().begin(),
                           [](double t, double grad) noexcept
                           { return grad * (1.0 - t * t); });
            return grad_input;
        }
    };

    // ── GELU ──────────────────────────────────────────────────────────────────
    // 使用近似公式：GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    class GELU final : public Layer
    {
    private:
        Matrix input_cache_;

        static double gelu_fn(double x) noexcept
        {
            constexpr double sqrt_2_over_pi = 0.7978845608028654; // std::sqrt(2.0 / M_PI)
            return 0.5 * x * (1.0 + std::tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)));
        }

        static double gelu_grad_fn(double x) noexcept
        {
            constexpr double sqrt_2_over_pi = 0.7978845608028654;
            const double inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            const double tanh_inner = std::tanh(inner);
            const double sech2 = 1.0 - tanh_inner * tanh_inner;
            const double d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
            return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner;
        }

    public:
        const char *name() const override { return "GELU"; }

        Matrix forward(const Matrix &input) override
        {
            input_cache_ = input;
            Matrix result(input.rows(), input.cols());
            std::transform(NN_EXEC_POLICY, input.data().begin(), input.data().end(),
                           result.data().begin(), gelu_fn);
            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (input_cache_.rows() != grad_output.rows() || input_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("gelu backward shape mismatch");
            }

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            std::transform(NN_EXEC_POLICY,
                           input_cache_.data().begin(), input_cache_.data().end(),
                           grad_output.data().begin(),
                           grad_input.data().begin(),
                           [](double x, double grad) noexcept
                           { return grad * gelu_grad_fn(x); });
            return grad_input;
        }
    };

    // ── Dropout ───────────────────────────────────────────────────────────────
    // 训练时以概率 p 随机置零并缩放，推理时直接传递
    class Dropout final : public Layer
    {
    private:
        double p_;
        bool training_;
        Matrix mask_; // 0.0 或 1.0 / (1 - p_)
        inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

    public:
        explicit Dropout(double p = 0.5, bool training = true)
            : p_(p), training_(training)
        {
            if (p_ < 0.0 || p_ >= 1.0)
            {
                throw std::invalid_argument("dropout probability must be in [0, 1)");
            }
        }

        const char *name() const override { return "Dropout"; }

        void on_mode_change(bool training) noexcept override { training_ = training; }

        Matrix forward(const Matrix &input) override
        {
            if (!training_ || p_ == 0.0)
            {
                return input;
            }

            const double scale = 1.0 / (1.0 - p_);
            std::bernoulli_distribution dist(1.0 - p_);
            mask_ = Matrix(input.rows(), input.cols());

            std::transform(NN_EXEC_POLICY, input.data().begin(), input.data().end(),
                           mask_.data().begin(),
                           [&](double /*value*/) noexcept -> double
                           { return dist(rng_) ? scale : 0.0; });

            Matrix result(input.rows(), input.cols());
            std::transform(NN_EXEC_POLICY,
                           input.data().begin(), input.data().end(),
                           mask_.data().begin(),
                           result.data().begin(),
                           [](double val, double m) noexcept
                           { return val * m; });
            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (!training_ || p_ == 0.0)
            {
                return grad_output;
            }

            if (mask_.rows() != grad_output.rows() || mask_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("dropout backward shape mismatch");
            }

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            std::transform(NN_EXEC_POLICY,
                           grad_output.data().begin(), grad_output.data().end(),
                           mask_.data().begin(),
                           grad_input.data().begin(),
                           [](double grad, double m) noexcept
                           { return grad * m; });
            return grad_input;
        }
    };

    // ── BatchNorm1d ──────────────────────────────────────────────────────────
    // 批归一化 1D：输入 (num_features, batch_size)；按特征维（行）统计。
    // 训练：用 batch mean/var 归一化 + 动量更新 running stats。
    // 推理：用 running mean/var 归一化。
    // 可选 affine：归一化后再做 gamma*x_hat + beta 仿射。
    // BN2d 复用本类：把 (C, N*H*W) 视为 (C, batch)，forward 完全相同。
    class BatchNorm1d : public Layer
    {
    private:
        std::size_t num_features_;
        double eps_;
        double momentum_;
        bool affine_;

        Matrix gamma_;        // (num_features, 1)
        Matrix beta_;         // (num_features, 1)
        Matrix dgamma_;       // (num_features, 1)
        Matrix dbeta_;        // (num_features, 1)
        Matrix running_mean_; // (num_features, 1), init=0
        Matrix running_var_;  // (num_features, 1), init=1
        int64_t num_batches_tracked_{0};
        bool is_training_{true};

        Matrix input_cache_;
        Matrix batch_mean_;   // (num_features, 1)
        Matrix batch_var_;    // (num_features, 1)
        Matrix normalized_;   // (num_features, batch_size)

        // 行主序下，按行求和 = 按特征聚合（每个 i 一行 = 一个特征的 batch 内值）
        static std::vector<double> rowwise_sum(const Matrix &m)
        {
            std::vector<double> result(m.rows(), 0.0);
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(m.rows()),
                          [&](std::size_t i) noexcept
                          {
                              double s = 0.0;
                              for (std::size_t j = 0; j < m.cols(); ++j)
                                  s += m.at_unchecked(i, j);
                              result[i] = s;
                          });
            return result;
        }

        // 行主序下，按行求均值
        static std::vector<double> rowwise_mean(const Matrix &m)
        {
            std::vector<double> result = rowwise_sum(m);
            if (m.cols() == 0) return result;
            const double denom = static_cast<double>(m.cols());
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(m.rows()),
                          [&](std::size_t i) noexcept { result[i] /= denom; });
            return result;
        }

        // 行主序下，按行求总体方差（分母 = cols_，PyTorch BatchNorm 约定）
        static std::vector<double> rowwise_var(const Matrix &m, const std::vector<double> &mean)
        {
            std::vector<double> result(m.rows(), 0.0);
            if (m.cols() == 0) return result;
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(m.rows()),
                          [&](std::size_t i)
                          {
                              const double m_i = mean[i];
                              double s = 0.0;
                              for (std::size_t j = 0; j < m.cols(); ++j)
                              {
                                  const double d = m.at_unchecked(i, j) - m_i;
                                  s += d * d;
                              }
                              result[i] = s / static_cast<double>(m.cols());
                          });
            return result;
        }

    public:
        BatchNorm1d(std::size_t num_features, double eps = 1e-5,
                    double momentum = 0.1, bool affine = true)
            : num_features_(num_features), eps_(eps), momentum_(momentum), affine_(affine),
              gamma_(affine ? num_features : 0, 1, affine ? 1.0 : 0.0),
              beta_(affine ? num_features : 0, 1, 0.0),
              dgamma_(num_features, 1, 0.0),
              dbeta_(num_features, 1, 0.0),
              running_mean_(num_features, 1, 0.0),
              running_var_(num_features, 1, 1.0),
              batch_mean_(num_features, 1, 0.0),
              batch_var_(num_features, 1, 0.0)
        {
            if (num_features_ == 0)
            {
                throw std::invalid_argument("BatchNorm1d: num_features must be > 0");
            }
        }

        // ── 公开访问器（测试 / 调试） ──
        [[nodiscard]] const Matrix &running_mean() const noexcept { return running_mean_; }
        [[nodiscard]] const Matrix &running_var() const noexcept { return running_var_; }
        [[nodiscard]] int64_t num_batches_tracked() const noexcept { return num_batches_tracked_; }
        [[nodiscard]] bool is_training() const noexcept { return is_training_; }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            if (!affine_) return {};
            return {std::ref(gamma_), std::ref(beta_)};
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            if (!affine_) return {};
            return {std::ref(dgamma_), std::ref(dbeta_)};
        }

        void zero_grad() noexcept override
        {
            if (!affine_) return;
            dgamma_.zero();
            dbeta_.zero();
        }

        [[nodiscard]] std::size_t param_count() const noexcept override
        {
            return affine_ ? 2 * num_features_ : 0;
        }

        void on_mode_change(bool training) noexcept override { is_training_ = training; }

        bool has_state() const override { return true; }

        const char *name() const override { return "BatchNorm1d"; }

        Matrix forward(const Matrix &input) override
        {
            if (input.rows() != num_features_)
            {
                throw std::invalid_argument("BatchNorm1d forward: input rows != num_features");
            }

            const std::size_t batch_size = input.cols();
            input_cache_ = input;

            if (batch_size == 0)
            {
                normalized_ = Matrix(num_features_, 0);
                return Matrix(num_features_, 0);
            }

            if (is_training_)
            {
                const std::vector<double> mu_vec = rowwise_mean(input);
                const std::vector<double> var_vec = rowwise_var(input, mu_vec);

                for (std::size_t i = 0; i < num_features_; ++i)
                {
                    batch_mean_.set_value_unchecked(i, 0, mu_vec[i]);
                    batch_var_.set_value_unchecked(i, 0, var_vec[i]);
                }

                // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
                std::transform(NN_EXEC_POLICY,
                               running_mean_.data().begin(), running_mean_.data().end(),
                               batch_mean_.data().begin(),
                               running_mean_.data().begin(),
                               [this](double rm, double bm) noexcept
                               { return (1.0 - momentum_) * rm + momentum_ * bm; });
                std::transform(NN_EXEC_POLICY,
                               running_var_.data().begin(), running_var_.data().end(),
                               batch_var_.data().begin(),
                               running_var_.data().begin(),
                               [this](double rv, double bv) noexcept
                               { return (1.0 - momentum_) * rv + momentum_ * bv; });
                ++num_batches_tracked_;
            }

            // normalized_ = (input - mean) / sqrt(var + eps)
            const Matrix &mean_src = is_training_ ? batch_mean_ : running_mean_;
            const Matrix &var_src = is_training_ ? batch_var_ : running_var_;
            normalized_ = Matrix(num_features_, batch_size);
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(input.size()),
                          [&](std::size_t idx)
                          {
                              const std::size_t i = idx / batch_size;
                              const double mu = mean_src.at_unchecked(i, 0);
                              const double std_inv = 1.0 / std::sqrt(var_src.at_unchecked(i, 0) + eps_);
                              normalized_.data()[idx] = (input.data()[idx] - mu) * std_inv;
                          });

            if (!affine_)
            {
                return normalized_;
            }

            Matrix output(num_features_, batch_size);
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(input.size()),
                          [&](std::size_t idx)
                          {
                              const std::size_t i = idx / batch_size;
                              output.data()[idx] = normalized_.data()[idx] * gamma_.at_unchecked(i, 0)
                                                  + beta_.at_unchecked(i, 0);
                          });
            return output;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (grad_output.rows() != num_features_)
            {
                throw std::invalid_argument("BatchNorm1d backward: grad_output rows != num_features");
            }
            if (input_cache_.rows() != num_features_ || input_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("BatchNorm1d backward: cache shape mismatch");
            }

            const std::size_t batch_size = grad_output.cols();
            if (batch_size == 0)
            {
                return Matrix(num_features_, 0);
            }

            if (affine_)
            {
                // dgamma: 行主序下对每行求和 = 每特征 (grad_output * normalized) 的 batch 内和
                for (std::size_t i = 0; i < num_features_; ++i)
                {
                    double s = 0.0;
                    for (std::size_t j = 0; j < batch_size; ++j)
                        s += grad_output.at_unchecked(i, j) * normalized_.at_unchecked(i, j);
                    dgamma_.set_value_unchecked(i, 0, s);
                }
                // dbeta: 每特征 grad_output 的 batch 内和
                for (std::size_t i = 0; i < num_features_; ++i)
                {
                    double s = 0.0;
                    for (std::size_t j = 0; j < batch_size; ++j)
                        s += grad_output.at_unchecked(i, j);
                    dbeta_.set_value_unchecked(i, 0, s);
                }
            }

            // dx_hat = grad_output * gamma (broadcast)
            Matrix dx_hat(num_features_, batch_size);
            if (affine_)
            {
                std::for_each(NN_EXEC_POLICY,
                              counting_iterator<std::size_t>(0),
                              counting_iterator<std::size_t>(grad_output.size()),
                              [&](std::size_t idx)
                              {
                                  const std::size_t i = idx / batch_size;
                                  dx_hat.data()[idx] = grad_output.data()[idx]
                                                      * gamma_.at_unchecked(i, 0);
                              });
            }
            else
            {
                dx_hat = grad_output;
            }

            // 各项按行（每特征）的聚合
            std::vector<double> sum_dxhat(num_features_, 0.0);
            std::vector<double> sum_dxhat_xhat(num_features_, 0.0);
            std::vector<double> std_inv(num_features_, 0.0);
            for (std::size_t i = 0; i < num_features_; ++i)
            {
                double sd = 0.0;
                double sdx = 0.0;
                for (std::size_t j = 0; j < batch_size; ++j)
                {
                    sd += dx_hat.at_unchecked(i, j);
                    sdx += dx_hat.at_unchecked(i, j) * normalized_.at_unchecked(i, j);
                }
                sum_dxhat[i] = sd;
                sum_dxhat_xhat[i] = sdx;
                std_inv[i] = 1.0 / std::sqrt(batch_var_.at_unchecked(i, 0) + eps_);
            }

            // dx = (1/N) * (N*dx_hat - sum_dxhat - normalized_*sum_dxhat_xhat) * std_inv
            const double scale = 1.0 / static_cast<double>(batch_size);
            const double N = static_cast<double>(batch_size);
            Matrix dx(num_features_, batch_size);
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(dx.size()),
                          [&](std::size_t idx)
                          {
                              const std::size_t i = idx / batch_size;
                              dx.data()[idx] = scale
                                  * (N * dx_hat.data()[idx] - sum_dxhat[i]
                                     - normalized_.data()[idx] * sum_dxhat_xhat[i])
                                  * std_inv[i];
                          });

            return dx;
        }

        void save_state(std::ostream &os) const override
        {
            auto write_mat = [&](const Matrix &m)
            {
                const std::size_t rows = m.rows();
                const std::size_t cols = m.cols();
                os.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
                os.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
                os.write(reinterpret_cast<const char *>(m.data().data()),
                         m.size() * sizeof(double));
            };
            write_mat(running_mean_);
            write_mat(running_var_);
            os.write(reinterpret_cast<const char *>(&num_batches_tracked_),
                     sizeof(num_batches_tracked_));
        }

        void load_state(std::istream &is) override
        {
            auto read_mat = [&](Matrix &m)
            {
                std::size_t rows = 0, cols = 0;
                is.read(reinterpret_cast<char *>(&rows), sizeof(rows));
                is.read(reinterpret_cast<char *>(&cols), sizeof(cols));
                if (rows != m.rows() || cols != m.cols())
                {
                    throw std::runtime_error("BatchNorm1d: state matrix shape mismatch");
                }
                is.read(reinterpret_cast<char *>(m.data().data()),
                        m.size() * sizeof(double));
            };
            read_mat(running_mean_);
            read_mat(running_var_);
            is.read(reinterpret_cast<char *>(&num_batches_tracked_),
                    sizeof(num_batches_tracked_));
        }
    };

    // ── BatchNorm2d ──────────────────────────────────────────────────────────
    // 把 (C, N*H*W) 视为 (C, batch) —— BN1d::forward 已接受该形状。
    // 因此无需 override forward/backward；只重写 name() 用于日志/汇总。
    class BatchNorm2d : public BatchNorm1d
    {
    public:
        using BatchNorm1d::BatchNorm1d;
        const char *name() const override { return "BatchNorm2d"; }
    };

    // ── LayerNorm ────────────────────────────────────────────────────────────
    class LayerNorm final : public Layer
    {
    private:
        std::size_t normalized_shape_;
        double eps_;
        Matrix gamma_;
        Matrix beta_;
        Matrix dgamma_;
        Matrix dbeta_;
        Matrix normalized_cache_;
        std::vector<double> mean_cache_;
        std::vector<double> inv_std_cache_;

    public:
        explicit LayerNorm(std::size_t normalized_shape, double eps = 1e-5)
            : normalized_shape_(normalized_shape), eps_(eps),
              gamma_(normalized_shape, 1, 1.0),
              beta_(normalized_shape, 1, 0.0),
              dgamma_(normalized_shape, 1, 0.0),
              dbeta_(normalized_shape, 1, 0.0) {}

        const char *name() const override { return "LayerNorm"; }
        [[nodiscard]] std::size_t param_count() const noexcept override { return 2 * normalized_shape_; }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            return {std::ref(gamma_), std::ref(beta_)};
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            return {std::ref(dgamma_), std::ref(dbeta_)};
        }

        Matrix forward(const Matrix &input) override
        {
            if (input.rows() != normalized_shape_)
                throw std::invalid_argument("LayerNorm forward: input rows != normalized_shape");

            const std::size_t feat = input.rows();
            const std::size_t batch = input.cols();
            const double N = static_cast<double>(feat);

            mean_cache_.resize(batch);
            inv_std_cache_.resize(batch);
            normalized_cache_ = Matrix(feat, batch);

            for (std::size_t j = 0; j < batch; ++j)
            {
                double sum = 0.0;
                for (std::size_t i = 0; i < feat; ++i)
                    sum += input.at_unchecked(i, j);
                double mu = sum / N;
                mean_cache_[j] = mu;

                double var_sum = 0.0;
                for (std::size_t i = 0; i < feat; ++i)
                {
                    double d = input.at_unchecked(i, j) - mu;
                    var_sum += d * d;
                }
                double inv_std = 1.0 / std::sqrt(var_sum / N + eps_);
                inv_std_cache_[j] = inv_std;

                for (std::size_t i = 0; i < feat; ++i)
                    normalized_cache_.set_value_unchecked(i, j,
                        (input.at_unchecked(i, j) - mu) * inv_std);
            }

            Matrix output(feat, batch);
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(feat * batch),
                          [&](std::size_t idx) {
                              std::size_t i = idx / batch;
                              output.data()[idx] = normalized_cache_.data()[idx]
                                  * gamma_.at_unchecked(i, 0) + beta_.at_unchecked(i, 0);
                          });
            return output;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (grad_output.rows() != normalized_shape_)
                throw std::invalid_argument("LayerNorm backward: grad_output rows != normalized_shape");
            if (normalized_cache_.rows() != normalized_shape_
                || normalized_cache_.cols() != grad_output.cols())
                throw std::invalid_argument("LayerNorm backward: forward cache shape mismatch");

            const std::size_t feat = grad_output.rows();
            const std::size_t batch = grad_output.cols();
            const double N = static_cast<double>(feat);

            dgamma_.zero();
            dbeta_.zero();
            for (std::size_t i = 0; i < feat; ++i)
            {
                double dg = 0.0, db = 0.0;
                for (std::size_t j = 0; j < batch; ++j)
                {
                    dg += grad_output.at_unchecked(i, j)
                          * normalized_cache_.at_unchecked(i, j);
                    db += grad_output.at_unchecked(i, j);
                }
                dgamma_.set_value_unchecked(i, 0, dg);
                dbeta_.set_value_unchecked(i, 0, db);
            }

            Matrix grad_input(feat, batch);
            for (std::size_t j = 0; j < batch; ++j)
            {
                double inv_std = inv_std_cache_[j];
                double sum_dxhat = 0.0, sum_dxhat_xhat = 0.0;
                for (std::size_t i = 0; i < feat; ++i)
                {
                    double dxhat = grad_output.at_unchecked(i, j) * gamma_.at_unchecked(i, 0);
                    sum_dxhat += dxhat;
                    sum_dxhat_xhat += dxhat * normalized_cache_.at_unchecked(i, j);
                }
                for (std::size_t i = 0; i < feat; ++i)
                {
                    double dxhat = grad_output.at_unchecked(i, j) * gamma_.at_unchecked(i, 0);
                    grad_input.set_value_unchecked(i, j,
                        inv_std / N * (N * dxhat - sum_dxhat
                            - normalized_cache_.at_unchecked(i, j) * sum_dxhat_xhat));
                }
            }
            return grad_input;
        }
    };


    // ── Softmax ──────────────────────────────────────────────────────────────
    class Softmax final : public Layer
    {
    private:
        Matrix output_cache_;

    public:
        const char *name() const override { return "Softmax"; }

        Matrix forward(const Matrix &input) override
        {
            const std::size_t rows = input.rows();
            const std::size_t cols = input.cols();
            if (rows == 0)
                throw std::invalid_argument("Softmax forward: input must have at least one row");
            output_cache_ = Matrix(rows, cols);

            for (std::size_t j = 0; j < cols; ++j)
            {
                double max_val = input.at_unchecked(0, j);
                for (std::size_t i = 1; i < rows; ++i)
                    if (input.at_unchecked(i, j) > max_val)
                        max_val = input.at_unchecked(i, j);

                double sum_exp = 0.0;
                for (std::size_t i = 0; i < rows; ++i)
                {
                    double e = std::exp(input.at_unchecked(i, j) - max_val);
                    output_cache_.set_value_unchecked(i, j, e);
                    sum_exp += e;
                }
                for (std::size_t i = 0; i < rows; ++i)
                    output_cache_.set_value_unchecked(i, j,
                        output_cache_.at_unchecked(i, j) / sum_exp);
            }
            return output_cache_;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (output_cache_.rows() != grad_output.rows()
                || output_cache_.cols() != grad_output.cols())
                throw std::invalid_argument("Softmax backward: output cache shape mismatch with grad_output");

            const std::size_t rows = grad_output.rows();
            const std::size_t cols = grad_output.cols();
            Matrix grad_input(rows, cols);

            for (std::size_t j = 0; j < cols; ++j)
            {
                double dot = 0.0;
                for (std::size_t i = 0; i < rows; ++i)
                    dot += grad_output.at_unchecked(i, j) * output_cache_.at_unchecked(i, j);
                for (std::size_t i = 0; i < rows; ++i)
                    grad_input.set_value_unchecked(i, j,
                        output_cache_.at_unchecked(i, j)
                        * (grad_output.at_unchecked(i, j) - dot));
            }
            return grad_input;
        }
    };

    // ── PositionalEncoding ───────────────────────────────────────────────────
    class PositionalEncoding final : public Layer
    {
    private:
        Matrix pe_;
        std::size_t d_model_;
        std::size_t max_seq_len_;

    public:
        PositionalEncoding(std::size_t d_model, std::size_t max_seq_len)
            : pe_(d_model, max_seq_len), d_model_(d_model), max_seq_len_(max_seq_len)
        {
            for (std::size_t pos = 0; pos < max_seq_len; ++pos)
                for (std::size_t i = 0; i < d_model; ++i)
                {
                    double angle = static_cast<double>(pos)
                        / std::pow(10000.0, 2.0 * static_cast<double>(i / 2)
                                            / static_cast<double>(d_model));
                    pe_.set_value_unchecked(i, pos,
                        (i % 2 == 0) ? std::sin(angle) : std::cos(angle));
                }
        }

        const char *name() const override { return "PositionalEncoding"; }

        Matrix forward(const Matrix &input) override
        {
            if (input.rows() != d_model_)
                throw std::invalid_argument("PositionalEncoding forward: input rows != d_model");
            if (input.cols() > max_seq_len_)
                throw std::invalid_argument("PositionalEncoding forward: sequence length exceeds max_seq_len");

            Matrix output = input;
            for (std::size_t j = 0; j < input.cols(); ++j)
                for (std::size_t i = 0; i < d_model_; ++i)
                    output.data()[i * input.cols() + j] += pe_.at_unchecked(i, j);
            return output;
        }

        Matrix backward(const Matrix &grad_output) override { return grad_output; }
    };

    // ── MultiHeadAttention ───────────────────────────────────────────────────
    class MultiHeadAttention final : public Layer
    {
    private:
        std::size_t d_model_;
        std::size_t num_heads_;
        std::size_t head_dim_;
        double scale_;

        Matrix W_q_, W_k_, W_v_, W_o_;
        Matrix dW_q_, dW_k_, dW_v_, dW_o_;

        Matrix input_cache_;
        Matrix Q_cache_, K_cache_, V_cache_, concat_cache_;
        std::vector<Matrix> attn_cache_;

        inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

        static Matrix xavier_init(std::size_t rows, std::size_t cols)
        {
            Matrix m(rows, cols);
            double limit = std::sqrt(6.0 / static_cast<double>(rows + cols));
            std::uniform_real_distribution<double> dist(-limit, limit);
            std::generate(m.data().begin(), m.data().end(), [&] { return dist(rng_); });
            return m;
        }

        static Matrix add_bias_broadcast(const Matrix &m, std::size_t rows)
        {
            (void)rows;
            return m;
        }

        static void softmax_rows_inplace(Matrix &m)
        {
            const std::size_t rows = m.rows(), cols = m.cols();
            for (std::size_t i = 0; i < rows; ++i)
            {
                double mx = m.at_unchecked(i, 0);
                for (std::size_t j = 1; j < cols; ++j)
                    if (m.at_unchecked(i, j) > mx) mx = m.at_unchecked(i, j);
                double s = 0.0;
                for (std::size_t j = 0; j < cols; ++j)
                {
                    double e = std::exp(m.at_unchecked(i, j) - mx);
                    m.set_value_unchecked(i, j, e);
                    s += e;
                }
                for (std::size_t j = 0; j < cols; ++j)
                    m.set_value_unchecked(i, j, m.at_unchecked(i, j) / s);
            }
        }

        static Matrix softmax_backward_rows(const Matrix &grad, const Matrix &attn)
        {
            const std::size_t rows = grad.rows(), cols = grad.cols();
            Matrix result(rows, cols);
            for (std::size_t i = 0; i < rows; ++i)
            {
                double dot = 0.0;
                for (std::size_t j = 0; j < cols; ++j)
                    dot += grad.at_unchecked(i, j) * attn.at_unchecked(i, j);
                for (std::size_t j = 0; j < cols; ++j)
                    result.set_value_unchecked(i, j,
                        attn.at_unchecked(i, j) * (grad.at_unchecked(i, j) - dot));
            }
            return result;
        }

        static std::size_t validated_head_dim(std::size_t d_model, std::size_t num_heads)
        {
            if (d_model == 0)
                throw std::invalid_argument("d_model must be > 0");
            if (num_heads == 0)
                throw std::invalid_argument("num_heads must be > 0");
            if (d_model % num_heads != 0)
                throw std::invalid_argument("d_model must be divisible by num_heads");
            return d_model / num_heads;
        }

    public:
        MultiHeadAttention(std::size_t d_model, std::size_t num_heads)
            : d_model_(d_model), num_heads_(num_heads),
              head_dim_(validated_head_dim(d_model, num_heads)),
              scale_(1.0 / std::sqrt(static_cast<double>(head_dim_))),
              W_q_(xavier_init(d_model, d_model)),
              W_k_(xavier_init(d_model, d_model)),
              W_v_(xavier_init(d_model, d_model)),
              W_o_(xavier_init(d_model, d_model)),
              dW_q_(d_model, d_model), dW_k_(d_model, d_model),
              dW_v_(d_model, d_model), dW_o_(d_model, d_model)
        {}

        const char *name() const override { return "MultiHeadAttention"; }
        [[nodiscard]] std::size_t param_count() const noexcept override
        {
            return 4 * d_model_ * d_model_;
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            return {std::ref(W_q_), std::ref(W_k_), std::ref(W_v_), std::ref(W_o_)};
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            return {std::ref(dW_q_), std::ref(dW_k_), std::ref(dW_v_), std::ref(dW_o_)};
        }

        Matrix forward(const Matrix &input) override
        {
            const std::size_t sl = input.cols();
            input_cache_ = input;

            Q_cache_ = W_q_ * input;
            K_cache_ = W_k_ * input;
            V_cache_ = W_v_ * input;

            concat_cache_ = Matrix(d_model_, sl);
            attn_cache_.resize(num_heads_);

            for (std::size_t h = 0; h < num_heads_; ++h)
            {
                const std::size_t off = h * head_dim_;
                Matrix Q_h = Q_cache_.row_slice(off, head_dim_);
                Matrix K_h = K_cache_.row_slice(off, head_dim_);
                Matrix V_h = V_cache_.row_slice(off, head_dim_);

                Matrix scores = Q_h.matmul_TN(K_h);
                scores.scale_inplace(scale_);
                softmax_rows_inplace(scores);
                attn_cache_[h] = scores;

                Matrix head_out = V_h.matmul_NT(scores);
                concat_cache_.set_row_slice(off, head_out);
            }

            return W_o_ * concat_cache_;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            const std::size_t sl = grad_output.cols();

            dW_o_ = grad_output.matmul_NT(concat_cache_);
            Matrix grad_concat = W_o_.matmul_TN(grad_output);

            Matrix grad_Q(d_model_, sl), grad_K(d_model_, sl), grad_V(d_model_, sl);

            for (std::size_t h = 0; h < num_heads_; ++h)
            {
                const std::size_t off = h * head_dim_;
                Matrix Q_h = Q_cache_.row_slice(off, head_dim_);
                Matrix K_h = K_cache_.row_slice(off, head_dim_);
                Matrix V_h = V_cache_.row_slice(off, head_dim_);
                const Matrix &attn_h = attn_cache_[h];

                Matrix grad_head = grad_concat.row_slice(off, head_dim_);

                Matrix grad_V_h = grad_head * attn_h;
                Matrix grad_attn = grad_head.matmul_TN(V_h);

                Matrix grad_scores = softmax_backward_rows(grad_attn, attn_h);
                grad_scores.scale_inplace(scale_);

                Matrix grad_Q_h = K_h.matmul_NT(grad_scores);
                Matrix grad_K_h = Q_h * grad_scores;

                grad_Q.set_row_slice(off, grad_Q_h);
                grad_K.set_row_slice(off, grad_K_h);
                grad_V.set_row_slice(off, grad_V_h);
            }

            dW_q_ = grad_Q.matmul_NT(input_cache_);
            dW_k_ = grad_K.matmul_NT(input_cache_);
            dW_v_ = grad_V.matmul_NT(input_cache_);

            Matrix grad_input = W_q_.matmul_TN(grad_Q);
            grad_input.add_inplace(W_k_.matmul_TN(grad_K));
            grad_input.add_inplace(W_v_.matmul_TN(grad_V));
            return grad_input;
        }
    };

    // ── CausalSelfAttention ──────────────────────────────────────────────────
    class CausalSelfAttention final : public Layer
    {
    private:
        std::size_t d_model_;
        std::size_t num_heads_;
        std::size_t head_dim_;
        double scale_;

        Matrix W_q_, W_k_, W_v_, W_o_;
        Matrix dW_q_, dW_k_, dW_v_, dW_o_;

        Matrix input_cache_;
        Matrix Q_cache_, K_cache_, V_cache_, concat_cache_;
        std::vector<Matrix> attn_cache_;

        inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

        static Matrix xavier_init(std::size_t rows, std::size_t cols)
        {
            Matrix m(rows, cols);
            double limit = std::sqrt(6.0 / static_cast<double>(rows + cols));
            std::uniform_real_distribution<double> dist(-limit, limit);
            std::generate(m.data().begin(), m.data().end(), [&] { return dist(rng_); });
            return m;
        }

        static void softmax_rows_inplace(Matrix &m)
        {
            const std::size_t rows = m.rows(), cols = m.cols();
            for (std::size_t i = 0; i < rows; ++i)
            {
                double mx = m.at_unchecked(i, 0);
                for (std::size_t j = 1; j < cols; ++j)
                    if (m.at_unchecked(i, j) > mx) mx = m.at_unchecked(i, j);
                double s = 0.0;
                for (std::size_t j = 0; j < cols; ++j)
                {
                    double e = std::exp(m.at_unchecked(i, j) - mx);
                    m.set_value_unchecked(i, j, e);
                    s += e;
                }
                for (std::size_t j = 0; j < cols; ++j)
                    m.set_value_unchecked(i, j, m.at_unchecked(i, j) / s);
            }
        }

        static Matrix softmax_backward_rows(const Matrix &grad, const Matrix &attn)
        {
            const std::size_t rows = grad.rows(), cols = grad.cols();
            Matrix result(rows, cols);
            for (std::size_t i = 0; i < rows; ++i)
            {
                double dot = 0.0;
                for (std::size_t j = 0; j < cols; ++j)
                    dot += grad.at_unchecked(i, j) * attn.at_unchecked(i, j);
                for (std::size_t j = 0; j < cols; ++j)
                    result.set_value_unchecked(i, j,
                        attn.at_unchecked(i, j) * (grad.at_unchecked(i, j) - dot));
            }
            return result;
        }

        static std::size_t validated_head_dim(std::size_t d_model, std::size_t num_heads)
        {
            if (d_model == 0)
                throw std::invalid_argument("d_model must be > 0");
            if (num_heads == 0)
                throw std::invalid_argument("num_heads must be > 0");
            if (d_model % num_heads != 0)
                throw std::invalid_argument("d_model must be divisible by num_heads");
            return d_model / num_heads;
        }

    public:
        CausalSelfAttention(std::size_t d_model, std::size_t num_heads)
            : d_model_(d_model), num_heads_(num_heads),
              head_dim_(validated_head_dim(d_model, num_heads)),
              scale_(1.0 / std::sqrt(static_cast<double>(head_dim_))),
              W_q_(xavier_init(d_model, d_model)),
              W_k_(xavier_init(d_model, d_model)),
              W_v_(xavier_init(d_model, d_model)),
              W_o_(xavier_init(d_model, d_model)),
              dW_q_(d_model, d_model), dW_k_(d_model, d_model),
              dW_v_(d_model, d_model), dW_o_(d_model, d_model)
        {}

        const char *name() const override { return "CausalSelfAttention"; }
        [[nodiscard]] std::size_t param_count() const noexcept override
        {
            return 4 * d_model_ * d_model_;
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            return {std::ref(W_q_), std::ref(W_k_), std::ref(W_v_), std::ref(W_o_)};
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            return {std::ref(dW_q_), std::ref(dW_k_), std::ref(dW_v_), std::ref(dW_o_)};
        }

        Matrix forward(const Matrix &input) override
        {
            const std::size_t sl = input.cols();
            input_cache_ = input;

            Q_cache_ = W_q_ * input;
            K_cache_ = W_k_ * input;
            V_cache_ = W_v_ * input;

            concat_cache_ = Matrix(d_model_, sl);
            attn_cache_.resize(num_heads_);

            for (std::size_t h = 0; h < num_heads_; ++h)
            {
                const std::size_t off = h * head_dim_;
                Matrix Q_h = Q_cache_.row_slice(off, head_dim_);
                Matrix K_h = K_cache_.row_slice(off, head_dim_);
                Matrix V_h = V_cache_.row_slice(off, head_dim_);

                Matrix scores = Q_h.matmul_TN(K_h);
                scores.scale_inplace(scale_);

                // Causal mask
                for (std::size_t i = 0; i < sl; ++i)
                    for (std::size_t j = i + 1; j < sl; ++j)
                        scores.set_value_unchecked(i, j, -1e9);

                softmax_rows_inplace(scores);
                attn_cache_[h] = scores;

                Matrix head_out = V_h.matmul_NT(scores);
                concat_cache_.set_row_slice(off, head_out);
            }

            return W_o_ * concat_cache_;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            const std::size_t sl = grad_output.cols();

            dW_o_ = grad_output.matmul_NT(concat_cache_);
            Matrix grad_concat = W_o_.matmul_TN(grad_output);

            Matrix grad_Q(d_model_, sl), grad_K(d_model_, sl), grad_V(d_model_, sl);

            for (std::size_t h = 0; h < num_heads_; ++h)
            {
                const std::size_t off = h * head_dim_;
                Matrix Q_h = Q_cache_.row_slice(off, head_dim_);
                Matrix K_h = K_cache_.row_slice(off, head_dim_);
                Matrix V_h = V_cache_.row_slice(off, head_dim_);
                const Matrix &attn_h = attn_cache_[h];

                Matrix grad_head = grad_concat.row_slice(off, head_dim_);

                Matrix grad_V_h = grad_head * attn_h;
                Matrix grad_attn = grad_head.matmul_TN(V_h);

                Matrix grad_scores = softmax_backward_rows(grad_attn, attn_h);
                grad_scores.scale_inplace(scale_);

                Matrix grad_Q_h = K_h.matmul_NT(grad_scores);
                Matrix grad_K_h = Q_h * grad_scores;

                grad_Q.set_row_slice(off, grad_Q_h);
                grad_K.set_row_slice(off, grad_K_h);
                grad_V.set_row_slice(off, grad_V_h);
            }

            dW_q_ = grad_Q.matmul_NT(input_cache_);
            dW_k_ = grad_K.matmul_NT(input_cache_);
            dW_v_ = grad_V.matmul_NT(input_cache_);

            Matrix grad_input = W_q_.matmul_TN(grad_Q);
            grad_input.add_inplace(W_k_.matmul_TN(grad_K));
            grad_input.add_inplace(W_v_.matmul_TN(grad_V));
            return grad_input;
        }
    };

    // ── FeedForward ──────────────────────────────────────────────────────────
    class FeedForward final : public Layer
    {
    private:
        Linear linear1_;
        GELU gelu_;
        Linear linear2_;

    public:
        FeedForward(std::size_t d_model, std::size_t d_ff)
            : linear1_(d_model, d_ff), gelu_(), linear2_(d_ff, d_model) {}

        const char *name() const override { return "FeedForward"; }
        [[nodiscard]] std::size_t param_count() const noexcept override
        {
            return linear1_.param_count() + linear2_.param_count();
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            auto p = linear1_.parameters();
            auto p2 = linear2_.parameters();
            p.insert(p.end(), p2.begin(), p2.end());
            return p;
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            auto g = linear1_.param_gradients();
            auto g2 = linear2_.param_gradients();
            g.insert(g.end(), g2.begin(), g2.end());
            return g;
        }

        Matrix forward(const Matrix &input) override
        {
            return linear2_.forward(gelu_.forward(linear1_.forward(input)));
        }

        Matrix backward(const Matrix &grad_output) override
        {
            return linear1_.backward(gelu_.backward(linear2_.backward(grad_output)));
        }
    };

    // ── TransformerEncoderLayer ──────────────────────────────────────────────
    class TransformerEncoderLayer final : public Layer
    {
    private:
        LayerNorm norm1_;
        MultiHeadAttention mha_;
        LayerNorm norm2_;
        FeedForward ffn_;
        Matrix normed1_cache_;
        Matrix normed2_cache_;
        Matrix attn_out_cache_;

    public:
        TransformerEncoderLayer(std::size_t d_model, std::size_t num_heads,
                                std::size_t d_ff)
            : norm1_(d_model), mha_(d_model, num_heads),
              norm2_(d_model), ffn_(d_model, d_ff) {}

        const char *name() const override { return "TransformerEncoderLayer"; }
        [[nodiscard]] std::size_t param_count() const noexcept override
        {
            return norm1_.param_count() + mha_.param_count()
                 + norm2_.param_count() + ffn_.param_count();
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            auto p = norm1_.parameters();
            auto p2 = mha_.parameters();
            p.insert(p.end(), p2.begin(), p2.end());
            auto p3 = norm2_.parameters();
            p.insert(p.end(), p3.begin(), p3.end());
            auto p4 = ffn_.parameters();
            p.insert(p.end(), p4.begin(), p4.end());
            return p;
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            auto g = norm1_.param_gradients();
            auto g2 = mha_.param_gradients();
            g.insert(g.end(), g2.begin(), g2.end());
            auto g3 = norm2_.param_gradients();
            g.insert(g.end(), g3.begin(), g3.end());
            auto g4 = ffn_.param_gradients();
            g.insert(g.end(), g4.begin(), g4.end());
            return g;
        }

        Matrix forward(const Matrix &input) override
        {
            normed1_cache_ = norm1_.forward(input);
            attn_out_cache_ = mha_.forward(normed1_cache_);
            Matrix residual1 = input + attn_out_cache_;

            normed2_cache_ = norm2_.forward(residual1);
            Matrix ffn_out = ffn_.forward(normed2_cache_);
            return residual1 + ffn_out;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            Matrix grad_residual1 = grad_output + norm2_.backward(ffn_.backward(grad_output));
            return grad_residual1 + norm1_.backward(mha_.backward(grad_residual1));
        }
    };

    // ── TransformerEncoder ───────────────────────────────────────────────────
    class TransformerEncoder final : public Layer
    {
    private:
        std::vector<TransformerEncoderLayer> layers_;
        std::size_t d_model_;
        std::size_t num_patches_;
        std::size_t cached_batch_size_{0};
        std::vector<Matrix> per_sample_cache_;
        Matrix forward_input_cache_;

    public:
        TransformerEncoder(std::size_t d_model, std::size_t num_heads,
                           std::size_t d_ff, std::size_t num_layers,
                           std::size_t num_patches)
            : d_model_(d_model), num_patches_(num_patches)
        {
            layers_.reserve(num_layers);
            for (std::size_t i = 0; i < num_layers; ++i)
                layers_.emplace_back(d_model, num_heads, d_ff);
        }

        const char *name() const override { return "TransformerEncoder"; }
        [[nodiscard]] std::size_t param_count() const noexcept override
        {
            std::size_t total = 0;
            for (const auto &l : layers_) total += l.param_count();
            return total;
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            std::vector<std::reference_wrapper<Matrix>> result;
            for (auto &l : layers_)
                for (auto &p : l.parameters()) result.push_back(p);
            return result;
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            std::vector<std::reference_wrapper<Matrix>> result;
            for (auto &l : layers_)
                for (auto &g : l.param_gradients()) result.push_back(g);
            return result;
        }

        Matrix forward(const Matrix &input) override
        {
            const std::size_t total_cols = input.cols();
            cached_batch_size_ = total_cols / num_patches_;
            if (cached_batch_size_ == 0 || total_cols % num_patches_ != 0)
                throw std::invalid_argument(
                    "TransformerEncoder: input cols must be a multiple of num_patches");

            forward_input_cache_ = input;
            per_sample_cache_.clear();
            per_sample_cache_.reserve(cached_batch_size_);

            Matrix output(d_model_, cached_batch_size_);

            for (std::size_t b = 0; b < cached_batch_size_; ++b)
            {
                Matrix sample(d_model_, num_patches_);
                for (std::size_t i = 0; i < d_model_; ++i)
                    for (std::size_t j = 0; j < num_patches_; ++j)
                        sample.set_value_unchecked(i, j,
                            input.at_unchecked(i, b * num_patches_ + j));

                for (auto &layer : layers_)
                    sample = layer.forward(sample);

                per_sample_cache_.push_back(sample);

                for (std::size_t i = 0; i < d_model_; ++i)
                {
                    double avg = 0.0;
                    for (std::size_t j = 0; j < num_patches_; ++j)
                        avg += sample.at_unchecked(i, j);
                    output.set_value_unchecked(i, b, avg / static_cast<double>(num_patches_));
                }
            }
            return output;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            const double inv_patches = 1.0 / static_cast<double>(num_patches_);
            Matrix grad_input(d_model_, cached_batch_size_ * num_patches_);

            auto pg = param_gradients();
            std::vector<Matrix> grad_accum;
            grad_accum.reserve(pg.size());
            for (auto &g : pg)
                grad_accum.emplace_back(g.get().rows(), g.get().cols(), 0.0);

            for (std::size_t b = 0; b < cached_batch_size_; ++b)
            {
                Matrix sample(d_model_, num_patches_);
                for (std::size_t i = 0; i < d_model_; ++i)
                    for (std::size_t j = 0; j < num_patches_; ++j)
                        sample.set_value_unchecked(i, j,
                            forward_input_cache_.at_unchecked(i, b * num_patches_ + j));

                for (auto &layer : layers_)
                    sample = layer.forward(sample);

                Matrix grad_sample(d_model_, num_patches_);
                for (std::size_t i = 0; i < d_model_; ++i)
                {
                    double g = grad_output.at_unchecked(i, b) * inv_patches;
                    for (std::size_t j = 0; j < num_patches_; ++j)
                        grad_sample.set_value_unchecked(i, j, g);
                }

                for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
                    grad_sample = it->backward(grad_sample);

                auto current_pg = param_gradients();
                for (std::size_t k = 0; k < grad_accum.size(); ++k)
                    grad_accum[k].add_inplace(current_pg[k].get());

                for (std::size_t i = 0; i < d_model_; ++i)
                    for (std::size_t j = 0; j < num_patches_; ++j)
                        grad_input.set_value_unchecked(i, b * num_patches_ + j,
                            grad_sample.at_unchecked(i, j));
            }

            for (std::size_t k = 0; k < grad_accum.size(); ++k)
                pg[k].get() = grad_accum[k];

            return grad_input;
        }
    };

    // ── GPTBlock ─────────────────────────────────────────────────────────────
    class GPTBlock final : public Layer
    {
    private:
        CausalSelfAttention self_attn_;
        LayerNorm norm1_;
        FeedForward ff_;
        LayerNorm norm2_;

        Matrix residual1_cache_;
        Matrix residual2_cache_;

    public:
        GPTBlock(std::size_t d_model, std::size_t num_heads, std::size_t d_ff)
            : self_attn_(d_model, num_heads),
              norm1_(d_model),
              ff_(d_model, d_ff),
              norm2_(d_model)
        {}

        const char *name() const override { return "GPTBlock"; }
        [[nodiscard]] std::size_t param_count() const noexcept override
        {
            return self_attn_.param_count() + norm1_.param_count()
                 + ff_.param_count() + norm2_.param_count();
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            auto params = self_attn_.parameters();
            auto n1 = norm1_.parameters();
            auto f  = ff_.parameters();
            auto n2 = norm2_.parameters();
            params.insert(params.end(), n1.begin(), n1.end());
            params.insert(params.end(), f.begin(), f.end());
            params.insert(params.end(), n2.begin(), n2.end());
            return params;
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            auto grads = self_attn_.param_gradients();
            auto gn1 = norm1_.param_gradients();
            auto gf  = ff_.param_gradients();
            auto gn2 = norm2_.param_gradients();
            grads.insert(grads.end(), gn1.begin(), gn1.end());
            grads.insert(grads.end(), gf.begin(), gf.end());
            grads.insert(grads.end(), gn2.begin(), gn2.end());
            return grads;
        }

        Matrix forward(const Matrix &input) override
        {
            // 子层1: CausalSelfAttention + 残差 (Pre-Norm)
            residual1_cache_ = input;
            Matrix sa_out = self_attn_.forward(norm1_.forward(input));
            residual2_cache_ = input + sa_out;

            // 子层2: FFN + 残差 (Pre-Norm)
            Matrix ff_out = ff_.forward(norm2_.forward(residual2_cache_));
            return residual2_cache_ + ff_out;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            Matrix grad_residual1 = grad_output;
            Matrix grad_ff_out    = grad_output;

            Matrix b_ff = ff_.backward(grad_ff_out);
            Matrix b_n2 = norm2_.backward(b_ff);
            grad_residual1 = grad_residual1 + b_n2;

            Matrix grad_input = grad_residual1;
            Matrix grad_attn_out = grad_residual1;

            Matrix b_sa = self_attn_.backward(grad_attn_out);
            Matrix b_n1 = norm1_.backward(b_sa);
            grad_input = grad_input + b_n1;

            return grad_input;
        }
    };

    // ── PatchEmbedding ───────────────────────────────────────────────────────
    class PatchEmbedding final : public Layer
    {
    private:
        std::size_t img_size_;
        std::size_t patch_size_;
        std::size_t d_model_;
        std::size_t num_patches_;
        std::size_t patch_dim_;
        Linear proj_;
        Matrix patches_cache_;
        std::size_t cached_batch_size_{0};

        static std::size_t validated_num_patches(std::size_t img_size, std::size_t patch_size)
        {
            if (patch_size == 0)
                throw std::invalid_argument("PatchEmbedding: patch_size must be > 0");
            if (img_size == 0)
                throw std::invalid_argument("PatchEmbedding: img_size must be > 0");
            if (img_size % patch_size != 0)
                throw std::invalid_argument("PatchEmbedding: img_size must be divisible by patch_size");
            const std::size_t grid = img_size / patch_size;
            return grid * grid;
        }

    public:
        PatchEmbedding(std::size_t img_size, std::size_t patch_size, std::size_t d_model)
            : img_size_(img_size), patch_size_(patch_size), d_model_(d_model),
              num_patches_(validated_num_patches(img_size, patch_size)),
              patch_dim_(patch_size * patch_size),
              proj_(patch_dim_, d_model) {}

        const char *name() const override { return "PatchEmbedding"; }
        [[nodiscard]] std::size_t param_count() const noexcept override { return proj_.param_count(); }

        std::vector<std::reference_wrapper<Matrix>> parameters() override { return proj_.parameters(); }
        std::vector<std::reference_wrapper<Matrix>> param_gradients() override { return proj_.param_gradients(); }

        Matrix forward(const Matrix &input) override
        {
            if (input.rows() != img_size_ * img_size_)
                throw std::invalid_argument("PatchEmbedding forward: input rows must equal img_size^2");

            cached_batch_size_ = input.cols();
            const std::size_t grid = img_size_ / patch_size_;

            patches_cache_ = Matrix(patch_dim_, num_patches_ * cached_batch_size_);

            for (std::size_t b = 0; b < cached_batch_size_; ++b)
            {
                for (std::size_t py = 0; py < grid; ++py)
                    for (std::size_t px = 0; px < grid; ++px)
                    {
                        std::size_t pid = py * grid + px;
                        for (std::size_t dy = 0; dy < patch_size_; ++dy)
                            for (std::size_t dx = 0; dx < patch_size_; ++dx)
                            {
                                std::size_t row = py * patch_size_ + dy;
                                std::size_t col = px * patch_size_ + dx;
                                std::size_t pixel_idx = row * img_size_ + col;
                                std::size_t patch_feat = dy * patch_size_ + dx;
                                patches_cache_.set_value_unchecked(
                                    patch_feat, b * num_patches_ + pid,
                                    input.at_unchecked(pixel_idx, b));
                            }
                    }
            }

            return proj_.forward(patches_cache_);
        }

        Matrix backward(const Matrix &grad_output) override
        {
            Matrix grad_patches = proj_.backward(grad_output);
            const std::size_t grid = img_size_ / patch_size_;

            Matrix grad_input(img_size_ * img_size_, cached_batch_size_);

            for (std::size_t b = 0; b < cached_batch_size_; ++b)
            {
                for (std::size_t py = 0; py < grid; ++py)
                    for (std::size_t px = 0; px < grid; ++px)
                    {
                        std::size_t pid = py * grid + px;
                        for (std::size_t dy = 0; dy < patch_size_; ++dy)
                            for (std::size_t dx = 0; dx < patch_size_; ++dx)
                            {
                                std::size_t row = py * patch_size_ + dy;
                                std::size_t col = px * patch_size_ + dx;
                                std::size_t pixel_idx = row * img_size_ + col;
                                std::size_t patch_feat = dy * patch_size_ + dx;
                                grad_input.set_value_unchecked(pixel_idx, b,
                                    grad_patches.at_unchecked(
                                        patch_feat, b * num_patches_ + pid));
                            }
                    }
            }
            return grad_input;
        }
    };
}

#endif
