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
}

#endif
