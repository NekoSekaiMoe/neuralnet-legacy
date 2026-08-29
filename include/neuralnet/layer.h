#ifndef LAYER_HPP
#define LAYER_HPP

#include <algorithm>
#include <cassert>
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
#include <neuralnet/matrix.h>

#include <neuralnet/nn/config.h>

namespace nn
{
    /**
     * @brief Abstract base class for neural network layers.
     */
    class Layer
    {
    public:
        virtual ~Layer() = default;
        /**
         * @brief Performs forward pass computation.
         * @param input Input matrix.
         * @return Output matrix.
         */
        virtual Matrix forward(const Matrix &input) = 0;
        /**
         * @brief Performs backward pass computation.
         * @param grad_output Gradient with respect to output.
         * @return Gradient with respect to input.
         */
        virtual Matrix backward(const Matrix &grad_output) = 0;
        /**
 * @brief Provides access to trainable parameters.
 * @return References to trainable parameter matrices.
 */
        virtual std::vector<std::reference_wrapper<Matrix>> parameters() { return {}; }
        /**
 * @brief Provides references to the layer's parameter gradients.
 * @return A vector of references to parameter gradient matrices.
 */
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

        /**
         * Computes the linear transformation for the input.
         * @param input Input feature matrix.
         * @returns The transformed matrix with the layer bias added.
         * @throws std::invalid_argument If the input feature count does not match the layer.
         */
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
            nn::for_range(result.size(), result.rows(),
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

        /**
         * Applies the rectified linear activation element-wise.
         *
         * @returns A matrix with negative values replaced by zero.
         */
        Matrix forward(const Matrix &input) override
        {
            input_cache_ = input;
            Matrix result(input.rows(), input.cols());
            nn::transform(input.size(), input.data().begin(), input.data().end(),
                           result.data().begin(), [](double value) noexcept
                           { return value > 0.0 ? value : 0.0; });
            return result;
        }

        /**
         * Computes the input gradient for the ReLU activation.
         *
         * @param grad_output Gradient propagated from the subsequent layer.
         * @returns The gradient propagated through positive input values.
         */
        Matrix backward(const Matrix &grad_output) override
        {
            if (input_cache_.rows() != grad_output.rows() || input_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("relu backward shape mismatch");
            }

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            nn::transform(input_cache_.size(),
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

        /**
         * Applies the leaky rectified linear activation element-wise.
         * @param input Input matrix to activate.
         * @returns Matrix containing the activated values.
         */
        Matrix forward(const Matrix &input) override
        {
            input_cache_ = input;
            Matrix result(input.rows(), input.cols());
            nn::transform(input.size(), input.data().begin(), input.data().end(),
                           result.data().begin(),
                           [this](double value) noexcept
                           { return value > 0.0 ? value : negative_slope_ * value; });
            return result;
        }

        /**
         * Computes the gradient propagated through the LeakyReLU activation.
         *
         * @param grad_output Gradient with respect to the layer output.
         * @returns Gradient with respect to the layer input.
         * @throws std::invalid_argument If the gradient shape differs from the cached input shape.
         */
        Matrix backward(const Matrix &grad_output) override
        {
            if (input_cache_.rows() != grad_output.rows() || input_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("leaky_relu backward shape mismatch");
            }

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            nn::transform(input_cache_.size(),
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

        /**
         * Applies the sigmoid activation element-wise.
         *
         * @param input Values to activate.
         * @returns The sigmoid-transformed values.
         */
        Matrix forward(const Matrix &input) override
        {
            Matrix result(input.rows(), input.cols());
            nn::transform(input.size(), input.data().begin(), input.data().end(),
                           result.data().begin(),
                           [](double value) noexcept
                           { return 1.0 / (1.0 + std::exp(-value)); });
            output_cache_ = result;
            return result;
        }

        /**
         * Computes the gradient of the sigmoid activation with respect to its input.
         * @param grad_output Gradient propagated from the following layer.
         * @returns The gradient propagated to the input.
         * @throws std::invalid_argument If the gradient shape differs from the cached output shape.
         */
        Matrix backward(const Matrix &grad_output) override
        {
            if (output_cache_.rows() != grad_output.rows() || output_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("sigmoid backward shape mismatch");
            }

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            nn::transform(output_cache_.size(),
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

        /**
         * Applies the hyperbolic tangent activation element-wise.
         *
         * @param input Values to activate.
         * @returns The activated values, with each element transformed by the hyperbolic tangent.
         */
        Matrix forward(const Matrix &input) override
        {
            Matrix result(input.rows(), input.cols());
            nn::transform(input.size(), input.data().begin(), input.data().end(),
                           result.data().begin(),
                           [](double value) noexcept
                           { return std::tanh(value); });
            output_cache_ = result;
            return result;
        }

        /**
         * Computes the input gradient for the cached hyperbolic tangent output.
         *
         * @param grad_output Gradient propagated from the subsequent layer.
         * @returns The gradient propagated to the input.
         */
        Matrix backward(const Matrix &grad_output) override
        {
            if (output_cache_.rows() != grad_output.rows() || output_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("tanh backward shape mismatch");
            }

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            nn::transform(output_cache_.size(),
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

        /**
         * Applies the Gaussian Error Linear Unit activation to the input.
         * @param input Values to activate.
         * @returns A matrix containing the GELU-activated values.
         */
        Matrix forward(const Matrix &input) override
        {
            input_cache_ = input;
            Matrix result(input.rows(), input.cols());
            nn::transform(input.size(), input.data().begin(), input.data().end(),
                           result.data().begin(), gelu_fn);
            return result;
        }

        /**
         * Computes the gradient of the GELU activation with respect to its input.
         * @param grad_output Gradient propagated from the subsequent layer.
         * @returns The gradient propagated to the preceding layer.
         */
        Matrix backward(const Matrix &grad_output) override
        {
            if (input_cache_.rows() != grad_output.rows() || input_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("gelu backward shape mismatch");
            }

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            nn::transform(input_cache_.size(),
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

        /**
         * Applies inverted dropout during training.
         * @param input Values to regularize.
         * @returns The masked and scaled input during training, or the unchanged input otherwise.
         */
        Matrix forward(const Matrix &input) override
        {
            if (!training_ || p_ == 0.0)
            {
                return input;
            }

            const double scale = 1.0 / (1.0 - p_);
            std::bernoulli_distribution dist(1.0 - p_);
            mask_ = Matrix(input.rows(), input.cols());

            // RNG 必须串行：mt19937 并发调用是数据竞争（UB），且串行保证可复现
            for (std::size_t idx = 0; idx < input.size(); ++idx)
                mask_.data()[idx] = dist(rng_) ? scale : 0.0;

            Matrix result(input.rows(), input.cols());
            nn::transform(input.size(),
                           input.data().begin(), input.data().end(),
                           mask_.data().begin(),
                           result.data().begin(),
                           [](double val, double m) noexcept
                           { return val * m; });
            return result;
        }

        /**
         * Propagates gradients through the dropout layer.
         * @param grad_output Gradient received from the subsequent layer.
         * @returns The input gradient, with the dropout mask applied during training or unchanged when dropout is inactive.
         * @throws std::invalid_argument If the gradient shape does not match the cached dropout mask.
         */
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
            nn::transform(grad_output.size(),
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
    /**
         * Normalizes feature rows across the batch and optionally applies learnable
         * scale and bias parameters.
         *
         * @param num_features Number of feature rows in each input.
         * @param eps Small value added to variance for numerical stability.
         * @param momentum Weight assigned to each batch's statistics when updating
         * running statistics.
         * @param affine Whether to apply learnable scale and bias parameters.
         */
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
        /**
             * Initializes a one-dimensional batch normalization layer.
             *
             * @param num_features Number of input features.
             * @param eps Small value added to the variance for numerical stability.
             * @param momentum Weight assigned to the current batch when updating running statistics.
             * @param affine Whether to apply learnable scaling and bias parameters.
             */
            bool is_training_{true};

        Matrix input_cache_;
        Matrix batch_mean_;   // (num_features, 1)
        Matrix batch_var_;    // (num_features, 1)
        Matrix normalized_;   // (num_features, batch_size)

        // 行主序下，按行求和 = 按特征聚合（每个 i 一行 = 一个特征的 batch 内值）
        static std::vector<double> rowwise_sum(const Matrix &m)
        {
            std::vector<double> result(m.rows(), 0.0);
            nn::for_range(m.size(), m.rows(),
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
            nn::for_range(m.rows(), m.rows(),
                          [&](std::size_t i) noexcept { result[i] /= denom; });
            return result;
        }

        // 行主序下，按行求总体方差（分母 = cols_，PyTorch BatchNorm 约定）
        static std::vector<double> rowwise_var(const Matrix &m, const std::vector<double> &mean)
        {
            std::vector<double> result(m.rows(), 0.0);
            if (m.cols() == 0) return result;
            nn::for_range(m.size(), m.rows(),
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
                nn::transform(running_mean_.size(),
                               running_mean_.data().begin(), running_mean_.data().end(),
                               batch_mean_.data().begin(),
                               running_mean_.data().begin(),
                               [this](double rm, double bm) noexcept
                               { return (1.0 - momentum_) * rm + momentum_ * bm; });
                nn::transform(running_var_.size(),
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
            nn::for_range(input.size(), input.size(),
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
            nn::for_range(input.size(), input.size(),
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
                nn::for_range(grad_output.size(), grad_output.size(),
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
            nn::for_range(dx.size(), dx.size(),
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
    /**
     * Identifies the layer as a two-dimensional batch-normalization layer.
     *
     * @returns The layer name, `"BatchNorm2d"`.
     */
    class BatchNorm2d : public BatchNorm1d
    {
    public:
        using BatchNorm1d::BatchNorm1d;
        const char *name() const override { return "BatchNorm2d"; }
    };

    // ── Conv2D（im2col + GEMM，移植自上游 compute_layer_conv.hpp）──────
    // 布局：输入/输出 (C*H*W, batch)，列 = batch 样本；
    //       权重 W (C_out, C_in*k*k) + 偏置 b (C_out, 1)。
    // forward:  col = im2col(x) (C_in*k*k, batch*OH*OW)
    //           Z = W × col + b → 重排 (C_out*OH*OW, batch)
    // backward: gZ = 重排(grad_out)；grad_W = gZ×colᵀ；grad_b = 行和(gZ)；
    //           grad_col = Wᵀ×gZ → col2im 散射累加 → grad_x
    // 梯度语义：替换（zero 后填），与 Linear 一致。
    /**
     * @brief 2D convolution layer using im2col + GEMM algorithm.
     *
     * Implements spatial convolution with configurable kernel size, stride, and padding.
     * Layout: input/output shape is (C*H*W, batch), weights are (C_out, C_in*k*k).
     * Uses im2col transformation followed by matrix multiplication for efficiency.
     */
    class Conv2D final : public Layer
    {
    private:
        std::size_t in_channels_, out_channels_;
        std::size_t kernel_, stride_, padding_;
        std::size_t in_h_, in_w_;
        std::size_t out_h_, out_w_;

        Matrix W_, b_;
        Matrix grad_W_, grad_b_;
        Matrix col_cache_; /**
 * Thread-local random number generator for stochastic layer operations.
 */

        inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

        // im2col：input (C_in*H*W, batch) → col (C_in*k*k, batch*OH*OW)
        /**
                           * Converts batched image data into column-form patches for convolution.
                           *
                           * @param input Flattened input images arranged by channel, height, width, and batch.
                           * @param batch Number of images in the batch.
                           * @return Matrix containing one flattened convolution window per output position, with zero padding outside the input boundaries.
                           */
        [[nodiscard]] Matrix im2col_(const Matrix &input, std::size_t batch) const
        {
            const std::size_t C_in = in_channels_, k = kernel_;
            const std::size_t H_out = out_h_, W_out = out_w_;
            Matrix col(C_in * k * k, batch * H_out * W_out);

            nn::for_range(col.size(), batch * H_out * W_out,
                          [&](std::size_t pos)
                          {
                              const std::size_t b = pos / (H_out * W_out);
                              const std::size_t oh = (pos / W_out) % H_out;
                              const std::size_t ow = pos % W_out;
                              for (std::size_t ci = 0; ci < C_in; ++ci)
                                  for (std::size_t kh = 0; kh < k; ++kh)
                                      for (std::size_t kw = 0; kw < k; ++kw)
                                      {
                                          const long ih = static_cast<long>(oh * stride_ + kh)
                                                          - static_cast<long>(padding_);
                                          const long iw = static_cast<long>(ow * stride_ + kw)
                                                          - static_cast<long>(padding_);
                                          double v = 0.0; // 零填充
                                          if (ih >= 0 && iw >= 0
                                              && ih < static_cast<long>(in_h_)
                                              && iw < static_cast<long>(in_w_))
                                              v = input.at_unchecked(
                                                  ci * in_h_ * in_w_ + ih * in_w_ + iw, b);
                                          col.set_value_unchecked(ci * k * k + kh * k + kw, pos, v);
                                      }
                          });
            return col;
        }

        // col2im：col (C_in*k*k, batch*OH*OW) → (C_in*H*W, batch)，散射累加。
        // 同一 (b, ci) 内不同输出位置可能写同一输入格（stride < k 时窗口重叠），
        /**
         * Reconstructs batched image-shaped values from their column representation.
         *
         * @param col Column representation of the values to reconstruct.
         * @param batch Number of samples in the batch.
         * @returns Reconstructed values with one flattened image per batch sample.
         */
        [[nodiscard]] Matrix col2im_(const Matrix &col, std::size_t batch) const
        {
            const std::size_t C_in = in_channels_, k = kernel_;
            const std::size_t H_out = out_h_, W_out = out_w_;
            Matrix out(C_in * in_h_ * in_w_, batch);

            nn::for_range(col.size(), batch * C_in,
                          [&](std::size_t bc)
                          {
                              const std::size_t b = bc / C_in;
                              const std::size_t ci = bc % C_in;
                              for (std::size_t oh = 0; oh < H_out; ++oh)
                                  for (std::size_t ow = 0; ow < W_out; ++ow)
                                      for (std::size_t kh = 0; kh < k; ++kh)
                                          for (std::size_t kw = 0; kw < k; ++kw)
                                          {
                                              const long ih = static_cast<long>(oh * stride_ + kh)
                                                              - static_cast<long>(padding_);
                                              const long iw = static_cast<long>(ow * stride_ + kw)
                                                              - static_cast<long>(padding_);
                                              if (ih < 0 || iw < 0
                                                  || ih >= static_cast<long>(in_h_)
                                                  || iw >= static_cast<long>(in_w_))
                                                  continue;
                                              const std::size_t r = ci * k * k + kh * k + kw;
                                              const std::size_t orow = ci * in_h_ * in_w_
                                                                      + ih * in_w_ + iw;
                                              out.set_value_unchecked(
                                                  orow, b,
                                                  out.at_unchecked(orow, b)
                                                      + col.at_unchecked(r, b * H_out * W_out
                                                                         + oh * W_out + ow));
                                          }
                          });
            return out;
        }

        /**
         * Rearranges convolution outputs from channel-major layout into sample-major layout.
         * @param Z Convolution output matrix with columns grouped by batch sample and spatial position.
         * @param batch Number of samples in the batch.
         * @returns Matrix arranged with all output channels and spatial positions per sample.
         */
        [[nodiscard]] Matrix cols_to_samples_(const Matrix &Z, std::size_t batch) const
        {
            const std::size_t area = out_h_ * out_w_;
            Matrix out(out_channels_ * area, batch);
            nn::for_range(out.size(), out.size(),
                          [&](std::size_t idx)
                          {
                              const std::size_t r = idx / batch; // (co*area + p)
                              const std::size_t b = idx % batch;
                              const std::size_t co = r / area, p = r % area;
                              out.data()[idx] = Z.at_unchecked(co, b * area + p);
                          });
            return out;
        }

        /**
         * Rearranges convolution outputs by grouping spatial positions within each batch.
         * @param out Output matrix arranged as `(channels * height * width, batch)`.
         * @param batch Number of input samples.
         * @returns Matrix arranged as `(channels, batch * height * width)`.
         */
        [[nodiscard]] Matrix samples_to_cols_(const Matrix &out, std::size_t batch) const
        {
            const std::size_t area = out_h_ * out_w_;
            Matrix Z(out_channels_, batch * area);
            nn::for_range(Z.size(), Z.size(),
                          [&](std::size_t idx)
                          {
                              const std::size_t co = idx / (batch * area);
                              const std::size_t j = idx % (batch * area);
                              const std::size_t b = j / area, p = j % area;
                              Z.data()[idx] = out.at_unchecked(co * area + p, b);
                          });
            return Z;
        }

    public:
        Conv2D(std::size_t in_channels, std::size_t out_channels,
               std::size_t kernel, std::size_t in_h, std::size_t in_w,
               std::size_t stride = 1, std::size_t padding = 0)
            : in_channels_(in_channels), out_channels_(out_channels),
              kernel_(kernel), stride_(stride != 0 ? stride : 1), padding_(padding),
              in_h_(in_h), in_w_(in_w),
              W_(out_channels_, in_channels_ * kernel_ * kernel_),
              b_(out_channels_, 1),
              grad_W_(out_channels_, in_channels_ * kernel_ * kernel_),
              grad_b_(out_channels_, 1)
        {
            // 守卫：kernel 过大时 (in + 2*pad - kernel) 无符号下溢 → 巨尺寸分配
            if (kernel_ == 0 || in_h_ + 2 * padding_ < kernel_ || in_w_ + 2 * padding_ < kernel_)
                throw std::invalid_argument("Conv2D: kernel 过大 (kernel > in + 2*padding)");
            out_h_ = (in_h_ + 2 * padding_ - kernel_) / stride_ + 1;
            out_w_ = (in_w_ + 2 * padding_ - kernel_) / stride_ + 1;

            // He 风格均匀初始化（fan_in = C_in*k*k），与 Linear 的 Xavier 同风格
            const std::size_t fan_in = in_channels_ * kernel_ * kernel_;
            const double limit = std::sqrt(6.0 / static_cast<double>(fan_in + out_channels_));
            std::uniform_real_distribution<double> dist(-limit, limit);
            std::generate(W_.data().begin(), W_.data().end(), [&] { return dist(rng_); });
            // b_ 零初始化（Matrix(rows, cols) 默认 0）
        }

        /**
 * Identifies the layer type.
 *
 * @return The layer name, "Conv2D".
 */
const char *name() const override { return "Conv2D"; }
        [[nodiscard]] std::size_t param_count() const noexcept override
        {
            return W_.size() + b_.size();
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            return {std::ref(W_), std::ref(b_)};
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            return {std::ref(grad_W_), std::ref(grad_b_)};
        }

        /**
         * Computes the convolution output for a batch of flattened images.
         *
         * @param input Batch of flattened images with dimensions
         *              {@code in_channels * in_h * in_w} by batch size.
         * @returns Convolution results with dimensions
         *          {@code out_channels * out_h * out_w} by batch size.
         * @throws std::invalid_argument If the input feature count is invalid.
         */
        Matrix forward(const Matrix &input) override
        {
            if (input.rows() != in_channels_ * in_h_ * in_w_)
                throw std::invalid_argument("Conv2D forward: input rows != C_in*H*W");
            const std::size_t batch = input.cols();

            col_cache_ = im2col_(input, batch);

            // Z = W × col → (C_out, batch*OH*OW)，逐行加偏置
            Matrix Z = W_ * col_cache_;
            {
                const std::size_t zcols = Z.cols();
                nn::for_range(Z.size(), Z.size(),
                              [&](std::size_t idx)
                              {
                                  const std::size_t co = idx / zcols;
                                  Z.data()[idx] += b_.at_unchecked(co, 0);
                              });
            }
            return cols_to_samples_(Z, batch);
        }

        /**
         * Computes parameter gradients and propagates gradients to the convolution input.
         *
         * @param grad_output Gradient of the loss with respect to the layer output.
         * @returns Gradient of the loss with respect to the layer input.
         * @throws std::invalid_argument If forward() has not been called or grad_output has an invalid shape.
         */
        Matrix backward(const Matrix &grad_output) override
        {
            if (col_cache_.empty())
                throw std::invalid_argument("Conv2D backward: forward not called");
            const std::size_t batch = grad_output.cols();
            if (grad_output.rows() != out_channels_ * out_h_ * out_w_)
                throw std::invalid_argument("Conv2D backward: grad_output rows != C_out*OH*OW");

            Matrix gZ = samples_to_cols_(grad_output, batch);

            // grad_W = gZ × colᵀ（替换语义）；grad_b = gZ 行和
            grad_W_ = gZ.matmul_NT(col_cache_);
            const std::vector<double> gb = gZ.rowwise_sum();
            for (std::size_t co = 0; co < out_channels_; ++co)
                grad_b_.set_value_unchecked(co, 0, gb[co]);

            // grad_col = Wᵀ × gZ → col2im 散射
            Matrix grad_col = W_.matmul_TN(gZ);
            return col2im_(grad_col, batch);
        }
    };

    // ── MaxPool2D（记录 argmax，backward 散射；移植自上游）────────────
    // 输入/输出 (C*H*W, batch)；Hp = (H-pool)/stride+1，Wp 同理。
    // forward:  每 pool×pool 窗口取 max 并记录行索引；
    // backward: 梯度散射回 argmax 位置（重叠窗口共享 argmax 时累加）。
    /**
     * @brief 2D max pooling layer with argmax tracking for backpropagation.
     *
     * Downsamples spatial dimensions by taking the maximum value in each pooling window.
     * Layout: input/output shape is (C*H*W, batch).
     * Output dimensions: out_h = (in_h - pool) / stride + 1, same for width.
     * Gradient flows back only to the max element in each window.
     */
    class MaxPool2D final : public Layer
    {
    private:
        std::size_t channels_, in_h_, in_w_;
        std::size_t pool_, stride_;
        std::size_t out_h_, out_w_;
        std::vector<std::size_t> max_indices_; // (C*Hp*Wp, batch) 扁平 argmax 行索引

    public:
        MaxPool2D(std::size_t channels, std::size_t in_h, std::size_t in_w,
                  std::size_t pool = 2, std::size_t stride = 0)
            : channels_(channels), in_h_(in_h), in_w_(in_w),
              pool_(pool), stride_(stride != 0 ? stride : (pool != 0 ? pool : 1))
        {
            if (pool_ == 0 || in_h_ < pool_ || in_w_ < pool_)
                throw std::invalid_argument("MaxPool2D: pool 窗口大于输入尺寸");
            out_h_ = (in_h_ - pool_) / stride_ + 1;
            out_w_ = (in_w_ - pool_) / stride_ + 1;
        }

        const char *name() const override { return "MaxPool2D"; }

        /**
                           * Applies max pooling to each channel of a batched flattened image tensor.
                           *
                           * @param input Input matrix shaped `(channels * input_height * input_width, batch)`.
                           * @returns Pooled output shaped `(channels * output_height * output_width, batch)`.
                           * @throws std::invalid_argument If the input row count does not match the configured image shape.
                           */
                          Matrix forward(const Matrix &input) override
        {
            if (input.rows() != channels_ * in_h_ * in_w_)
                throw std::invalid_argument("MaxPool2D forward: input rows != C*H*W");
            const std::size_t batch = input.cols();
            const std::size_t out_area = out_h_ * out_w_;

            Matrix out(channels_ * out_area, batch);
            max_indices_.assign(channels_ * out_area * batch, 0);

            // 逐窗口独立 → 安全并行；work = 输出元素数 × pool²
            nn::for_range(out.size() * pool_ * pool_, out.size(),
                          [&](std::size_t oidx)
                          {
                              const std::size_t b = oidx % batch;
                              const std::size_t orow = oidx / batch;
                              const std::size_t c = orow / out_area;
                              const std::size_t p = orow % out_area;
                              const std::size_t oh = p / out_w_, ow = p % out_w_;

                              double best = -std::numeric_limits<double>::infinity();
                              std::size_t best_idx = 0;
                              for (std::size_t dh = 0; dh < pool_; ++dh)
                                  for (std::size_t dw = 0; dw < pool_; ++dw)
                                  {
                                      const std::size_t r = c * in_h_ * in_w_
                                          + (oh * stride_ + dh) * in_w_
                                          + (ow * stride_ + dw);
                                      const double v = input.at_unchecked(r, b);
                                      if (v > best)
                                      {
                                          best = v;
                                          best_idx = r;
                                      }
                                  }
                              out.data()[oidx] = best;
                              max_indices_[b * channels_ * out_area + orow] = best_idx;
                          });
            return out;
        }

        /**
         * Propagates gradients through the max-pooling operation.
         *
         * @param grad_output Gradients with respect to the pooled output.
         * @returns Gradients with respect to the input, accumulated at the selected maxima.
         */
        Matrix backward(const Matrix &grad_output) override
        {
            const std::size_t batch = grad_output.cols();
            const std::size_t out_area = out_h_ * out_w_;
            if (grad_output.rows() != channels_ * out_area)
                throw std::invalid_argument("MaxPool2D backward: grad_output rows mismatch");
            if (max_indices_.size() != channels_ * out_area * batch)
                throw std::invalid_argument("MaxPool2D backward: forward not called");

            // 同一 (b, c) 内重叠窗口可能散射到同一行 → 仅 (b, c) 粒度并行，块内串行累加
            Matrix gin(channels_ * in_h_ * in_w_, batch);
            nn::for_range(gin.size(), batch * channels_,
                          [&](std::size_t bc)
                          {
                              const std::size_t b = bc / channels_;
                              const std::size_t c = bc % channels_;
                              for (std::size_t p = 0; p < out_area; ++p)
                              {
                                  const std::size_t orow = c * out_area + p;
                                  const std::size_t idx = max_indices_[b * channels_ * out_area + orow];
                                  gin.set_value_unchecked(
                                      idx, b,
                                      gin.at_unchecked(idx, b) + grad_output.at_unchecked(orow, b));
                              }
                          });
            return gin;
        }
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

        /**
         * Normalizes each input sample across its feature dimension and applies learnable scale and bias.
         *
         * @param input Input matrix with one sample per column and `normalized_shape_` rows.
         * @returns The normalized and affine-transformed samples.
         * @throws std::invalid_argument If the input row count does not match the normalized shape.
         */
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
            nn::for_range(feat * batch, feat * batch,
                          [&](std::size_t idx) {
                              std::size_t i = idx / batch;
                              output.data()[idx] = normalized_cache_.data()[idx]
                                  * gamma_.at_unchecked(i, 0) + beta_.at_unchecked(i, 0);
                          });
            return output;
        }

        /**
         * Computes input gradients and parameter gradients for the layer-normalized samples.
         *
         * @param grad_output Gradient propagated from the subsequent layer.
         * @returns The gradient with respect to the layer input.
         * @throws std::invalid_argument If the gradient or cached forward-pass shape is invalid.
         */
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

    // ── RMSNorm（Root Mean Square 归一化，LLaMA/Mistral 风格，移植自上游）──
    // 与 LayerNorm 差异：不减均值、无 beta 偏置，仅按均方根归一化，
    // 每层少 2 次列归约 + 1 次广播。公式（对标上游 compute_layer_mlp.hpp）：
    //   forward:  mean_sq = (1/F)Σx²;  rms_inv = rsqrt(mean_sq + ε);
    //             normed = x·rms_inv;  out = normed·γ
    //   backward: gy = g·γ;  gy_n = gy⊙normed;  m = (1/F)Σ_f gy_n;
    //             grad_x = (gy − m·normed)·rms_inv;  grad_γ = Σ_batch gy_n
    /**
     * @brief Root Mean Square Layer Normalization (LLaMA/Mistral style).
     *
     * Simpler than LayerNorm: no mean subtraction, no beta bias parameter.
     * Normalizes by RMS: out = (x / sqrt(mean(x^2) + eps)) * gamma.
     * More efficient than LayerNorm (fewer reductions and broadcasts).
     * Commonly used in modern LLMs like LLaMA and Mistral.
     */
    class RMSNorm final : public Layer
    {
    private:
        std::size_t normalized_shape_;
        double eps_;
        Matrix gamma_;   // (normalized_shape, 1)，初始化为 1（无 beta）
        Matrix dgamma_;

        // backward 缓存
        Matrix normalized_cache_;   // (features, batch)
        std::vector<double> rms_inv_cache_; // (batch)

    public:
        explicit RMSNorm(std::size_t normalized_shape, double eps = 1e-5)
            : normalized_shape_(normalized_shape), eps_(eps),
              gamma_(normalized_shape, 1, 1.0),
              dgamma_(normalized_shape, 1, 0.0) {}

        const char *name() const override { return "RMSNorm"; }
        [[nodiscard]] std::size_t param_count() const noexcept override { return normalized_shape_; }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            return {std::ref(gamma_)};
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            return {std::ref(dgamma_)};
        }

        Matrix forward(const Matrix &input) override
        {
            if (input.rows() != normalized_shape_)
                throw std::invalid_argument("RMSNorm forward: input rows != normalized_shape");

            const std::size_t feat = input.rows();
            const std::size_t batch = input.cols();
            const double inv_feat = 1.0 / static_cast<double>(feat);

            rms_inv_cache_.resize(batch);
            normalized_cache_ = Matrix(feat, batch);

            for (std::size_t j = 0; j < batch; ++j)
            {
                double sum_sq = 0.0;
                for (std::size_t i = 0; i < feat; ++i)
                {
                    const double x = input.at_unchecked(i, j);
                    sum_sq += x * x;
                }
                const double rms_inv = 1.0 / std::sqrt(sum_sq * inv_feat + eps_);
                rms_inv_cache_[j] = rms_inv;
                for (std::size_t i = 0; i < feat; ++i)
                    normalized_cache_.set_value_unchecked(i, j, input.at_unchecked(i, j) * rms_inv);
            }

            Matrix output(feat, batch);
            nn::for_range(feat * batch, feat * batch,
                          [&](std::size_t idx)
                          {
                              const std::size_t i = idx / batch;
                              output.data()[idx] = normalized_cache_.data()[idx]
                                  * gamma_.at_unchecked(i, 0);
                          });
            return output;
        }

        /**
         * Computes gradients for the RMS normalization layer.
         *
         * @param grad_output Gradient propagated from the subsequent layer.
         * @returns The gradient with respect to the layer input.
         * @throws std::invalid_argument If the gradient or cached forward-pass shape is invalid.
         */
        Matrix backward(const Matrix &grad_output) override
        {
            if (grad_output.rows() != normalized_shape_)
                throw std::invalid_argument("RMSNorm backward: grad_output rows != normalized_shape");
            if (normalized_cache_.rows() != normalized_shape_
                || normalized_cache_.cols() != grad_output.cols())
                throw std::invalid_argument("RMSNorm backward: forward cache shape mismatch");

            const std::size_t feat = grad_output.rows();
            const std::size_t batch = grad_output.cols();
            const double inv_feat = 1.0 / static_cast<double>(feat);

            // 行主序友好：按行同时累计 grad_γ 与每列的 m[j] = (1/F)Σ_i gy_n
            dgamma_.zero();
            std::vector<double> m_vec(batch, 0.0);
            for (std::size_t i = 0; i < feat; ++i)
            {
                const double g_i = gamma_.at_unchecked(i, 0);
                double dg = 0.0;
                for (std::size_t j = 0; j < batch; ++j)
                {
                    const double grad = grad_output.at_unchecked(i, j);
                    const double normalized = normalized_cache_.at_unchecked(i, j);
                    dg += grad * normalized;
                    const double gy_n = grad * g_i * normalized;
                    m_vec[j] += gy_n;
                }
                dgamma_.set_value_unchecked(i, 0, dg);
            }

            // grad_x = (gy − m·normed)·rms_inv
            Matrix grad_input(feat, batch);
            nn::for_range(feat * batch, feat * batch,
                          [&](std::size_t idx)
                          {
                              const std::size_t i = idx / batch;
                              const std::size_t j = idx % batch;
                              const double gy = grad_output.data()[idx]
                                                * gamma_.at_unchecked(i, 0);
                              grad_input.data()[idx] =
                                  (gy - m_vec[j] * inv_feat * normalized_cache_.data()[idx])
                                  * rms_inv_cache_[j];
                          });
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
        bool causal_;

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
        MultiHeadAttention(std::size_t d_model, std::size_t num_heads, bool causal = false)
            : d_model_(d_model), num_heads_(num_heads),
              head_dim_(validated_head_dim(d_model, num_heads)),
              scale_(1.0 / std::sqrt(static_cast<double>(head_dim_))),
              causal_(causal),
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
                if (causal_)
                {
                    for (std::size_t i = 0; i < sl; ++i)
                        for (std::size_t j = i + 1; j < sl; ++j)
                            scores.set_value_unchecked(i, j, -1e9);
                }
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

                if (causal_)
                {
                    for (std::size_t i = 0; i < sl; ++i)
                        for (std::size_t j = i + 1; j < sl; ++j)
                            grad_scores.set_value_unchecked(i, j, 0.0);
                }

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

        /**
         * Computes the gradient propagated through the feed-forward network.
         * @param grad_output Gradient of the loss with respect to the layer output.
         * @returns Gradient of the loss with respect to the layer input.
         */
        Matrix backward(const Matrix &grad_output) override
        {
            return linear1_.backward(gelu_.backward(linear2_.backward(grad_output)));
        }
    };

    // ── SwiGLU（SiLU 门控 FFN 激活，LLaMA 风格，移植自上游）────────────
    // 输入 (2·d_ff, batch)：gate = 前 d_ff 行，up = 后 d_ff 行
    //   forward:  s = σ(gate);  out = SiLU(gate) ⊙ up = gate·s·up
    //   backward: grad_gate = g⊙up⊙s⊙(1 + gate⊙(1−s))
    //             grad_up   = g⊙gate⊙s
    //             两者拼回 (2·d_ff, batch)
    /**
     * @brief Swish-Gated Linear Unit activation (LLaMA-style).
     *
     * Gated activation function: out = SiLU(gate) * up, where SiLU(x) = x * sigmoid(x).
     * Input shape: (2*d_ff, batch), split into gate (first d_ff rows) and up (last d_ff rows).
     * Output shape: (d_ff, batch).
     * Used in modern LLMs as a drop-in replacement for GELU in FFN blocks.
     */
    class SwiGLU final : public Layer
    {
    private:
        std::size_t d_ff_;
        Matrix gate_cache_;    // (d_ff, batch)
        Matrix sigmoid_cache_; // σ(gate)
        Matrix up_cache_;      // (d_ff, batch)

    public:
        explicit SwiGLU(std::size_t d_ff) : d_ff_(d_ff) {}

        const char *name() const override { return "SwiGLU"; }

        /**
         * Computes the gated linear unit activation for the input.
         *
         * @param input Input containing gate and up components with 2*d_ff_ rows.
         * @returns The gated activation with d_ff_ rows.
         * @throws std::invalid_argument If the input does not have 2*d_ff_ rows.
         */
        Matrix forward(const Matrix &input) override
        {
            if (input.rows() != 2 * d_ff_)
                throw std::invalid_argument("SwiGLU forward: input rows != 2*d_ff");

            const std::size_t batch = input.cols();
            gate_cache_ = input.row_slice(0, d_ff_);
            up_cache_ = input.row_slice(d_ff_, d_ff_);
            sigmoid_cache_ = Matrix(d_ff_, batch);

            nn::transform(gate_cache_.size(),
                          gate_cache_.data().begin(), gate_cache_.data().end(),
                          sigmoid_cache_.data().begin(),
                          [](double g) noexcept { return 1.0 / (1.0 + std::exp(-g)); });

            Matrix output(d_ff_, batch);
            nn::for_range(output.size(), output.size(),
                          [&](std::size_t idx) noexcept
                          {
                              output.data()[idx] = gate_cache_.data()[idx]
                                  * sigmoid_cache_.data()[idx] * up_cache_.data()[idx];
                          });
            return output;
        }

        /**
         * Computes gradients for the gate and up portions of the input.
         * @param grad_output Gradient propagated from the following layer.
         * @returns Gradients concatenated for the gate and up input portions.
         */
        Matrix backward(const Matrix &grad_output) override
        {
            if (grad_output.rows() != d_ff_)
                throw std::invalid_argument("SwiGLU backward: grad_output rows != d_ff");
            if (gate_cache_.rows() != d_ff_ || gate_cache_.cols() != grad_output.cols())
                throw std::invalid_argument("SwiGLU backward: forward cache shape mismatch");

            const std::size_t batch = grad_output.cols();
            Matrix grad_gate(d_ff_, batch), grad_up(d_ff_, batch);

            nn::for_range(grad_output.size(), grad_output.size(),
                          [&](std::size_t idx) noexcept
                          {
                              const double g = grad_output.data()[idx];
                              const double s = sigmoid_cache_.data()[idx];
                              const double gate = gate_cache_.data()[idx];
                              const double up = up_cache_.data()[idx];
                              grad_gate.data()[idx] = g * up * s * (1.0 + gate * (1.0 - s));
                              grad_up.data()[idx] = g * gate * s;
                          });

            Matrix grad_input(2 * d_ff_, batch);
            grad_input.set_row_slice(0, grad_gate);
            grad_input.set_row_slice(d_ff_, grad_up);
            return grad_input;
        }
    };

    // ── SwiGLUFeedForward（LLaMA 风格 FFN，移植自上游）──────────────────
    // Linear(d_model → 2·d_ff) → SwiGLU → Linear(d_ff → d_model)
    // 与 FeedForward（GELU 版）同构，可直接替换。
    /**
     * @brief Feed-forward network with SwiGLU activation (LLaMA-style).
     *
     * Architecture: Linear(d_model -> 2*d_ff) -> SwiGLU -> Linear(d_ff -> d_model).
     * Drop-in replacement for FeedForward (which uses GELU).
     * Commonly used in LLaMA and similar modern transformer architectures.
     */
    class SwiGLUFeedForward final : public Layer
    {
    private:
        Linear linear1_;
        SwiGLU swiglu_;
        Linear linear2_;

    public:
        SwiGLUFeedForward(std::size_t d_model, std::size_t d_ff)
            : linear1_(d_model, 2 * d_ff), swiglu_(d_ff), linear2_(d_ff, d_model) {}

        const char *name() const override { return "SwiGLUFeedForward"; }
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

        /**
         * Projects the input through the SwiGLU feed-forward network.
         *
         * @param input Input feature matrix.
         * @returns The transformed output matrix.
         */
        Matrix forward(const Matrix &input) override
        {
            return linear2_.forward(swiglu_.forward(linear1_.forward(input)));
        }

        /**
         * Propagates gradients through the feed-forward network.
         *
         * @param grad_output Gradient from the subsequent layer.
         * @return Gradient with respect to the network input.
         */
        Matrix backward(const Matrix &grad_output) override
        {
            return linear1_.backward(swiglu_.backward(linear2_.backward(grad_output)));
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
        MultiHeadAttention self_attn_;
        LayerNorm norm1_;
        FeedForward ff_;
        LayerNorm norm2_;


    public:
        GPTBlock(std::size_t d_model, std::size_t num_heads, std::size_t d_ff)
            : self_attn_(d_model, num_heads, true),
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
            Matrix sa_out = self_attn_.forward(norm1_.forward(input));
            Matrix residual2 = input + sa_out;

            // 子层2: FFN + 残差 (Pre-Norm)
            Matrix ff_out = ff_.forward(norm2_.forward(residual2));
            return residual2 + ff_out;
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

    // ── GPTModel ─────────────────────────────────────────────────────────────
    class GPTModel final : public Layer
    {
    private:
        std::size_t vocab_size_;
        std::size_t d_model_;
        std::size_t seq_len_;

        Matrix token_emb_;
        Matrix grad_token_emb_;

        Matrix pos_emb_;
        Matrix grad_pos_emb_;

        std::vector<GPTBlock> blocks_;
        LayerNorm ln_f_;
        Linear lm_head_;

        std::vector<Matrix> stored_inputs_;
        std::vector<std::vector<std::size_t>> stored_tokens_;
        std::size_t batch_size_{0};

    public:
        GPTModel(std::size_t vocab_size, std::size_t d_model, std::size_t seq_len,
                 std::size_t num_heads, std::size_t d_ff, std::size_t num_layers)
            : vocab_size_(vocab_size), d_model_(d_model), seq_len_(seq_len),
              token_emb_(vocab_size, d_model),
              grad_token_emb_(vocab_size, d_model),
              pos_emb_(seq_len, d_model),
              grad_pos_emb_(seq_len, d_model),
              ln_f_(d_model),
              lm_head_(d_model, vocab_size)
        {
            constexpr double emb_init_std = 0.02;
            std::mt19937_64 rng{42};
            std::normal_distribution<double> dist(0.0, emb_init_std);
            for (std::size_t i = 0; i < token_emb_.size(); ++i)
                token_emb_.data()[i] = dist(rng);
            for (std::size_t i = 0; i < pos_emb_.size(); ++i)
                pos_emb_.data()[i] = dist(rng);

            for (std::size_t i = 0; i < num_layers; ++i)
                blocks_.emplace_back(d_model, num_heads, d_ff);
        }

        const char *name() const override { return "GPTModel"; }

        [[nodiscard]] std::size_t param_count() const noexcept override
        {
            std::size_t p = token_emb_.size() + pos_emb_.size();
            for (const auto &b : blocks_)
                p += b.param_count();
            p += ln_f_.param_count();
            p += lm_head_.param_count();
            return p;
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            std::vector<std::reference_wrapper<Matrix>> params;
            params.push_back(std::ref(token_emb_));
            params.push_back(std::ref(pos_emb_));
            for (auto &b : blocks_)
            {
                auto bp = b.parameters();
                params.insert(params.end(), bp.begin(), bp.end());
            }
            auto lp = ln_f_.parameters();
            params.insert(params.end(), lp.begin(), lp.end());
            auto hp = lm_head_.parameters();
            params.insert(params.end(), hp.begin(), hp.end());
            return params;
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            std::vector<std::reference_wrapper<Matrix>> grads;
            grads.push_back(std::ref(grad_token_emb_));
            grads.push_back(std::ref(grad_pos_emb_));
            for (auto &b : blocks_)
            {
                auto bg = b.param_gradients();
                grads.insert(grads.end(), bg.begin(), bg.end());
            }
            auto lg = ln_f_.param_gradients();
            grads.insert(grads.end(), lg.begin(), lg.end());
            auto hg = lm_head_.param_gradients();
            grads.insert(grads.end(), hg.begin(), hg.end());
            return grads;
        }

        Matrix forward(const Matrix &input) override
        {
            const std::size_t sl = input.rows();
            batch_size_ = input.cols();
            stored_inputs_.resize(batch_size_);
            stored_tokens_.resize(batch_size_);

            for (auto &v : stored_tokens_)
                v.clear();

            Matrix output(vocab_size_, sl * batch_size_);

            for (std::size_t b = 0; b < batch_size_; ++b)
            {
                stored_tokens_[b].reserve(sl);

                Matrix x(d_model_, sl);
                for (std::size_t t = 0; t < sl; ++t)
                {
                    if (t >= seq_len_)
                        throw std::invalid_argument("GPTModel forward: sequence length exceeds seq_len_");

                    auto token_id = static_cast<std::size_t>(input.at_unchecked(t, b));
                    if (token_id >= vocab_size_)
                        token_id = 0;
                    stored_tokens_[b].push_back(token_id);

                    for (std::size_t d = 0; d < d_model_; ++d)
                    {
                        double pe = pos_emb_.at_unchecked(t, d);
                        x.set_value_unchecked(d, t,
                            token_emb_.at_unchecked(token_id, d) + pe);
                    }
                }

                stored_inputs_[b] = x;

                for (std::size_t l = 0; l < blocks_.size(); ++l)
                    x = blocks_[l].forward(x);

                x = ln_f_.forward(x);
                Matrix logits = lm_head_.forward(x);

                for (std::size_t r = 0; r < vocab_size_; ++r)
                    for (std::size_t t = 0; t < sl; ++t)
                        output.set_value_unchecked(r, t * batch_size_ + b,
                            logits.at_unchecked(r, t));
            }

            return output;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            const std::size_t sl = grad_output.cols() / batch_size_;
            grad_token_emb_.zero();
            grad_pos_emb_.zero();

            Matrix grad_input(sl, batch_size_);
            grad_input.zero();

            auto all_grads = param_gradients();
            std::vector<Matrix> accum_grads;
            for (auto& g : all_grads) accum_grads.push_back(Matrix(g.get().rows(), g.get().cols(), 0.0));

            for (std::size_t b = 0; b < batch_size_; ++b)
            {
                Matrix grad_logits(vocab_size_, sl);
                for (std::size_t r = 0; r < vocab_size_; ++r)
                    for (std::size_t t = 0; t < sl; ++t)
                        grad_logits.set_value_unchecked(r, t,
                            grad_output.at_unchecked(r, t * batch_size_ + b));

                // Re-forward to rebuild cache for this sample
                Matrix x = stored_inputs_[b];
                for (std::size_t l = 0; l < blocks_.size(); ++l)
                    x = blocks_[l].forward(x);
                x = ln_f_.forward(x);
                lm_head_.forward(x); // rebuild lm_head_ cache

                for (auto& g : all_grads) g.get().zero();

                Matrix grad_ln = lm_head_.backward(grad_logits);
                grad_ln = ln_f_.backward(grad_ln);

                for (int l = static_cast<int>(blocks_.size()) - 1; l >= 0; --l)
                    grad_ln = blocks_[l].backward(grad_ln);

                const auto &tokens = stored_tokens_[b];
                for (std::size_t t = 0; t < sl; ++t)
                {
                    const std::size_t tid = tokens[t];
                    for (std::size_t d = 0; d < d_model_; ++d)
                    {
                        double g = grad_ln.at_unchecked(d, t);
                        grad_token_emb_.data()[tid * d_model_ + d] += g;
                        grad_pos_emb_.data()[t * d_model_ + d] += g;
                    }
                }
                
                for (std::size_t i = 0; i < all_grads.size(); ++i) {
                    auto& ag = accum_grads[i];
                    auto& cg = all_grads[i].get();
                    for (std::size_t j = 0; j < ag.size(); ++j) ag.data()[j] += cg.data()[j];
                }
            }
            
            for (std::size_t i = 0; i < all_grads.size(); ++i) {
                auto& ag = accum_grads[i];
                auto& cg = all_grads[i].get();
                for (std::size_t j = 0; j < ag.size(); ++j) cg.data()[j] = ag.data()[j];
            }

            return grad_input;
        }

        std::vector<std::size_t> generate(const std::vector<std::size_t> &prompt,
                                          std::size_t max_new_tokens,
                                          double temperature = 1.0)
        {
            std::vector<std::size_t> context(prompt);
            std::vector<std::size_t> generated;
            std::mt19937_64 rng{std::random_device{}()};
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            
            auto saved_inputs = stored_inputs_;
            auto saved_tokens = stored_tokens_;
            auto saved_bs = batch_size_;

            for (std::size_t step = 0; step < max_new_tokens; ++step)
            {
                std::size_t start = 0;
                if (context.size() > seq_len_)
                    start = context.size() - seq_len_;

                std::size_t cur_len = context.size() - start;
                Matrix input(cur_len, 1);
                for (std::size_t t = 0; t < cur_len; ++t)
                    input.set_value_unchecked(t, 0, static_cast<double>(context[start + t]));

                Matrix logits = forward(input);

                std::vector<double> last_logits(vocab_size_);
                for (std::size_t v = 0; v < vocab_size_; ++v)
                    last_logits[v] = logits.at_unchecked(v, cur_len - 1);

                if (temperature > 0.0 && temperature != 1.0)
                {
                    for (auto &v : last_logits)
                        v /= temperature;
                }

                double max_val = last_logits[0];
                for (std::size_t v = 1; v < vocab_size_; ++v)
                    max_val = std::max(max_val, last_logits[v]);
                
                double sum_exp = 0.0;
                for (auto &v : last_logits)
                {
                    v = std::exp(v - max_val);
                    sum_exp += v;
                }
                for (auto &v : last_logits)
                    v /= sum_exp;

                std::size_t next_token;
                if (temperature > 0.0 && temperature != 1.0)
                {
                    double r = dist(rng);
                    double cumulative = 0.0;
                    next_token = vocab_size_ - 1;
                    for (std::size_t v = 0; v < vocab_size_; ++v)
                    {
                        cumulative += last_logits[v];
                        if (r <= cumulative)
                        {
                            next_token = v;
                            break;
                        }
                    }
                }
                else
                {
                    next_token = 0;
                    double best = last_logits[0];
                    for (std::size_t v = 1; v < vocab_size_; ++v)
                    {
                        if (last_logits[v] > best)
                        {
                            best = last_logits[v];
                            next_token = v;
                        }
                    }
                }

                generated.push_back(next_token);
                context.push_back(next_token);
            }
            
            stored_inputs_ = saved_inputs;
            stored_tokens_ = saved_tokens;
            batch_size_ = saved_bs;

            return generated;
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
