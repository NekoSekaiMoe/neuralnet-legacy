#ifndef LAYER_HPP
#define LAYER_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <execution>
#include <functional>
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

        Matrix forward(const Matrix &input) override
        {
            if (input.rows() != W_.cols())
            {
                throw std::invalid_argument("linear forward input shape mismatch");
            }

            input_cache_ = input;
            const Matrix product = W_ * input;
            Matrix result(product.rows(), product.cols());

            // 并行 bias 加法：使用 counting_iterator 生成索引
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(product.size()),
                          [&](std::size_t idx) noexcept
                          {
                              const std::size_t row = idx / product.cols();
                              result.data()[idx] = product.data()[idx] + b_.at_unchecked(row, 0);
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

            const std::size_t in_feat = W_.cols();
            const std::size_t out_feat = W_.rows();
            const std::size_t batch = grad_output.cols();

            // 计算 grad_input: dL/dx = W^T * dL/dy
            Matrix grad_input(in_feat, batch);
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(in_feat * batch),
                          [&, this](std::size_t idx) noexcept
                          {
                              const std::size_t input_feature = idx / batch;
                              const std::size_t batch_index = idx % batch;
                              double sum = 0.0;
                              for (std::size_t out_feature = 0; out_feature < out_feat; ++out_feature)
                              {
                                  sum += W_.at_unchecked(out_feature, input_feature) *
                                         grad_output.at_unchecked(out_feature, batch_index);
                              }
                              grad_input.set_value_unchecked(input_feature, batch_index, sum);
                          });

            // 计算 grad_W: dL/dW = dL/dy * x^T
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(out_feat * in_feat),
                          [&, this](std::size_t idx) noexcept
                          {
                              const std::size_t out_feature = idx / in_feat;
                              const std::size_t input_feature = idx % in_feat;
                              double sum = 0.0;
                              for (std::size_t batch_index = 0; batch_index < batch; ++batch_index)
                              {
                                  sum += grad_output.at_unchecked(out_feature, batch_index) *
                                         input_cache_.at_unchecked(input_feature, batch_index);
                              }
                              grad_W_.set_value_unchecked(out_feature, input_feature, 
                                                          grad_W_.at_unchecked(out_feature, input_feature) + sum);
                          });

            // 计算 grad_b: dL/db = sum(dL/dy, dim=batch)
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(out_feat),
                          [&, this](std::size_t out_feature) noexcept
                          {
                              double sum = 0.0;
                              for (std::size_t batch_index = 0; batch_index < batch; ++batch_index)
                              {
                                  sum += grad_output.at_unchecked(out_feature, batch_index);
                              }
                              grad_b_.set_value_unchecked(out_feature, 0, sum);
                          });

            return grad_input;
        }
    };

    class ReLU final : public Layer
    {
    private:
        Matrix input_cache_;

    public:
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

        void set_training(bool training) noexcept { training_ = training; }

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
}

#endif
