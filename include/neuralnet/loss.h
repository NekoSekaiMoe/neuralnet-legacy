#ifndef LOSS_HPP
#define LOSS_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <neuralnet/nn/config.h>
#include <neuralnet/matrix.h>

namespace nn
{
    /**
     * @brief Abstract base class for loss functions.
     */
    class Loss
    {
    public:
        virtual ~Loss() = default;
        /**
         * @brief Computes the loss between predictions and targets.
         * @param pred Predicted values.
         * @param target Target (ground truth) values.
         * @return Loss value.
         */
        virtual double forward(const Matrix &pred, const Matrix &target) = 0;
        /**
         * @brief Returns gradient with respect to input.
         * @return Gradient matrix.
         */
        virtual const Matrix &backward() const = 0;
    };

    /**
     * @brief Mean Squared Error loss function.
     *
     * Computes: loss = mean((pred - target)^2).
     * Gradient: 2 * (pred - target) / n.
     */
    class MSELoss : public Loss
    {
    private:
        Matrix grad_input_;

    public:
        MSELoss() = default;

        [[nodiscard]] double forward(const Matrix &pred, const Matrix &target)
        {
            if (pred.rows() != target.rows() || pred.cols() != target.cols())
            {
                throw std::invalid_argument("mse loss shape mismatch");
            }
            if (pred.empty())
            {
                throw std::invalid_argument("mse loss cannot be computed on an empty matrix");
            }

            grad_input_ = Matrix(pred.rows(), pred.cols());
            const auto total = static_cast<double>(pred.size());

            const double sum_sq = nn::transform_reduce(
                pred.size(),
                pred.data().begin(), pred.data().end(),
                target.data().begin(),
                0.0,
                std::plus<>{},
                [](double prediction, double actual) noexcept
                {
                    const double diff = prediction - actual;
                    return diff * diff;
                });

            const double loss = sum_sq / total;
            const double factor = 2.0 / total;

            nn::transform(pred.size(),
                           pred.data().begin(), pred.data().end(),
                           target.data().begin(),
                           grad_input_.data().begin(),
                           [factor](double prediction, double actual) noexcept
                           {
                               return factor * (prediction - actual);
                           });

            return loss;
        }

        [[nodiscard]] const Matrix &backward() const noexcept { return grad_input_; }
    };

    /**
     * @brief Cross-entropy loss for multi-class classification.
     *
     * Computes numerically stable cross-entropy using log_softmax.
     * Expects one-hot encoded targets.
     */
    class CrossEntropyLoss : public Loss
    {
    private:
        Matrix grad_input_;

    public:
        CrossEntropyLoss() = default;

        // 数值稳定的 log_softmax 重写（对标上游 compute_loss.hpp）：
        //   log_softmax = (logits - col_max) - log(Σ exp(logits - col_max))
        // 1) 去除旧实现的 std::array<double,128> 栈缓冲（类数>128 时栈越界写，
        //    GPT 例程 vocab=8796 每 step 越界 ~69KB，UB）
        // 2) loss 直接由 log_softmax 计算：避免 log(softmax) 在极负 logits 下
        //    变 -inf、与 target=0 相乘得 0*(-inf)=NaN
        // 3) 梯度 = (softmax - target) / batch：与均值损失定义一致
        //    （旧实现漏掉 1/batch，与 SGD/动量/裁剪的尺度约定不一致）
        /**
         * @brief Computes cross-entropy loss with numerically stable log_softmax.
         *
         * Improvements over previous implementation:
         * - Uses dynamic allocation instead of fixed stack buffer (avoids overflow for large vocab)
         * - Computes loss directly from log_softmax (prevents NaN from 0 * -inf)
         * - Gradient scaled by 1/batch for consistency with mean loss definition
         *
         * @param logits Raw model outputs (classes, batch).
         * @param target_onehot One-hot encoded targets (classes, batch).
         * @return Mean cross-entropy loss over the batch.
         */
        [[nodiscard]] double forward(const Matrix &logits, const Matrix &target_onehot)
        {
            const std::size_t classes = logits.rows();
            const std::size_t batch = logits.cols();
            if (target_onehot.rows() != classes || target_onehot.cols() != batch)
                throw std::invalid_argument("cross entropy loss: target shape mismatch");
            grad_input_ = Matrix(classes, batch);

            // 列（样本）间独立的 scratch：shifted 矩阵与每列 loss，写入互不相交
            Matrix shifted(classes, batch);
            std::vector<double> loss_cols(batch, 0.0);

            nn::for_range(classes * batch, batch, [&](std::size_t i)
            {
                // 1) 列最大值（数值稳定性）
                double max_val = logits.at_unchecked(0, i);
                for (std::size_t c = 1; c < classes; ++c)
                {
                    const double val = logits.at_unchecked(c, i);
                    if (val > max_val) max_val = val;
                }

                // 2) shifted = logits - col_max；同步累加 Σexp
                double sum_exp = 0.0;
                for (std::size_t c = 0; c < classes; ++c)
                {
                    const double s = logits.at_unchecked(c, i) - max_val;
                    shifted.set_value_unchecked(c, i, s);
                    sum_exp += std::exp(s);
                }
                const double log_denom = std::log(sum_exp); // sum_exp ≥ 1，必有限

                // 3) 每类：loss -= target·log_softmax；grad = (softmax-target)/batch
                double col_loss = 0.0;
                for (std::size_t c = 0; c < classes; ++c)
                {
                    const double log_sm = shifted.at_unchecked(c, i) - log_denom;
                    const double t = target_onehot.at_unchecked(c, i);
                    col_loss -= t * log_sm;
                    const double softmax_c = std::exp(shifted.at_unchecked(c, i)) / sum_exp;
                    grad_input_.set_value_unchecked(c, i, softmax_c - t);
                }
                loss_cols[i] = col_loss;
            });

            const double inv_batch = 1.0 / static_cast<double>(batch);
            double total_loss = 0.0;
            for (double l : loss_cols)
                total_loss += l;
            grad_input_.scale_inplace(inv_batch);

            return total_loss * inv_batch;
        }

        [[nodiscard]] const Matrix &backward() const noexcept { return grad_input_; }
    };

} // namespace nn

#endif // LOSS_HPP