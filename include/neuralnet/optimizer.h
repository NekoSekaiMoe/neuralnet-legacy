#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <execution>
#include <functional>
#include <vector>

#include <neuralnet/matrix.h>

namespace nn
{

    /**
     * @brief Abstract base class for optimizers.
     */
    class Optimizer
    {
    public:
        virtual ~Optimizer() = default;
        /**
         * @brief Performs a single optimization step (parameter update).
         */
        virtual void step() = 0;
        /**
         * @brief Zeros all parameter gradients.
         */
        virtual void zero_grad() = 0;
    };

    /**
     * @brief Stochastic Gradient Descent optimizer.
     *
     * Updates parameters using: param = param - learning_rate * grad.
     */
    class SGD : public Optimizer
    {
    private:
        double lr_;
        std::vector<std::reference_wrapper<Matrix>> params_;
        std::vector<std::reference_wrapper<Matrix>> grads_;

    public:
        SGD(std::vector<std::reference_wrapper<Matrix>> params,
            std::vector<std::reference_wrapper<Matrix>> grads,
            double lr)
            : lr_(lr), params_(std::move(params)), grads_(std::move(grads))
        {
            if (params_.size() != grads_.size())
            {
                throw std::invalid_argument("params and grads must have same size");
            }
        }

        void step() override
        {
            for (std::size_t i = 0; i < params_.size(); ++i)
            {
                auto &p = params_[i].get();
                auto &g = grads_[i].get();
                nn::transform(p.size(),
                               p.data().begin(), p.data().end(),
                               g.data().begin(),
                               p.data().begin(),
                               [this](double param, double grad)
                               { return param - lr_ * grad; });
            }
        }

        void zero_grad() override
        {
            for (auto &g_ref : grads_)
            {
                auto &g = g_ref.get();
                std::fill(g.data().begin(), g.data().end(), 0.0);
            }
        }
    };

    /**
     * @brief SGD with momentum optimizer.
     *
     * Maintains exponentially weighted moving average of gradients.
     * Updates: velocity = beta * velocity + (1 - beta) * grad,
     *          param = param - learning_rate * velocity.
     */
    class SGD_w_Momentum : public Optimizer
    {
    private:
        double lr_;
        double beta_;
        std::vector<std::reference_wrapper<Matrix>> params_;
        std::vector<std::reference_wrapper<Matrix>> grads_;
        std::vector<Matrix> velocity_;

    public:
        SGD_w_Momentum(std::vector<std::reference_wrapper<Matrix>> params,
                       std::vector<std::reference_wrapper<Matrix>> grads,
                       double lr, double beta = 0.9)
            : lr_(lr), beta_(beta), params_(std::move(params)), grads_(std::move(grads))
        {
            if (params_.size() != grads_.size())
            {
                throw std::invalid_argument("params and grads must have same size");
            }
            // 初始化速度向量为与参数同形的零矩阵
            velocity_.reserve(params_.size());
            for (const auto &p_ref : params_)
            {
                const Matrix &p = p_ref.get();
                velocity_.emplace_back(p.rows(), p.cols());
            }
        }

        void step() override
        {
            for (std::size_t i = 0; i < params_.size(); ++i)
            {
                auto &p = params_[i].get();
                auto &g = grads_[i].get();
                auto &v = velocity_[i];
                auto &p_vec = p.data();
                auto &g_vec = g.data();
                auto &v_vec = v.data();

                const std::size_t n = p_vec.size();
                nn::for_range(n, n,
                              [&](std::size_t idx)
                              {
                                  v_vec[idx] = beta_ * v_vec[idx] + (1 - beta_) * g_vec[idx];
                                  p_vec[idx] = p_vec[idx] - lr_ * v_vec[idx];
                              });
            }
        }

        void zero_grad() override
        {
            for (auto &g_ref : grads_)
            {
                auto &g = g_ref.get();
                std::fill(g.data().begin(), g.data().end(), 0.0);
            }
        }
    };

    // ── Adam 优化器 ────────────────────────────────────────────────────────────
    // Adam: A Method for Stochastic Optimization (Kingma & Ba, 2015)
    /**
     * @brief Adam optimizer with adaptive learning rates.
     *
     * Maintains first and second moment estimates with bias correction.
     * Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2015).
     */
    class Adam : public Optimizer
    {
    private:
        std::size_t step_count_;
        std::vector<Matrix> m_; // 一阶矩估计
        std::vector<Matrix> v_; // 二阶矩估计

    protected:
        double lr_;
        double beta1_;
        double beta2_;
        double eps_;
        std::vector<std::reference_wrapper<Matrix>> params_;
        std::vector<std::reference_wrapper<Matrix>> grads_;

    public:
        Adam(std::vector<std::reference_wrapper<Matrix>> params,
             std::vector<std::reference_wrapper<Matrix>> grads,
             double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
            : step_count_(0), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
              params_(std::move(params)), grads_(std::move(grads))
        {
            if (params_.size() != grads_.size())
            {
                throw std::invalid_argument("params and grads must have same size");
            }
            // 初始化一阶矩和二阶矩为零矩阵
            for (const auto &p_ref : params_)
            {
                const auto &p = p_ref.get();
                m_.emplace_back(p.rows(), p.cols());
                v_.emplace_back(p.rows(), p.cols());
            }
        }

        void step() override
        {
            ++step_count_;
            const double bias_correction1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
            const double bias_correction2 = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

            for (std::size_t i = 0; i < params_.size(); ++i)
            {
                auto &p = params_[i].get();
                auto &g = grads_[i].get();
                auto &m = m_[i];
                auto &v = v_[i];

                const std::size_t n = p.size();
                auto &p_vec = p.data();
                auto &g_vec = g.data();
                auto &m_vec = m.data();
                auto &v_vec = v.data();

                nn::for_range(n, n,
                              [&](std::size_t idx)
                              {
                                  // 更新一阶矩: m = beta1 * m + (1 - beta1) * g
                                  m_vec[idx] = beta1_ * m_vec[idx] + (1.0 - beta1_) * g_vec[idx];
                                  // 更新二阶矩: v = beta2 * v + (1 - beta2) * g^2
                                  v_vec[idx] = beta2_ * v_vec[idx] + (1.0 - beta2_) * g_vec[idx] * g_vec[idx];
                                  // 偏差校正
                                  const double m_hat = m_vec[idx] / bias_correction1;
                                  const double v_hat = v_vec[idx] / bias_correction2;
                                  // 更新参数
                                  p_vec[idx] = p_vec[idx] - lr_ * m_hat / (std::sqrt(v_hat) + eps_);
                              });
            }
        }

        void zero_grad() override
        {
            for (auto &g_ref : grads_)
            {
                auto &g = g_ref.get();
                std::fill(g.data().begin(), g.data().end(), 0.0);
            }
        }
    };

    // ── AdamW：解耦权重衰减（Decoupled Weight Decay，移植自上游）─────
    // 与 Adam + L2 正则化的区别：
    //   - L2:  g' = g + wd*p，用 g' 做 Adam 更新 → wd 受自适应学习率缩放
    //   - AdamW: 直接 p *= (1-lr*wd)，梯度更新不受 wd 影响
    //   → 衰减对所有参数等效，不因自适应学习率而被稀释；
    //     transformer/GPT 训练的标准配置。
    /**
     * @brief AdamW optimizer with decoupled weight decay.
     *
     * Unlike Adam with L2 regularization, weight decay is applied directly to parameters
     * (p *= (1 - lr*wd)) rather than being added to gradients. This prevents weight decay
     * from being diluted by adaptive learning rates, making it more effective.
     * Standard optimizer choice for training transformers and GPT models.
     */
    class AdamW : public Adam
    {
    private:
        double wd_; // 权重衰减系数

    public:
        AdamW(std::vector<std::reference_wrapper<Matrix>> params,
              std::vector<std::reference_wrapper<Matrix>> grads,
              double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999,
              double eps = 1e-8, double weight_decay = 0.01)
            : Adam(std::move(params), std::move(grads), lr, beta1, beta2, eps),
              wd_(weight_decay) {}

        void step() override
        {
            // 权重衰减解耦：先 p = (1-lr*wd)*p，再做标准 Adam 更新。
            // 零梯度时 m/v 保持 0，Adam 增量为 0 → p_t = p_0*(1-lr*wd)^t。
            if (wd_ != 0.0)
            {
                const double decay = 1.0 - lr_ * wd_;
                for (auto &p_ref : params_)
                    p_ref.get().scale_inplace(decay);
            }
            Adam::step();
        }
    };

} // namespace nn

#endif // OPTIMIZER_HPP
