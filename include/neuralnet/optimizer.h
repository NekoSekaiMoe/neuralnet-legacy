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

    class Optimizer
    {
    public:
        virtual ~Optimizer() = default;
        virtual void step() = 0;
        virtual void zero_grad() = 0;
    };

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
                std::transform(NN_EXEC_POLICY,
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
                std::for_each(NN_EXEC_POLICY,
                              counting_iterator<std::size_t>(0),
                              counting_iterator<std::size_t>(n),
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
    class Adam : public Optimizer
    {
    private:
        double lr_;
        double beta1_;
        double beta2_;
        double eps_;
        std::size_t step_count_;
        std::vector<std::reference_wrapper<Matrix>> params_;
        std::vector<std::reference_wrapper<Matrix>> grads_;
        std::vector<Matrix> m_; // 一阶矩估计
        std::vector<Matrix> v_; // 二阶矩估计

    public:
        Adam(std::vector<std::reference_wrapper<Matrix>> params,
             std::vector<std::reference_wrapper<Matrix>> grads,
             double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
            : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), step_count_(0),
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

                std::for_each(NN_EXEC_POLICY,
                              counting_iterator<std::size_t>(0),
                              counting_iterator<std::size_t>(n),
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
} // namespace nn

#endif // OPTIMIZER_HPP
