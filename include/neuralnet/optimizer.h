#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <algorithm>
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
} // namespace nn

#endif // OPTIMIZER_HPP
