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

        /**
         * Updates each parameter using its corresponding gradient and learning rate.
         */
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

        /**
             * Clears all gradient matrices by setting their elements to zero.
             */
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

        /**
             * Clears all parameter gradients.
             */
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

        /**
             * Clears all parameter gradients by setting their elements to zero.
             */
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

    // ── Muon：MomentUm Orthogonalized by Newton-Schulz（移植自上游）─────
    // 算法（Keller Jordan et al., 2024，https://kellerjordan.github.io/posts/muon/）：
    //   1. SGD-Momentum: v = μ*v + g（Nesterov 时 update = g + μ*v）
    //   2. Newton-Schulz 五次迭代正交化：X ← (a + bA + cA²)X，A = XXᵀ（短边侧）
    //   3. p -= lr * 0.2 * sqrt(max(m,n)) * NS(update)   （谱范数≈1 的形状补偿）
    // 仅对二维参数（rows>1 且 cols>1）正交化；一维参数（bias/gain）退化为标准
    // SGD-Momentum 更新。实践建议：embedding/lm_head 用 AdamW，隐藏层用 Muon。
    /**
     * @brief Muon optimizer (momentum orthogonalized by Newton-Schulz iteration).
     *
     * For each 2D parameter: velocity accumulates momentum, the update is
     * orthogonalized via a quintic Newton-Schulz iteration (spectral norm ~ 1),
     * then scaled by 0.2 * sqrt(max(rows, cols)). 1D params (bias/gain) fall
     * back to plain SGD-Momentum updates.
     * Reference: Keller Jordan et al. 2024, https://kellerjordan.github.io/posts/muon/
     */
    class Muon final : public Optimizer
    {
    public:
        // 调优后的 quintic 多项式系数：φ^N(x) → 1 for x ∈ [0,1]
        static constexpr double kNsA = 3.4445;
        static constexpr double kNsB = -4.7750;
        static constexpr double kNsC = 2.0315;

    private:
        double lr_;
        double momentum_;
        bool nesterov_;
        std::size_t ns_steps_;
        double ns_eps_;
        std::vector<std::reference_wrapper<Matrix>> params_;
        std::vector<std::reference_wrapper<Matrix>> grads_;
        std::vector<Matrix> velocities_;

        /**
         * Newton-Schulz 正交化：返回谱范数≈1 的矩阵（原矩阵方向不变）。
         * 母矩阵沿短边构造（宽/方阵用 XXᵀ，高窄用 XᵀX），
         * 避免大参数（如词嵌入 50257×1024）在长边侧构造平方矩阵。
         */
        [[nodiscard]] static Matrix newton_schulz_(const Matrix &G,
                                                   std::size_t steps, double eps)
        {
            // 归一化：X = G / sqrt(||G||_F² + eps²)，使谱范数 ≤ 1（NS 收敛域）
            double norm_sq = 0.0;
            for (const double v : G.data())
                norm_sq += v * v;
            const double inv_norm = 1.0 / std::sqrt(norm_sq + eps * eps);

            const std::size_t m = G.rows();
            const std::size_t n = G.cols();
            Matrix X = G * inv_norm;

            if (m <= n)
            {
                // 宽/方阵：行正交化，A = X·Xᵀ（m×m，短边）
                // X ← (aI + bA + cA²)·X，驱动 X·Xᵀ → I
                for (std::size_t t = 0; t < steps; ++t)
                {
                    Matrix A = X.matmul_NT(X);   // (m, m)
                    Matrix A2 = A * A;
                    A.scale_inplace(kNsB);
                    A2.scale_inplace(kNsC);
                    A.add_inplace(A2);           // A = bA + cA²
                    Matrix BX = A * X;
                    X.scale_inplace(kNsA);       // X = aX + (bA+cA²)X
                    X.add_inplace(BX);
                }
            }
            else
            {
                // 高窄：列正交化，G = Xᵀ·X（n×n，短边）
                // X ← X·(aI + bG + cG²)，驱动 Xᵀ·X → I
                for (std::size_t t = 0; t < steps; ++t)
                {
                    Matrix Gr = X.matmul_TN(X);  // (n, n)
                    Matrix Gr2 = Gr * Gr;
                    Gr.scale_inplace(kNsB);
                    Gr2.scale_inplace(kNsC);
                    Gr.add_inplace(Gr2);         // Gr = bG + cG²
                    Matrix XM = X * Gr;
                    X.scale_inplace(kNsA);
                    X.add_inplace(XM);           // X = aX + X(bG+cG²)
                }
            }
            return X;
        }

    public:
        Muon(std::vector<std::reference_wrapper<Matrix>> params,
             std::vector<std::reference_wrapper<Matrix>> grads,
             double lr, double momentum = 0.95, bool nesterov = true,
             std::size_t ns_steps = 5, double ns_eps = 1e-7)
            : lr_(lr), momentum_(momentum), nesterov_(nesterov),
              ns_steps_(ns_steps), ns_eps_(ns_eps),
              params_(std::move(params)), grads_(std::move(grads))
        {
            if (params_.size() != grads_.size())
                throw std::invalid_argument("params and grads must have same size");
            velocities_.reserve(params_.size());
            for (const auto &p_ref : params_)
            {
                const Matrix &p = p_ref.get();
                velocities_.emplace_back(p.rows(), p.cols());
            }
        }

        void step() override
        {
            for (std::size_t i = 0; i < params_.size(); ++i)
            {
                Matrix &p = params_[i].get();
                const Matrix &g = grads_[i].get();
                Matrix &v = velocities_[i];

                // 1. 动量：v = μ*v + g；Nesterov：update = g + μ*v
                v.scale_inplace(momentum_);
                v.add_inplace(g);
                Matrix nesterov_buf;
                if (nesterov_)
                {
                    nesterov_buf = v * momentum_;
                    nesterov_buf.add_inplace(g);
                }
                const Matrix &update = nesterov_ ? nesterov_buf : v;

                if (p.rows() > 1 && p.cols() > 1)
                {
                    // 2+3. 正交化 + 形状补偿缩放（0.2*sqrt(max(m,n))）
                    Matrix ortho = newton_schulz_(update, ns_steps_, ns_eps_);
                    const std::size_t big = std::max(p.rows(), p.cols());
                    const double muon_scale = 0.2 * std::sqrt(static_cast<double>(big));
                    ortho.scale_inplace(-lr_ * muon_scale);
                    p.add_inplace(ortho);
                }
                else
                {
                    // 一维参数：标准 SGD-Momentum 更新（与上游一致，用 Nesterov 更新量）
                    for (std::size_t idx = 0; idx < p.size(); ++idx)
                        p.data()[idx] -= lr_ * update.data()[idx];
                }
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
