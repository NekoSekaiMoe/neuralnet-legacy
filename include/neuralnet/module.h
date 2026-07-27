#ifndef NN_MODULE_H
#define NN_MODULE_H

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cmath>
#include <random>
#include <neuralnet/tensor.h>

namespace nn
{
    // The base class for all Tensor-based autograd modules
    class Module
    {
    public:
        virtual ~Module() = default;

        // Returns all trainable parameters of this module and its submodules
        virtual std::vector<Tensor> parameters() { return {}; }
        
        virtual std::size_t param_count() const noexcept { return 0; }
        virtual const char *name() const = 0;

        virtual Tensor forward(const Tensor &input) = 0;
        
        virtual void on_mode_change(bool training) {}
        virtual bool has_state() const { return false; }
    };

    // Sequential container for Modules
    class Sequential : public Module
    {
    private:
        std::vector<std::unique_ptr<Module>> modules_;
        bool is_training_{true};

    public:
        Sequential() = default;

        template <typename ModuleType, typename... Args>
        Sequential &add(Args &&...args)
        {
            modules_.emplace_back(std::make_unique<ModuleType>(std::forward<Args>(args)...));
            return *this;
        }

        const char *name() const override { return "Sequential"; }

        std::vector<Tensor> parameters() override
        {
            std::vector<Tensor> result;
            for (auto &m : modules_)
            {
                auto p = m->parameters();
                result.insert(result.end(), p.begin(), p.end());
            }
            return result;
        }

        std::size_t param_count() const noexcept override
        {
            std::size_t count = 0;
            for (auto &m : modules_) count += m->param_count();
            return count;
        }

        void on_mode_change(bool training) override
        {
            is_training_ = training;
            for (auto &m : modules_) m->on_mode_change(training);
        }

        Tensor forward(const Tensor &input) override
        {
            if (modules_.empty()) throw std::runtime_error("Sequential is empty");
            Tensor out = modules_.front()->forward(input);
            for (std::size_t i = 1; i < modules_.size(); ++i)
            {
                out = modules_[i]->forward(out);
            }
            return out;
        }
    };

    // --- Basic Modules ---

    class LinearModule : public Module
    {
    private:
        std::size_t in_features_;
        std::size_t out_features_;
        bool bias_;
        Tensor weight_;
        Tensor b_;
        inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

    public:
        LinearModule(std::size_t in_features, std::size_t out_features, bool bias = true)
            : in_features_(in_features), out_features_(out_features), bias_(bias)
        {
            double limit = std::sqrt(6.0 / static_cast<double>(in_features + out_features));
            std::uniform_real_distribution<double> dist(-limit, limit);
            
            Matrix w_mat(out_features, in_features);
            for(auto& val : w_mat.data()) val = dist(rng_);
            weight_ = Tensor(w_mat, true);

            if (bias_) {
                b_ = Tensor(Matrix(out_features, 1, 0.0), true);
            } else {
                b_ = Tensor();
            }
        }

        const char *name() const override { return "LinearModule"; }

        std::size_t param_count() const noexcept override {
            return in_features_ * out_features_ + (bias_ ? out_features_ : 0);
        }

        std::vector<Tensor> parameters() override {
            if (bias_) return {weight_, b_};
            return {weight_};
        }

        Tensor forward(const Tensor &input) override {
            return linear(input, weight_, b_);
        }
    };

    class ReLUModule : public Module {
    public:
        const char *name() const override { return "ReLUModule"; }
        Tensor forward(const Tensor &input) override { return relu(input); }
    };

    class SigmoidModule : public Module {
    public:
        const char *name() const override { return "SigmoidModule"; }
        Tensor forward(const Tensor &input) override { return sigmoid(input); }
    };

    class TanhModule : public Module {
    public:
        const char *name() const override { return "TanhModule"; }
        Tensor forward(const Tensor &input) override { return tanh(input); }
    };

    class LeakyReLUModule : public Module {
    private:
        double negative_slope_;
    public:
        explicit LeakyReLUModule(double negative_slope = 0.01) : negative_slope_(negative_slope) {}
        const char *name() const override { return "LeakyReLUModule"; }
        Tensor forward(const Tensor &input) override { return leaky_relu(input, negative_slope_); }
    };

    class GELUModule : public Module {
    public:
        const char *name() const override { return "GELUModule"; }
        Tensor forward(const Tensor &input) override { return gelu(input); }
    };

    class DropoutModule : public Module {
    private:
        double p_;
        bool is_training_{true};
        inline static thread_local std::mt19937_64 rng_{std::random_device{}()};
    public:
        explicit DropoutModule(double p = 0.5) : p_(p) {}
        const char *name() const override { return "DropoutModule"; }
        void on_mode_change(bool training) override { is_training_ = training; }
        
        Tensor forward(const Tensor &input) override {
            if (!is_training_ || p_ == 0.0) return input;
            
            bool req_grad = input.requires_grad();
            Matrix out_data(input.rows(), input.cols());
            Matrix mask(input.rows(), input.cols());
            const double scale = 1.0 / (1.0 - p_);
            std::bernoulli_distribution dist(1.0 - p_);
            
            auto& out_span = out_data.data();
            auto& mask_span = mask.data();
            const auto& in_span = input.data().data();
            for (std::size_t i = 0; i < out_span.size(); ++i) {
                mask_span[i] = dist(rng_) ? scale : 0.0;
                out_span[i] = in_span[i] * mask_span[i];
            }
            
            auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
            if (req_grad) {
                out_node->children.push_back(input.node());
                auto in_node = input.node();
                out_node->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out_node),
                                         in_node_w = std::weak_ptr<TensorNode>(in_node),
                                         mask]() {
                    auto out = out_node_w.lock();
                    auto in = in_node_w.lock();
                    if (!out || !out->grad || !in || !in->requires_grad) return;
                    Matrix a_grad = *(out->grad);
                    auto& ag_span = a_grad.data();
                    const auto& m_span = mask.data();
                    for (std::size_t i = 0; i < a_grad.size(); ++i) {
                        ag_span[i] *= m_span[i];
                    }
                    in->accumulate_grad(a_grad);
                };
            }
            return Tensor(out_node);
        }
    };

    class BatchNorm1dModule : public Module {
    private:
        std::size_t num_features_;
        double eps_;
        double momentum_;
        bool affine_;
        bool is_training_{true};

        Tensor weight_; // gamma
        Tensor bias_;   // beta
        Matrix running_mean_;
        Matrix running_var_;
        int64_t num_batches_tracked_{0};

    public:
        BatchNorm1dModule(std::size_t num_features, double eps = 1e-5, double momentum = 0.1, bool affine = true)
            : num_features_(num_features), eps_(eps), momentum_(momentum), affine_(affine),
              running_mean_(num_features, 1, 0.0), running_var_(num_features, 1, 1.0)
        {
            if (num_features_ == 0) throw std::invalid_argument("BatchNorm1dModule: num_features must be > 0");
            if (affine_) {
                weight_ = Tensor(Matrix(num_features, 1, 1.0), true);
                bias_ = Tensor(Matrix(num_features, 1, 0.0), true);
            }
        }

        const char *name() const override { return "BatchNorm1dModule"; }
        void on_mode_change(bool training) override { is_training_ = training; }
        bool has_state() const override { return true; }
        std::size_t param_count() const noexcept override { return affine_ ? 2 * num_features_ : 0; }
        std::vector<Tensor> parameters() override {
            if (affine_) return {weight_, bias_};
            return {};
        }

        Tensor forward(const Tensor &input) override {
            if (input.rows() != num_features_) throw std::invalid_argument("BatchNorm1dModule forward: input rows != num_features");
            const std::size_t batch_size = input.cols();
            if (batch_size == 0) return Tensor(Matrix(num_features_, 0), input.requires_grad());

            Matrix batch_mean(num_features_, 1, 0.0);
            Matrix batch_var(num_features_, 1, 0.0);

            if (is_training_) {
                for (std::size_t i = 0; i < num_features_; ++i) {
                    double s = 0.0;
                    for (std::size_t j = 0; j < batch_size; ++j) s += input.data().at_unchecked(i, j);
                    double mu = s / batch_size;
                    batch_mean.set_value_unchecked(i, 0, mu);

                    double var_s = 0.0;
                    for (std::size_t j = 0; j < batch_size; ++j) {
                        double d = input.data().at_unchecked(i, j) - mu;
                        var_s += d * d;
                    }
                    batch_var.set_value_unchecked(i, 0, var_s / batch_size);
                }

                auto& rm = running_mean_.data();
                auto& rv = running_var_.data();
                const auto& bm = batch_mean.data();
                const auto& bv = batch_var.data();
                for (std::size_t i = 0; i < num_features_; ++i) {
                    rm[i] = (1.0 - momentum_) * rm[i] + momentum_ * bm[i];
                    rv[i] = (1.0 - momentum_) * rv[i] + momentum_ * bv[i];
                }
                ++num_batches_tracked_;
            }

            const Matrix& mean_src = is_training_ ? batch_mean : running_mean_;
            const Matrix& var_src = is_training_ ? batch_var : running_var_;
            
            Matrix normalized(num_features_, batch_size);
            for (std::size_t i = 0; i < num_features_; ++i) {
                double mu = mean_src.at_unchecked(i, 0);
                double std_inv = 1.0 / std::sqrt(var_src.at_unchecked(i, 0) + eps_);
                for (std::size_t j = 0; j < batch_size; ++j) {
                    double val = (input.data().at_unchecked(i, j) - mu) * std_inv;
                    normalized.set_value_unchecked(i, j, val);
                }
            }

            Tensor normalized_tensor(normalized, input.requires_grad());
            if (input.requires_grad()) {
                normalized_tensor.node()->children.push_back(input.node());
                auto in_node = input.node();
                if (is_training_) {
                    normalized_tensor.node()->backward_op = [out_w = std::weak_ptr<TensorNode>(normalized_tensor.node()),
                                                             in_w = std::weak_ptr<TensorNode>(in_node),
                                                             batch_var, eps = eps_, num_features = num_features_, batch_size,
                                                             normalized]() {
                        auto out = out_w.lock();
                        auto in = in_w.lock();
                        if (!out || !out->grad || !in || !in->requires_grad) return;
                        Matrix a_grad = *(out->grad);
                        Matrix dx_hat(num_features, batch_size);
                        for (std::size_t i = 0; i < num_features; ++i) {
                            for (std::size_t j = 0; j < batch_size; ++j) {
                                dx_hat.set_value_unchecked(i, j, a_grad.at_unchecked(i, j));
                            }
                        }
                        std::vector<double> sum_dxhat(num_features, 0.0);
                        std::vector<double> sum_dxhat_xhat(num_features, 0.0);
                        std::vector<double> std_inv(num_features, 0.0);
                        for (std::size_t i = 0; i < num_features; ++i) {
                            double sd = 0.0;
                            double sdx = 0.0;
                            for (std::size_t j = 0; j < batch_size; ++j) {
                                sd += dx_hat.at_unchecked(i, j);
                                sdx += dx_hat.at_unchecked(i, j) * normalized.at_unchecked(i, j);
                            }
                            sum_dxhat[i] = sd;
                            sum_dxhat_xhat[i] = sdx;
                            std_inv[i] = 1.0 / std::sqrt(batch_var.at_unchecked(i, 0) + eps);
                        }
                        const double scale = 1.0 / static_cast<double>(batch_size);
                        const double N = static_cast<double>(batch_size);
                        Matrix dx(num_features, batch_size);
                        for (std::size_t i = 0; i < num_features; ++i) {
                            for (std::size_t j = 0; j < batch_size; ++j) {
                                dx.set_value_unchecked(i, j, scale * (N * dx_hat.at_unchecked(i, j) - sum_dxhat[i] - normalized.at_unchecked(i, j) * sum_dxhat_xhat[i]) * std_inv[i]);
                            }
                        }
                        in->accumulate_grad(dx);
                    };
                } else {
                    normalized_tensor.node()->backward_op = [out_w = std::weak_ptr<TensorNode>(normalized_tensor.node()),
                                                             in_w = std::weak_ptr<TensorNode>(in_node),
                                                             running_var = running_var_, eps = eps_, num_features = num_features_, batch_size]() {
                        auto out = out_w.lock();
                        auto in = in_w.lock();
                        if (!out || !out->grad || !in || !in->requires_grad) return;
                        Matrix a_grad = *(out->grad);
                        Matrix dx(num_features, batch_size);
                        for (std::size_t i = 0; i < num_features; ++i) {
                            double std_inv = 1.0 / std::sqrt(running_var.at_unchecked(i, 0) + eps);
                            for (std::size_t j = 0; j < batch_size; ++j) {
                                double g = a_grad.at_unchecked(i, j);
                                dx.set_value_unchecked(i, j, g * std_inv);
                            }
                        }
                        in->accumulate_grad(dx);
                    };
                }
            }

            if (!affine_) return normalized_tensor;

            // Broadcasted multiply and add: out = normalized * gamma + beta
            // We can just use our existing tensor ops if we reshape or write a manual broadcast
            // For now, let's just do it manually for efficiency and correct backprop.
            Matrix out_data(num_features_, batch_size);
            for (std::size_t i = 0; i < num_features_; ++i) {
                double g = weight_.data().at_unchecked(i, 0);
                double b = bias_.data().at_unchecked(i, 0);
                for (std::size_t j = 0; j < batch_size; ++j) {
                    out_data.set_value_unchecked(i, j, normalized.at_unchecked(i, j) * g + b);
                }
            }

            bool req_grad = normalized_tensor.requires_grad() || weight_.requires_grad() || bias_.requires_grad();
            auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
            if (req_grad) {
                out_node->children.push_back(normalized_tensor.node());
                if (weight_.requires_grad()) out_node->children.push_back(weight_.node());
                if (bias_.requires_grad()) out_node->children.push_back(bias_.node());
                
                auto norm_node = normalized_tensor.node();
                auto w_node = weight_.node();
                auto b_node = bias_.node();
                
                out_node->backward_op = [out_w = std::weak_ptr<TensorNode>(out_node),
                                         norm_w = std::weak_ptr<TensorNode>(norm_node),
                                         w_w = std::weak_ptr<TensorNode>(w_node),
                                         b_w = std::weak_ptr<TensorNode>(b_node),
                                         normalized, num_features = num_features_, batch_size]() {
                    auto out = out_w.lock();
                    auto norm = norm_w.lock();
                    auto w = w_w.lock();
                    auto b = b_w.lock();
                    if (!out || !out->grad) return;
                    Matrix a_grad = *(out->grad);
                    
                    if (w && w->requires_grad) {
                        Matrix dw(num_features, 1, 0.0);
                        for (std::size_t i = 0; i < num_features; ++i) {
                            double s = 0.0;
                            for (std::size_t j = 0; j < batch_size; ++j) {
                                s += a_grad.at_unchecked(i, j) * normalized.at_unchecked(i, j);
                            }
                            dw.set_value_unchecked(i, 0, s);
                        }
                        w->accumulate_grad(dw);
                    }
                    if (b && b->requires_grad) {
                        Matrix db(num_features, 1, 0.0);
                        for (std::size_t i = 0; i < num_features; ++i) {
                            double s = 0.0;
                            for (std::size_t j = 0; j < batch_size; ++j) {
                                s += a_grad.at_unchecked(i, j);
                            }
                            db.set_value_unchecked(i, 0, s);
                        }
                        b->accumulate_grad(db);
                    }
                    if (norm && norm->requires_grad) {
                        Matrix dnorm(num_features, batch_size);
                        for (std::size_t i = 0; i < num_features; ++i) {
                            double gamma = w ? w->data.at_unchecked(i, 0) : 1.0;
                            for (std::size_t j = 0; j < batch_size; ++j) {
                                dnorm.set_value_unchecked(i, j, a_grad.at_unchecked(i, j) * gamma);
                            }
                        }
                        norm->accumulate_grad(dnorm);
                    }
                };
            }
            return Tensor(out_node);
        }
    };

    // ── LayerNormModule ────────────────────────────────────────────────────────
    class LayerNormModule : public Module
    {
    private:
        std::size_t normalized_shape_;
        double eps_;
        Tensor gamma_;
        Tensor beta_;

    public:
        LayerNormModule(std::size_t normalized_shape, double eps = 1e-5)
            : normalized_shape_(normalized_shape), eps_(eps),
              gamma_(Tensor(Matrix(normalized_shape, 1, 1.0), true)),
              beta_(Tensor(Matrix(normalized_shape, 1, 0.0), true))
        {
        }

        const char *name() const override { return "LayerNormModule"; }
        std::size_t param_count() const noexcept override { return 2 * normalized_shape_; }
        bool has_state() const override { return false; }

        std::vector<Tensor> parameters() override
        {
            return {gamma_, beta_};
        }

        Tensor forward(const Tensor &input) override
        {
            const Matrix &in_m = input.data();
            if (in_m.rows() != normalized_shape_)
                throw std::invalid_argument("LayerNormModule forward: input rows != normalized_shape");

            const std::size_t feat = in_m.rows();
            const std::size_t batch = in_m.cols();
            const double N = static_cast<double>(feat);

            std::vector<double> inv_std_cache(batch);
            Matrix norm_m(feat, batch);

            for (std::size_t j = 0; j < batch; ++j)
            {
                double sum = 0.0;
                for (std::size_t i = 0; i < feat; ++i)
                    sum += in_m.at_unchecked(i, j);
                double mu = sum / N;

                double var_sum = 0.0;
                for (std::size_t i = 0; i < feat; ++i)
                {
                    double d = in_m.at_unchecked(i, j) - mu;
                    var_sum += d * d;
                }
                double inv_std = 1.0 / std::sqrt(var_sum / N + eps_);
                inv_std_cache[j] = inv_std;

                for (std::size_t i = 0; i < feat; ++i)
                    norm_m.set_value_unchecked(i, j, (in_m.at_unchecked(i, j) - mu) * inv_std);
            }

            Matrix out_m(feat, batch);
            const Matrix &g = gamma_.data();
            const Matrix &b = beta_.data();
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(feat * batch),
                          [&](std::size_t idx) {
                              std::size_t i = idx / batch;
                              out_m.data()[idx] = norm_m.data()[idx] * g.at_unchecked(i, 0) + b.at_unchecked(i, 0);
                          });

            bool req_grad = input.requires_grad() || gamma_.requires_grad() || beta_.requires_grad();
            Tensor out(out_m, req_grad);
            
            if (req_grad)
            {
                out.node()->children.push_back(input.node());
                out.node()->children.push_back(gamma_.node());
                out.node()->children.push_back(beta_.node());
                
                auto inv_std_cache_ptr = std::make_shared<std::vector<double>>(std::move(inv_std_cache));
                auto norm_m_ptr = std::make_shared<Matrix>(std::move(norm_m));
                
                out.node()->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out.node()),
                                           in_node_w = std::weak_ptr<TensorNode>(input.node()),
                                           gamma_node_w = std::weak_ptr<TensorNode>(gamma_.node()),
                                           beta_node_w = std::weak_ptr<TensorNode>(beta_.node()),
                                           inv_std_cache_ptr, norm_m_ptr, feat, batch, N]() {
                    auto out = out_node_w.lock();
                    if (!out || !out->grad) return;
                    auto in = in_node_w.lock();
                    auto gamma_node = gamma_node_w.lock();
                    auto beta_node = beta_node_w.lock();

                    const Matrix &grad_output = *(out->grad);

                    if ((gamma_node && gamma_node->requires_grad) || (beta_node && beta_node->requires_grad))
                    {
                        Matrix dgamma(feat, 1, 0.0);
                        Matrix dbeta(feat, 1, 0.0);
                        for (std::size_t i = 0; i < feat; ++i)
                        {
                            double dg = 0.0, db = 0.0;
                            for (std::size_t j = 0; j < batch; ++j)
                            {
                                dg += grad_output.at_unchecked(i, j) * norm_m_ptr->at_unchecked(i, j);
                                db += grad_output.at_unchecked(i, j);
                            }
                            dgamma.set_value_unchecked(i, 0, dg);
                            dbeta.set_value_unchecked(i, 0, db);
                        }
                        if (gamma_node && gamma_node->requires_grad) gamma_node->accumulate_grad(dgamma);
                        if (beta_node && beta_node->requires_grad) beta_node->accumulate_grad(dbeta);
                    }

                    if (in && in->requires_grad)
                    {
                        Matrix grad_input(feat, batch);
                        const Matrix &g = gamma_node->data;
                        for (std::size_t j = 0; j < batch; ++j)
                        {
                            double inv_std = (*inv_std_cache_ptr)[j];
                            double sum_dxhat = 0.0, sum_dxhat_xhat = 0.0;
                            for (std::size_t i = 0; i < feat; ++i)
                            {
                                double dxhat = grad_output.at_unchecked(i, j) * g.at_unchecked(i, 0);
                                sum_dxhat += dxhat;
                                sum_dxhat_xhat += dxhat * norm_m_ptr->at_unchecked(i, j);
                            }
                            for (std::size_t i = 0; i < feat; ++i)
                            {
                                double dxhat = grad_output.at_unchecked(i, j) * g.at_unchecked(i, 0);
                                grad_input.set_value_unchecked(i, j,
                                    inv_std / N * (N * dxhat - sum_dxhat
                                        - norm_m_ptr->at_unchecked(i, j) * sum_dxhat_xhat));
                            }
                        }
                        in->accumulate_grad(grad_input);
                    }
                };
            }
            return out;
        }
    };

    // ── SoftmaxModule ────────────────────────────────────────────────────────
    class SoftmaxModule : public Module
    {
    public:
        const char *name() const override { return "SoftmaxModule"; }

        Tensor forward(const Tensor &input) override
        {
            const Matrix &in_m = input.data();
            const std::size_t rows = in_m.rows();
            const std::size_t cols = in_m.cols();
            if (rows == 0)
                throw std::invalid_argument("SoftmaxModule forward: input must have at least one row");
                
            Matrix out_m(rows, cols);

            for (std::size_t j = 0; j < cols; ++j)
            {
                double max_val = in_m.at_unchecked(0, j);
                for (std::size_t i = 1; i < rows; ++i)
                    if (in_m.at_unchecked(i, j) > max_val)
                        max_val = in_m.at_unchecked(i, j);

                double sum_exp = 0.0;
                for (std::size_t i = 0; i < rows; ++i)
                {
                    double e = std::exp(in_m.at_unchecked(i, j) - max_val);
                    out_m.set_value_unchecked(i, j, e);
                    sum_exp += e;
                }
                for (std::size_t i = 0; i < rows; ++i)
                    out_m.set_value_unchecked(i, j,
                        out_m.at_unchecked(i, j) / sum_exp);
            }
            
            bool req_grad = input.requires_grad();
            Tensor out(out_m, req_grad);
            
            if (req_grad)
            {
                out.node()->children.push_back(input.node());
                auto out_m_ptr = std::make_shared<Matrix>(std::move(out_m));
                
                out.node()->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out.node()),
                                           in_node_w = std::weak_ptr<TensorNode>(input.node()),
                                           out_m_ptr, rows, cols]() {
                    auto out = out_node_w.lock();
                    if (!out || !out->grad) return;
                    auto in = in_node_w.lock();
                    if (!in || !in->requires_grad) return;
                    
                    const Matrix &grad_output = *(out->grad);
                    Matrix grad_input(rows, cols);
                    for (std::size_t j = 0; j < cols; ++j)
                    {
                        double dot = 0.0;
                        for (std::size_t i = 0; i < rows; ++i)
                            dot += grad_output.at_unchecked(i, j) * out_m_ptr->at_unchecked(i, j);
                        for (std::size_t i = 0; i < rows; ++i)
                            grad_input.set_value_unchecked(i, j,
                                out_m_ptr->at_unchecked(i, j)
                                * (grad_output.at_unchecked(i, j) - dot));
                    }
                    in->accumulate_grad(grad_input);
                };
            }
            return out;
        }
    };

    // ── PositionalEncodingModule ─────────────────────────────────────────────
    class PositionalEncodingModule : public Module
    {
    private:
        Matrix pe_;
        std::size_t d_model_;
        std::size_t max_seq_len_;

    public:
        PositionalEncodingModule(std::size_t d_model, std::size_t max_seq_len)
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

        const char *name() const override { return "PositionalEncodingModule"; }

        Tensor forward(const Tensor &input) override
        {
            const Matrix &in_m = input.data();
            if (in_m.rows() != d_model_)
                throw std::invalid_argument("PositionalEncodingModule forward: input rows != d_model");
            if (in_m.cols() > max_seq_len_)
                throw std::invalid_argument("PositionalEncodingModule forward: seq length > max_seq_len");

            Matrix out_m = in_m;
            for (std::size_t j = 0; j < in_m.cols(); ++j)
                for (std::size_t i = 0; i < d_model_; ++i)
                    out_m.data()[i * in_m.cols() + j] += pe_.at_unchecked(i, j);

            bool req_grad = input.requires_grad();
            Tensor out(out_m, req_grad);
            
            if (req_grad)
            {
                out.node()->children.push_back(input.node());
                out.node()->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out.node()),
                                           in_node_w = std::weak_ptr<TensorNode>(input.node())]() {
                    auto out = out_node_w.lock();
                    if (!out || !out->grad) return;
                    auto in = in_node_w.lock();
                    if (!in || !in->requires_grad) return;
                    
                    in->accumulate_grad(*(out->grad));
                };
            }
            return out;
        }
    };

    // ── MultiHeadAttentionModule ──────────────────────────────────────────────
    class MultiHeadAttentionModule : public Module
    {
    private:
        std::size_t d_model_;
        std::size_t num_heads_;
        std::size_t head_dim_;
        double scale_;
        bool causal_;

        Tensor W_q_, W_k_, W_v_, W_o_;

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
        MultiHeadAttentionModule(std::size_t d_model, std::size_t num_heads, bool causal = false)
            : d_model_(d_model), num_heads_(num_heads),
              head_dim_(validated_head_dim(d_model, num_heads)),
              scale_(1.0 / std::sqrt(static_cast<double>(head_dim_))),
              causal_(causal),
              W_q_(Tensor(xavier_init(d_model, d_model), true)),
              W_k_(Tensor(xavier_init(d_model, d_model), true)),
              W_v_(Tensor(xavier_init(d_model, d_model), true)),
              W_o_(Tensor(xavier_init(d_model, d_model), true))
        {}

        const char *name() const override { return "MultiHeadAttentionModule"; }
        [[nodiscard]] std::size_t param_count() const noexcept override
        {
            return 4 * d_model_ * d_model_;
        }

        std::vector<Tensor> parameters() override
        {
            return {W_q_, W_k_, W_v_, W_o_};
        }

        Tensor forward(const Tensor &input) override
        {
            const Matrix &in_m = input.data();
            const std::size_t sl = in_m.cols();

            const Matrix &wq = W_q_.data();
            const Matrix &wk = W_k_.data();
            const Matrix &wv = W_v_.data();
            const Matrix &wo = W_o_.data();

            Matrix Q_cache = wq * in_m;
            Matrix K_cache = wk * in_m;
            Matrix V_cache = wv * in_m;

            Matrix concat_cache(d_model_, sl);
            std::vector<Matrix> attn_cache(num_heads_);

            for (std::size_t h = 0; h < num_heads_; ++h)
            {
                const std::size_t off = h * head_dim_;
                Matrix Q_h = Q_cache.row_slice(off, head_dim_);
                Matrix K_h = K_cache.row_slice(off, head_dim_);
                Matrix V_h = V_cache.row_slice(off, head_dim_);

                Matrix scores = Q_h.matmul_TN(K_h);
                scores.scale_inplace(scale_);
                if (causal_)
                {
                    for (std::size_t i = 0; i < sl; ++i)
                        for (std::size_t j = i + 1; j < sl; ++j)
                            scores.set_value_unchecked(i, j, -1e9);
                }
                softmax_rows_inplace(scores);
                attn_cache[h] = scores;

                Matrix head_out = V_h.matmul_NT(scores);
                concat_cache.set_row_slice(off, head_out);
            }

            Matrix out_m = wo * concat_cache;

            bool req_grad = input.requires_grad() || W_q_.requires_grad() || W_k_.requires_grad() || W_v_.requires_grad() || W_o_.requires_grad();
            Tensor out(out_m, req_grad);

            if (req_grad)
            {
                out.node()->children.push_back(input.node());
                out.node()->children.push_back(W_q_.node());
                out.node()->children.push_back(W_k_.node());
                out.node()->children.push_back(W_v_.node());
                out.node()->children.push_back(W_o_.node());

                auto attn_cache_ptr = std::make_shared<std::vector<Matrix>>(std::move(attn_cache));
                auto Q_cache_ptr = std::make_shared<Matrix>(std::move(Q_cache));
                auto K_cache_ptr = std::make_shared<Matrix>(std::move(K_cache));
                auto V_cache_ptr = std::make_shared<Matrix>(std::move(V_cache));
                auto concat_cache_ptr = std::make_shared<Matrix>(std::move(concat_cache));
                
                std::size_t d_model = d_model_;
                std::size_t num_heads = num_heads_;
                std::size_t head_dim = head_dim_;
                double scale = scale_;
                bool causal = causal_;

                out.node()->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out.node()),
                                           in_node_w = std::weak_ptr<TensorNode>(input.node()),
                                           wq_node_w = std::weak_ptr<TensorNode>(W_q_.node()),
                                           wk_node_w = std::weak_ptr<TensorNode>(W_k_.node()),
                                           wv_node_w = std::weak_ptr<TensorNode>(W_v_.node()),
                                           wo_node_w = std::weak_ptr<TensorNode>(W_o_.node()),
                                           attn_cache_ptr, Q_cache_ptr, K_cache_ptr, V_cache_ptr, concat_cache_ptr,
                                           d_model, num_heads, head_dim, scale, causal, sl]() {
                    auto out = out_node_w.lock();
                    if (!out || !out->grad) return;
                    auto in = in_node_w.lock();
                    auto wq_node = wq_node_w.lock();
                    auto wk_node = wk_node_w.lock();
                    auto wv_node = wv_node_w.lock();
                    auto wo_node = wo_node_w.lock();

                    const Matrix &grad_output = *(out->grad);

                    if (wo_node && wo_node->requires_grad)
                    {
                        wo_node->accumulate_grad(grad_output.matmul_NT(*concat_cache_ptr));
                    }

                    Matrix grad_concat;
                    if (wo_node) {
                        grad_concat = wo_node->data.matmul_TN(grad_output);
                    } else {
                        return;
                    }

                    Matrix grad_Q(d_model, sl), grad_K(d_model, sl), grad_V(d_model, sl);

                    for (std::size_t h = 0; h < num_heads; ++h)
                    {
                        const std::size_t off = h * head_dim;
                        Matrix Q_h = Q_cache_ptr->row_slice(off, head_dim);
                        Matrix K_h = K_cache_ptr->row_slice(off, head_dim);
                        Matrix V_h = V_cache_ptr->row_slice(off, head_dim);
                        const Matrix &attn_h = (*attn_cache_ptr)[h];

                        Matrix grad_head = grad_concat.row_slice(off, head_dim);

                        Matrix grad_V_h = grad_head * attn_h;
                        Matrix grad_attn = grad_head.matmul_TN(V_h);

                        Matrix grad_scores = softmax_backward_rows(grad_attn, attn_h);
                        grad_scores.scale_inplace(scale);

                        if (causal)
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

                    if (wq_node && wq_node->requires_grad && in)
                    {
                        wq_node->accumulate_grad(grad_Q.matmul_NT(in->data));
                    }
                    if (wk_node && wk_node->requires_grad && in)
                    {
                        wk_node->accumulate_grad(grad_K.matmul_NT(in->data));
                    }
                    if (wv_node && wv_node->requires_grad && in)
                    {
                        wv_node->accumulate_grad(grad_V.matmul_NT(in->data));
                    }

                    if (in && in->requires_grad)
                    {
                        Matrix grad_input;
                        if (wq_node) grad_input = wq_node->data.matmul_TN(grad_Q);
                        if (wk_node) grad_input.add_inplace(wk_node->data.matmul_TN(grad_K));
                        if (wv_node) grad_input.add_inplace(wv_node->data.matmul_TN(grad_V));
                        in->accumulate_grad(grad_input);
                    }
                };
            }
            return out;
        }
    };

    // ── FeedForwardModule ──────────────────────────────────────────────────────
    class FeedForwardModule : public Sequential
    {
    public:
        FeedForwardModule(std::size_t d_model, std::size_t d_ff)
        {
            add<LinearModule>(d_model, d_ff);
            add<GELUModule>();
            add<LinearModule>(d_ff, d_model);
        }

        const char *name() const override { return "FeedForwardModule"; }
    };

    // ── TransformerEncoderLayerModule ──────────────────────────────────────────
    class TransformerEncoderLayerModule : public Module
    {
    private:
        LayerNormModule norm1_;
        MultiHeadAttentionModule mha_;
        LayerNormModule norm2_;
        FeedForwardModule ffn_;

    public:
        TransformerEncoderLayerModule(std::size_t d_model, std::size_t num_heads, std::size_t d_ff)
            : norm1_(d_model), mha_(d_model, num_heads), norm2_(d_model), ffn_(d_model, d_ff) {}

        const char *name() const override { return "TransformerEncoderLayerModule"; }
        
        std::size_t param_count() const noexcept override
        {
            return norm1_.param_count() + mha_.param_count() + norm2_.param_count() + ffn_.param_count();
        }
        
        std::vector<Tensor> parameters() override
        {
            std::vector<Tensor> p;
            auto p1 = norm1_.parameters(); p.insert(p.end(), p1.begin(), p1.end());
            auto p2 = mha_.parameters(); p.insert(p.end(), p2.begin(), p2.end());
            auto p3 = norm2_.parameters(); p.insert(p.end(), p3.begin(), p3.end());
            auto p4 = ffn_.parameters(); p.insert(p.end(), p4.begin(), p4.end());
            return p;
        }

        Tensor forward(const Tensor &input) override
        {
            Tensor normed1 = norm1_.forward(input);
            Tensor attn_out = mha_.forward(normed1);
            Tensor residual1 = nn::add(input, attn_out);

            Tensor normed2 = norm2_.forward(residual1);
            Tensor ffn_out = ffn_.forward(normed2);
            return nn::add(residual1, ffn_out);
        }
    };

    // ── TransformerEncoderModule ─────────────────────────────────────────────
    class TransformerEncoderModule : public Module
    {
    private:
        std::vector<std::shared_ptr<TransformerEncoderLayerModule>> layers_;
        std::size_t d_model_;
        std::size_t num_patches_;

    public:
        TransformerEncoderModule(std::size_t d_model, std::size_t num_heads,
                                 std::size_t d_ff, std::size_t num_layers,
                                 std::size_t num_patches)
            : d_model_(d_model), num_patches_(num_patches)
        {
            for (std::size_t i = 0; i < num_layers; ++i)
                layers_.push_back(std::make_shared<TransformerEncoderLayerModule>(d_model, num_heads, d_ff));
        }

        const char *name() const override { return "TransformerEncoderModule"; }

        std::size_t param_count() const noexcept override
        {
            std::size_t total = 0;
            for (const auto &l : layers_) total += l->param_count();
            return total;
        }

        std::vector<Tensor> parameters() override
        {
            std::vector<Tensor> p;
            for (auto &l : layers_)
            {
                auto lp = l->parameters();
                p.insert(p.end(), lp.begin(), lp.end());
            }
            return p;
        }

        Tensor forward(const Tensor &input) override
        {
            const Matrix &in_m = input.data();
            const std::size_t total_cols = in_m.cols();
            const std::size_t batch = total_cols / num_patches_;
            if (batch == 0 || total_cols % num_patches_ != 0)
                throw std::invalid_argument("TransformerEncoderModule: input cols must be a multiple of num_patches");

            Matrix out_m(d_model_, batch);
            
            auto sample_inputs = std::make_shared<std::vector<Tensor>>();
            auto sample_outputs = std::make_shared<std::vector<Tensor>>();
            
            bool req_grad = input.requires_grad();
            for (auto &p : parameters()) if (p.requires_grad()) req_grad = true;

            for (std::size_t b = 0; b < batch; ++b)
            {
                Matrix sample(d_model_, num_patches_);
                for (std::size_t i = 0; i < d_model_; ++i)
                    for (std::size_t j = 0; j < num_patches_; ++j)
                        sample.set_value_unchecked(i, j, in_m.at_unchecked(i, b * num_patches_ + j));

                Tensor sample_t(sample, req_grad);
                sample_inputs->push_back(sample_t);
                
                Tensor out_t = sample_t;
                for (auto &l : layers_)
                    out_t = l->forward(out_t);
                
                sample_outputs->push_back(out_t);

                for (std::size_t i = 0; i < d_model_; ++i)
                {
                    double avg = 0.0;
                    for (std::size_t j = 0; j < num_patches_; ++j)
                        avg += out_t.data().at_unchecked(i, j);
                    out_m.set_value_unchecked(i, b, avg / static_cast<double>(num_patches_));
                }
            }

            Tensor out(out_m, req_grad);

            if (req_grad)
            {
                out.node()->children.push_back(input.node());
                for (auto &p : parameters()) out.node()->children.push_back(p.node());

                out.node()->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out.node()),
                                           in_node_w = std::weak_ptr<TensorNode>(input.node()),
                                           sample_inputs, sample_outputs, batch,
                                           num_patches = num_patches_, d_model = d_model_]() {
                    auto out = out_node_w.lock();
                    if (!out || !out->grad) return;
                    auto in = in_node_w.lock();

                    const Matrix &grad_output = *(out->grad);
                    Matrix grad_input(d_model, batch * num_patches);

                    const double inv_patches = 1.0 / static_cast<double>(num_patches);

                    for (std::size_t b = 0; b < batch; ++b)
                    {
                        Tensor &out_t = (*sample_outputs)[b];
                        
                        Matrix grad_sample(d_model, num_patches);
                        for (std::size_t i = 0; i < d_model; ++i)
                        {
                            double g = grad_output.at_unchecked(i, b) * inv_patches;
                            for (std::size_t j = 0; j < num_patches; ++j)
                                grad_sample.set_value_unchecked(i, j, g);
                        }
                        
                        out_t.grad() = grad_sample;
                        out_t.backward();
                        
                        if (in && in->requires_grad)
                        {
                            Tensor &in_t = (*sample_inputs)[b];
                            if (in_t.grad().size() > 0)
                            {
                                for (std::size_t i = 0; i < d_model; ++i)
                                    for (std::size_t j = 0; j < num_patches; ++j)
                                        grad_input.set_value_unchecked(i, b * num_patches + j, in_t.grad().at_unchecked(i, j));
                            }
                        }
                    }
                    
                    if (in && in->requires_grad)
                    {
                        in->accumulate_grad(grad_input);
                    }
                };
            }
            return out;
        }
    };

    // ── PatchEmbeddingModule ─────────────────────────────────────────────────
    class PatchEmbeddingModule : public Module
    {
    private:
        std::size_t img_size_;
        std::size_t patch_size_;
        std::size_t d_model_;
        std::size_t num_patches_;
        std::size_t patch_dim_;
        LinearModule proj_;

        static std::size_t validated_num_patches(std::size_t img_size, std::size_t patch_size)
        {
            if (patch_size == 0)
                throw std::invalid_argument("PatchEmbeddingModule: patch_size must be > 0");
            if (img_size == 0)
                throw std::invalid_argument("PatchEmbeddingModule: img_size must be > 0");
            if (img_size % patch_size != 0)
                throw std::invalid_argument("PatchEmbeddingModule: img_size must be divisible by patch_size");
            const std::size_t grid = img_size / patch_size;
            return grid * grid;
        }

    public:
        PatchEmbeddingModule(std::size_t img_size, std::size_t patch_size, std::size_t d_model)
            : img_size_(img_size), patch_size_(patch_size), d_model_(d_model),
              num_patches_(validated_num_patches(img_size, patch_size)),
              patch_dim_(patch_size * patch_size),
              proj_(patch_dim_, d_model) {}

        const char *name() const override { return "PatchEmbeddingModule"; }
        std::size_t param_count() const noexcept override { return proj_.param_count(); }

        std::vector<Tensor> parameters() override { return proj_.parameters(); }

        Tensor forward(const Tensor &input) override
        {
            const Matrix &in_m = input.data();
            const std::size_t batch = in_m.cols();
            if (in_m.rows() != img_size_ * img_size_)
                throw std::invalid_argument("PatchEmbeddingModule: invalid input shape");

            Matrix patches(patch_dim_, num_patches_ * batch);
            const std::size_t grid = img_size_ / patch_size_;

            for (std::size_t b = 0; b < batch; ++b)
            {
                for (std::size_t py = 0; py < grid; ++py)
                {
                    for (std::size_t px = 0; px < grid; ++px)
                    {
                        const std::size_t p_idx = py * grid + px;
                        for (std::size_t dy = 0; dy < patch_size_; ++dy)
                        {
                            for (std::size_t dx = 0; dx < patch_size_; ++dx)
                            {
                                const std::size_t img_y = py * patch_size_ + dy;
                                const std::size_t img_x = px * patch_size_ + dx;
                                const std::size_t flat_img = img_y * img_size_ + img_x;
                                const std::size_t flat_patch = dy * patch_size_ + dx;
                                patches.set_value_unchecked(flat_patch, b * num_patches_ + p_idx,
                                                            in_m.at_unchecked(flat_img, b));
                            }
                        }
                    }
                }
            }

            bool req_grad = input.requires_grad();
            for (auto &p : parameters()) if (p.requires_grad()) req_grad = true;

            Tensor patches_t(patches, req_grad);
            
            if (req_grad)
            {
                patches_t.node()->children.push_back(input.node());
                patches_t.node()->backward_op = [patches_t_node_w = std::weak_ptr<TensorNode>(patches_t.node()),
                                                 in_node_w = std::weak_ptr<TensorNode>(input.node()),
                                                 batch, num_patches = num_patches_,
                                                 grid, patch_size = patch_size_, img_size = img_size_]() {
                    auto patches_t_node = patches_t_node_w.lock();
                    if (!patches_t_node || !patches_t_node->grad) return;
                    auto in = in_node_w.lock();

                    if (in && in->requires_grad)
                    {
                        const Matrix &grad_patches = *(patches_t_node->grad);
                        Matrix grad_input(img_size * img_size, batch);

                        for (std::size_t b = 0; b < batch; ++b)
                        {
                            for (std::size_t py = 0; py < grid; ++py)
                            {
                                for (std::size_t px = 0; px < grid; ++px)
                                {
                                    const std::size_t p_idx = py * grid + px;
                                    for (std::size_t dy = 0; dy < patch_size; ++dy)
                                    {
                                        for (std::size_t dx = 0; dx < patch_size; ++dx)
                                        {
                                            const std::size_t img_y = py * patch_size + dy;
                                            const std::size_t img_x = px * patch_size + dx;
                                            const std::size_t flat_img = img_y * img_size + img_x;
                                            const std::size_t flat_patch = dy * patch_size + dx;
                                            grad_input.set_value_unchecked(flat_img, b, 
                                                grad_patches.at_unchecked(flat_patch, b * num_patches + p_idx));
                                        }
                                    }
                                }
                            }
                        }
                        in->accumulate_grad(grad_input);
                    }
                };
            }
            
            return proj_.forward(patches_t);
        }
    };

    // ── GPTBlockModule ───────────────────────────────────────────────────────
    class GPTBlockModule : public Module
    {
    private:
        MultiHeadAttentionModule self_attn_;
        LayerNormModule norm1_;
        FeedForwardModule ff_;
        LayerNormModule norm2_;

    public:
        GPTBlockModule(std::size_t d_model, std::size_t num_heads, std::size_t d_ff)
            : self_attn_(d_model, num_heads, true),
              norm1_(d_model),
              ff_(d_model, d_ff),
              norm2_(d_model)
        {}

        const char *name() const override { return "GPTBlockModule"; }

        std::size_t param_count() const noexcept override
        {
            return self_attn_.param_count() + norm1_.param_count()
                 + ff_.param_count() + norm2_.param_count();
        }

        std::vector<Tensor> parameters() override
        {
            std::vector<Tensor> params;
            auto sa = self_attn_.parameters(); params.insert(params.end(), sa.begin(), sa.end());
            auto n1 = norm1_.parameters(); params.insert(params.end(), n1.begin(), n1.end());
            auto f = ff_.parameters(); params.insert(params.end(), f.begin(), f.end());
            auto n2 = norm2_.parameters(); params.insert(params.end(), n2.begin(), n2.end());
            return params;
        }

        Tensor forward(const Tensor &input) override
        {
            Tensor sa_out = self_attn_.forward(norm1_.forward(input));
            Tensor residual1 = nn::add(input, sa_out);

            Tensor ff_out = ff_.forward(norm2_.forward(residual1));
            return nn::add(residual1, ff_out);
        }
    };

    // ── GPTModelModule ───────────────────────────────────────────────────────
    class GPTModelModule : public Module
    {
    private:
        std::size_t vocab_size_;
        std::size_t d_model_;
        std::size_t seq_len_;

        Tensor token_emb_;
        Tensor pos_emb_;

        std::vector<std::shared_ptr<GPTBlockModule>> blocks_;
        LayerNormModule ln_f_;
        LinearModule lm_head_;

    public:
        GPTModelModule(std::size_t vocab_size, std::size_t d_model, std::size_t seq_len,
                       std::size_t num_heads, std::size_t d_ff, std::size_t num_layers)
            : vocab_size_(vocab_size), d_model_(d_model), seq_len_(seq_len),
              token_emb_(Matrix(vocab_size, d_model), true),
              pos_emb_(Matrix(seq_len, d_model), true),
              ln_f_(d_model),
              lm_head_(d_model, vocab_size)
        {
            constexpr double emb_init_std = 0.02;
            std::mt19937_64 rng{42};
            std::normal_distribution<double> dist(0.0, emb_init_std);
            for (std::size_t i = 0; i < token_emb_.data().size(); ++i)
                token_emb_.data().data()[i] = dist(rng);
            for (std::size_t i = 0; i < pos_emb_.data().size(); ++i)
                pos_emb_.data().data()[i] = dist(rng);

            for (std::size_t i = 0; i < num_layers; ++i)
                blocks_.push_back(std::make_shared<GPTBlockModule>(d_model, num_heads, d_ff));
        }

        const char *name() const override { return "GPTModelModule"; }

        std::size_t param_count() const noexcept override
        {
            std::size_t p = token_emb_.data().size() + pos_emb_.data().size();
            for (const auto &b : blocks_)
                p += b->param_count();
            p += ln_f_.param_count();
            p += lm_head_.param_count();
            return p;
        }

        std::vector<Tensor> parameters() override
        {
            std::vector<Tensor> params;
            params.push_back(token_emb_);
            params.push_back(pos_emb_);
            for (auto &b : blocks_)
            {
                auto bp = b->parameters();
                params.insert(params.end(), bp.begin(), bp.end());
            }
            auto lp = ln_f_.parameters(); params.insert(params.end(), lp.begin(), lp.end());
            auto hp = lm_head_.parameters(); params.insert(params.end(), hp.begin(), hp.end());
            return params;
        }

        Tensor forward(const Tensor &input) override
        {
            const Matrix &in_m = input.data();
            const std::size_t sl = in_m.rows();
            const std::size_t batch = in_m.cols();

            Matrix out_m(vocab_size_, sl * batch);
            
            auto sample_inputs = std::make_shared<std::vector<Tensor>>();
            auto sample_outputs = std::make_shared<std::vector<Tensor>>();

            bool req_grad = true; 

            for (std::size_t b = 0; b < batch; ++b)
            {
                Matrix x(d_model_, sl);
                for (std::size_t t = 0; t < sl; ++t)
                {
                    auto token_id = static_cast<std::size_t>(in_m.at_unchecked(t, b));
                    if (token_id >= vocab_size_) token_id = 0;

                    for (std::size_t d = 0; d < d_model_; ++d)
                    {
                        double pe = (t < seq_len_) ? pos_emb_.data().at_unchecked(t, d) : 0.0;
                        x.set_value_unchecked(d, t, token_emb_.data().at_unchecked(token_id, d) + pe);
                    }
                }

                Tensor x_t(x, req_grad);
                
                // Embeddings grad accumulation
                if (req_grad)
                {
                    x_t.node()->children.push_back(token_emb_.node());
                    x_t.node()->children.push_back(pos_emb_.node());
                    x_t.node()->backward_op = [x_t_node_w = std::weak_ptr<TensorNode>(x_t.node()),
                                               tok_node_w = std::weak_ptr<TensorNode>(token_emb_.node()),
                                               pos_node_w = std::weak_ptr<TensorNode>(pos_emb_.node()),
                                               in_m, b, sl, vocab_size = vocab_size_, d_model = d_model_, seq_len = seq_len_]() {
                        auto x_t_node = x_t_node_w.lock();
                        if (!x_t_node || !x_t_node->grad) return;
                        auto tok = tok_node_w.lock();
                        auto pos = pos_node_w.lock();

                        const Matrix &grad_x = *(x_t_node->grad);
                        if (tok)
                        {
                            Matrix grad_tok(vocab_size, d_model);
                            for (std::size_t t = 0; t < sl; ++t)
                            {
                                auto token_id = static_cast<std::size_t>(in_m.at_unchecked(t, b));
                                if (token_id >= vocab_size) token_id = 0;
                                for (std::size_t d = 0; d < d_model; ++d)
                                    grad_tok.data()[token_id * d_model + d] += grad_x.at_unchecked(d, t);
                            }
                            tok->accumulate_grad(grad_tok);
                        }
                        if (pos)
                        {
                            Matrix grad_pos(seq_len, d_model);
                            for (std::size_t t = 0; t < sl; ++t)
                            {
                                if (t < seq_len)
                                {
                                    for (std::size_t d = 0; d < d_model; ++d)
                                        grad_pos.data()[t * d_model + d] += grad_x.at_unchecked(d, t);
                                }
                            }
                            pos->accumulate_grad(grad_pos);
                        }
                    };
                }

                Tensor out_t = x_t;
                for (auto &l : blocks_)
                    out_t = l->forward(out_t);

                out_t = ln_f_.forward(out_t);
                Tensor logits_t = lm_head_.forward(out_t);

                sample_outputs->push_back(logits_t);

                for (std::size_t r = 0; r < vocab_size_; ++r)
                    for (std::size_t t = 0; t < sl; ++t)
                        out_m.set_value_unchecked(r, t * batch + b, logits_t.data().at_unchecked(r, t));
            }

            Tensor out(out_m, req_grad);

            if (req_grad)
            {
                out.node()->children.push_back(input.node());
                for (auto &p : parameters()) out.node()->children.push_back(p.node());

                out.node()->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out.node()),
                                           sample_outputs, batch, sl, vocab_size = vocab_size_]() {
                    auto out = out_node_w.lock();
                    if (!out || !out->grad) return;

                    const Matrix &grad_output = *(out->grad);

                    for (std::size_t b = 0; b < batch; ++b)
                    {
                        Tensor &logits_t = (*sample_outputs)[b];
                        
                        Matrix grad_sample(vocab_size, sl);
                        for (std::size_t r = 0; r < vocab_size; ++r)
                            for (std::size_t t = 0; t < sl; ++t)
                                grad_sample.set_value_unchecked(r, t, grad_output.at_unchecked(r, t * batch + b));
                        
                        logits_t.grad() = grad_sample;
                        logits_t.backward();
                    }
                };
            }
            
            return out;
        }

        std::vector<std::size_t> generate(const std::vector<std::size_t> &prompt,
                                          std::size_t max_new_tokens,
                                          double temperature = 1.0)
        {
            std::vector<std::size_t> context(prompt);
            std::vector<std::size_t> generated;
            std::mt19937_64 rng{std::random_device{}()};
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            
            for (std::size_t step = 0; step < max_new_tokens; ++step)
            {
                std::size_t start = 0;
                if (context.size() > seq_len_)
                    start = context.size() - seq_len_;

                std::size_t cur_len = context.size() - start;
                Matrix input_m(cur_len, 1);
                for (std::size_t t = 0; t < cur_len; ++t)
                    input_m.set_value_unchecked(t, 0, static_cast<double>(context[start + t]));

                Tensor input(input_m);
                Tensor logits_t = forward(input);
                Matrix logits = logits_t.data();

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
            
            return generated;
        }
    };

} // namespace nn

#endif // NN_MODULE_H
