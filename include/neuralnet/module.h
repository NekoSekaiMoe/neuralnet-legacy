#ifndef NN_MODULE_H
#define NN_MODULE_H

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
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

    public:
        LinearModule(std::size_t in_features, std::size_t out_features, bool bias = true)
            : in_features_(in_features), out_features_(out_features), bias_(bias)
        {
            double limit = std::sqrt(6.0 / static_cast<double>(in_features + out_features));
            std::mt19937_64 rng(42);
            std::uniform_real_distribution<double> dist(-limit, limit);
            
            Matrix w_mat(out_features, in_features);
            for(auto& val : w_mat.data()) val = dist(rng);
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
            if (input.requires_grad() && is_training_) {
                normalized_tensor.node()->children.push_back(input.node());
                auto in_node = input.node();
                normalized_tensor.node()->backward_op = [out_w = std::weak_ptr<TensorNode>(normalized_tensor.node()),
                                                         in_w = std::weak_ptr<TensorNode>(in_node),
                                                         batch_var, eps = eps_, num_features = num_features_, batch_size,
                                                         normalized]() {
                    auto out = out_w.lock();
                    auto in = in_w.lock();
                    if (!out || !out->grad || !in || !in->requires_grad) return;
                    Matrix a_grad = *(out->grad);
                    Matrix dx(num_features, batch_size);
                    double scale = 1.0 / batch_size;
                    
                    for (std::size_t i = 0; i < num_features; ++i) {
                        double sd = 0.0;
                        double sdx = 0.0;
                        for (std::size_t j = 0; j < batch_size; ++j) {
                            double g = a_grad.at_unchecked(i, j);
                            double nx = normalized.at_unchecked(i, j);
                            sd += g;
                            sdx += g * nx;
                        }
                        double std_inv = 1.0 / std::sqrt(batch_var.at_unchecked(i, 0) + eps);
                        for (std::size_t j = 0; j < batch_size; ++j) {
                            double g = a_grad.at_unchecked(i, j);
                            double nx = normalized.at_unchecked(i, j);
                            dx.set_value_unchecked(i, j, scale * (batch_size * g - sd - nx * sdx) * std_inv);
                        }
                    }
                    in->accumulate_grad(dx);
                };
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

} // namespace nn

#endif // NN_MODULE_H
