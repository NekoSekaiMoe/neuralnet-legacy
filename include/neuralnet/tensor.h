#pragma once

#include "matrix.h"
#include <memory>
#include <vector>
#include <functional>
#include <optional>
#include <stdexcept>
#include <unordered_set>

namespace nn {

// 前置声明
class Tensor;

struct TensorNode {
    Matrix data;
    std::optional<Matrix> grad;
    bool requires_grad;
    bool is_leaf;

    // 反向传播函数与依赖
    std::function<void()> backward_op;
    std::vector<std::shared_ptr<TensorNode>> children;

    TensorNode(const Matrix& m, bool req_grad = false, bool leaf = true)
        : data(m), requires_grad(req_grad), is_leaf(leaf)
    {
        if (requires_grad) {
            grad = Matrix(m.rows(), m.cols());
            grad->zero();
        }
    }

    void accumulate_grad(const Matrix& g) {
        if (!requires_grad) return;
        if (!grad) {
            grad = Matrix(data.rows(), data.cols());
            grad->zero();
        }
        
        // 累加梯度，注意需要做形状检查
        if (g.rows() == grad->rows() && g.cols() == grad->cols()) {
            // grad += g
            const auto& g_span = g.data();
            auto& my_span = grad->data();
            std::size_t n = grad->size();
            for (std::size_t i = 0; i < n; ++i) {
                my_span[i] += g_span[i];
            }
        } else if (g.rows() == grad->rows() && g.cols() == 1) {
            // Broadcasting over columns (e.g. bias gradient)
            const auto& g_span = g.data();
            auto& my_span = grad->data();
            std::size_t r = grad->rows();
            std::size_t c = grad->cols();
            for (std::size_t j = 0; j < c; ++j) {
                for (std::size_t i = 0; i < r; ++i) {
                    my_span[i + j * r] += g_span[i];
                }
            }
        } else {
            throw std::invalid_argument("accumulate_grad: shape mismatch");
        }
    }
};

class Tensor {
private:
    std::shared_ptr<TensorNode> node_;

public:
    Tensor() : node_(nullptr) {}
    
    Tensor(const Matrix& m, bool requires_grad = false)
        : node_(std::make_shared<TensorNode>(m, requires_grad, true)) {}
    
    Tensor(std::shared_ptr<TensorNode> node) : node_(std::move(node)) {}

    // 基本属性
    Matrix& data() { return node_->data; }
    const Matrix& data() const { return node_->data; }
    
    Matrix& grad() {
        if (!node_->grad) {
            node_->grad = Matrix(node_->data.rows(), node_->data.cols());
            node_->grad->zero();
        }
        return *node_->grad;
    }
    const Matrix& grad() const {
        if (!node_->grad) throw std::runtime_error("Tensor has no gradient");
        return *node_->grad;
    }

    bool requires_grad() const { return node_->requires_grad; }
    bool is_leaf() const { return node_->is_leaf; }
    
    std::size_t rows() const { return node_->data.rows(); }
    std::size_t cols() const { return node_->data.cols(); }
    
    std::shared_ptr<TensorNode> node() const { return node_; }

    // 反向传播
    void backward() {
        if (!node_->requires_grad) return;
        
        // 自动将起始梯度置为 1
        if (!node_->grad) {
            node_->grad = Matrix(rows(), cols());
            node_->grad->zero();
        }
        auto& span = node_->grad->data();
        for (std::size_t i = 0; i < node_->grad->size(); ++i) {
            span[i] = 1.0;
        }

        // 拓扑排序
        std::vector<std::shared_ptr<TensorNode>> topo;
        std::unordered_set<TensorNode*> visited;
        
        std::function<void(std::shared_ptr<TensorNode>)> build_topo = 
            [&](std::shared_ptr<TensorNode> v) {
                if (!v) return;
                if (visited.find(v.get()) == visited.end()) {
                    visited.insert(v.get());
                    for (auto& child : v->children) {
                        build_topo(child);
                    }
                    topo.push_back(v);
                }
            };
            
        build_topo(node_);
        
        // 反向执行
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if ((*it)->backward_op) {
                (*it)->backward_op();
            }
        }
    }
};

// ==================== 基础算子 ====================

// C = A + B
inline Tensor add(const Tensor& a, const Tensor& b) {
    bool req_grad = a.requires_grad() || b.requires_grad();
    Matrix out_data = a.data() + b.data(); // Matrix operator+
    auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
    
    if (req_grad) {
        out_node->children.push_back(a.node());
        out_node->children.push_back(b.node());
        
        auto a_node = a.node();
        auto b_node = b.node();
        
        out_node->backward_op = [out_w = std::weak_ptr<TensorNode>(out_node),
                                 in1_w = std::weak_ptr<TensorNode>(a_node),
                                 in2_w = std::weak_ptr<TensorNode>(b_node)]() {
            auto out = out_w.lock();
            auto in1 = in1_w.lock();
            auto in2 = in2_w.lock();
            if (out && out->grad) {
                std::cout << "out->grad(0,0): " << out->grad->at(0,0) << "\n";
                if (in1 && in1->requires_grad) {
                    in1->accumulate_grad(*(out->grad));
                    std::cout << "in1->grad(0,0) after: " << in1->grad->at(0,0) << "\n";
                }
                if (in2 && in2->requires_grad) in2->accumulate_grad(*(out->grad));
            }
        };
    }
    
    return Tensor(out_node);
}

// C = A * B (Element-wise)
inline Tensor mul(const Tensor& a, const Tensor& b) {
    bool req_grad = a.requires_grad() || b.requires_grad();
    if (a.rows() != b.rows() || a.cols() != b.cols()) throw std::invalid_argument("mul dimension mismatch");
    Matrix out_data(a.rows(), a.cols());
    auto& out_span = out_data.data();
    auto& a_span = a.data().data();
    auto& b_span = b.data().data();
    for (std::size_t i = 0; i < out_data.size(); ++i) {
        out_span[i] = a_span[i] * b_span[i];
    }
    
    auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
    
    if (req_grad) {
        out_node->children.push_back(a.node());
        out_node->children.push_back(b.node());
        
        auto a_node = a.node();
        auto b_node = b.node();
        
        out_node->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out_node),
                                 a_node_w = std::weak_ptr<TensorNode>(a_node),
                                 b_node_w = std::weak_ptr<TensorNode>(b_node)]() {
            auto out = out_node_w.lock();
            auto a = a_node_w.lock();
            auto b = b_node_w.lock();
            if (!out || !out->grad) return;
            if (a && a->requires_grad) {
                Matrix a_grad(a->data.rows(), a->data.cols());
                auto& a_g_span = a_grad.data();
                auto& out_g_span = out->grad->data();
                auto& b_span = b->data.data();
                for (std::size_t i = 0; i < a_grad.size(); ++i) a_g_span[i] = out_g_span[i] * b_span[i];
                a->accumulate_grad(a_grad);
            }
            if (b && b->requires_grad) {
                Matrix b_grad(b->data.rows(), b->data.cols());
                auto& b_g_span = b_grad.data();
                auto& out_g_span = out->grad->data();
                auto& a_span = a->data.data();
                for (std::size_t i = 0; i < b_grad.size(); ++i) b_g_span[i] = out_g_span[i] * a_span[i];
                b->accumulate_grad(b_grad);
            }
        };
    }
    
    return Tensor(out_node);
}

// C = matmul(A, B)
inline Tensor matmul(const Tensor& a, const Tensor& b) {
    bool req_grad = a.requires_grad() || b.requires_grad();
    Matrix out_data = a.data() * b.data();
    auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
    
    if (req_grad) {
        out_node->children.push_back(a.node());
        out_node->children.push_back(b.node());
        
        auto a_node = a.node();
        auto b_node = b.node();
        
        out_node->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out_node),
                                 a_node_w = std::weak_ptr<TensorNode>(a_node),
                                 b_node_w = std::weak_ptr<TensorNode>(b_node)]() {
            auto out = out_node_w.lock();
            auto a = a_node_w.lock();
            auto b = b_node_w.lock();
            if (!out || !out->grad) return;
            if (a && a->requires_grad) {
                // dA = dC * B^T
                a->accumulate_grad(out->grad->matmul_NT(b->data));
            }
            if (b && b->requires_grad) {
                // dB = A^T * dC
                b->accumulate_grad(a->data.matmul_TN(*(out->grad)));
            }
        };
    }
    
    return Tensor(out_node);
}

// ReLU
inline Tensor relu(const Tensor& a) {
    bool req_grad = a.requires_grad();
    Matrix out_data = a.data();
    auto& span = out_data.data();
    for (std::size_t i = 0; i < out_data.size(); ++i) {
        if (span[i] < 0) span[i] = 0;
    }
    auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
    if (req_grad) {
        out_node->children.push_back(a.node());
        auto a_node = a.node();
        out_node->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out_node),
                                 a_node_w = std::weak_ptr<TensorNode>(a_node)]() {
            auto out = out_node_w.lock();
            auto a = a_node_w.lock();
            if (!out || !out->grad || !a || !a->requires_grad) return;
            Matrix a_grad = *(out->grad);
            auto& ag_span = a_grad.data();
            auto& out_data_span = out->data.data();
            for (std::size_t i = 0; i < a_grad.size(); ++i) {
                if (out_data_span[i] <= 0) ag_span[i] = 0;
            }
            a->accumulate_grad(a_grad);
        };
    }
    return Tensor(out_node);
}

// Linear: W * x + b
inline Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    bool req_grad = x.requires_grad() || weight.requires_grad() || bias.requires_grad();
    Matrix out_data = weight.data() * x.data();
    const std::size_t cols = out_data.cols();
    auto& out_span = out_data.data();
    auto& b_span = bias.data().data();
    for (std::size_t r = 0; r < out_data.rows(); ++r) {
        double b_val = b_span[r];
        for (std::size_t c = 0; c < cols; ++c) {
            out_span[r * cols + c] += b_val;
        }
    }
    auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
    if (req_grad) {
        out_node->children.push_back(x.node());
        out_node->children.push_back(weight.node());
        out_node->children.push_back(bias.node());
        auto x_node = x.node();
        auto w_node = weight.node();
        auto b_node = bias.node();
        out_node->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out_node),
                                 x_node_w = std::weak_ptr<TensorNode>(x_node),
                                 w_node_w = std::weak_ptr<TensorNode>(w_node),
                                 b_node_w = std::weak_ptr<TensorNode>(b_node)]() {
            auto out = out_node_w.lock();
            auto x = x_node_w.lock();
            auto w = w_node_w.lock();
            auto b = b_node_w.lock();
            if (!out || !out->grad) return;
            
            if (x && x->requires_grad) {
                x->accumulate_grad(w->data.matmul_TN(*(out->grad)));
            }
            if (w && w->requires_grad) {
                w->accumulate_grad(out->grad->matmul_NT(x->data));
            }
            if (b && b->requires_grad) {
                std::vector<double> b_vec = out->grad->rowwise_sum();
                Matrix b_grad(b_vec.size(), 1);
                auto& bg_span = b_grad.data();
                for (std::size_t i = 0; i < b_vec.size(); ++i) {
                    bg_span[i] = b_vec[i];
                }
                b->accumulate_grad(b_grad);
            }
        };
    }
    return Tensor(out_node);
}

// LeakyReLU
inline Tensor leaky_relu(const Tensor& a, double negative_slope = 0.01) {
    bool req_grad = a.requires_grad();
    Matrix out_data = a.data();
    auto& span = out_data.data();
    for (std::size_t i = 0; i < out_data.size(); ++i) {
        if (span[i] < 0) span[i] *= negative_slope;
    }
    auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
    if (req_grad) {
        out_node->children.push_back(a.node());
        auto a_node = a.node();
        out_node->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out_node),
                                 a_node_w = std::weak_ptr<TensorNode>(a_node),
                                 negative_slope]() {
            auto out = out_node_w.lock();
            auto a = a_node_w.lock();
            if (!out || !out->grad || !a || !a->requires_grad) return;
            Matrix a_grad = *(out->grad);
            auto& ag_span = a_grad.data();
            auto& out_data_span = out->data.data();
            for (std::size_t i = 0; i < a_grad.size(); ++i) {
                if (out_data_span[i] < 0) ag_span[i] *= negative_slope;
            }
            a->accumulate_grad(a_grad);
        };
    }
    return Tensor(out_node);
}

// Sigmoid
inline Tensor sigmoid(const Tensor& a) {
    bool req_grad = a.requires_grad();
    Matrix out_data = a.data();
    auto& span = out_data.data();
    for (std::size_t i = 0; i < out_data.size(); ++i) {
        span[i] = 1.0 / (1.0 + std::exp(-span[i]));
    }
    auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
    if (req_grad) {
        out_node->children.push_back(a.node());
        auto a_node = a.node();
        out_node->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out_node),
                                 a_node_w = std::weak_ptr<TensorNode>(a_node)]() {
            auto out = out_node_w.lock();
            auto a = a_node_w.lock();
            if (!out || !out->grad || !a || !a->requires_grad) return;
            Matrix a_grad = *(out->grad);
            auto& ag_span = a_grad.data();
            auto& out_data_span = out->data.data();
            for (std::size_t i = 0; i < a_grad.size(); ++i) {
                double val = out_data_span[i];
                ag_span[i] *= val * (1.0 - val);
            }
            a->accumulate_grad(a_grad);
        };
    }
    return Tensor(out_node);
}

// Tanh
inline Tensor tanh(const Tensor& a) {
    bool req_grad = a.requires_grad();
    Matrix out_data = a.data();
    auto& span = out_data.data();
    for (std::size_t i = 0; i < out_data.size(); ++i) {
        span[i] = std::tanh(span[i]);
    }
    auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
    if (req_grad) {
        out_node->children.push_back(a.node());
        auto a_node = a.node();
        out_node->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out_node),
                                 a_node_w = std::weak_ptr<TensorNode>(a_node)]() {
            auto out = out_node_w.lock();
            auto a = a_node_w.lock();
            if (!out || !out->grad || !a || !a->requires_grad) return;
            Matrix a_grad = *(out->grad);
            auto& ag_span = a_grad.data();
            auto& out_data_span = out->data.data();
            for (std::size_t i = 0; i < a_grad.size(); ++i) {
                double val = out_data_span[i];
                ag_span[i] *= (1.0 - val * val);
            }
            a->accumulate_grad(a_grad);
        };
    }
    return Tensor(out_node);
}

// GELU
inline Tensor gelu(const Tensor& a) {
    bool req_grad = a.requires_grad();
    Matrix out_data = a.data();
    auto& span = out_data.data();
    auto& a_span = a.data().data();
    const double SQRT_2_OVER_PI = 0.7978845608028654;
    for (std::size_t i = 0; i < out_data.size(); ++i) {
        double x = a_span[i];
        double x3 = x * x * x;
        span[i] = 0.5 * x * (1.0 + std::tanh(SQRT_2_OVER_PI * (x + 0.044715 * x3)));
    }
    auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
    if (req_grad) {
        out_node->children.push_back(a.node());
        auto a_node = a.node();
        out_node->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out_node),
                                 a_node_w = std::weak_ptr<TensorNode>(a_node)]() {
            auto out = out_node_w.lock();
            auto a = a_node_w.lock();
            if (!out || !out->grad || !a || !a->requires_grad) return;
            Matrix a_grad = *(out->grad);
            auto& ag_span = a_grad.data();
            auto& a_span = a->data.data();
            const double SQRT_2_OVER_PI = 0.7978845608028654;
            for (std::size_t i = 0; i < a_grad.size(); ++i) {
                double x = a_span[i];
                double x3 = x * x * x;
                double inner = SQRT_2_OVER_PI * (x + 0.044715 * x3);
                double t = std::tanh(inner);
                double dt = 1.0 - t * t;
                double inner_grad = SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x * x);
                double dgelu = 0.5 * (1.0 + t) + 0.5 * x * dt * inner_grad;
                ag_span[i] *= dgelu;
            }
            a->accumulate_grad(a_grad);
        };
    }
    return Tensor(out_node);
}

// Dropout
inline Tensor dropout(const Tensor& a, double p, bool training, std::mt19937_64& rng) {
    if (!training || p == 0.0) return a; // Passthrough if not training

    bool req_grad = a.requires_grad();
    Matrix out_data = a.data();
    auto& span = out_data.data();
    
    // Create mask and apply it
    std::vector<double> mask(out_data.size());
    std::bernoulli_distribution dist(1.0 - p);
    const double scale = 1.0 / (1.0 - p);
    
    for (std::size_t i = 0; i < out_data.size(); ++i) {
        if (dist(rng)) {
            mask[i] = scale;
            span[i] *= scale;
        } else {
            mask[i] = 0.0;
            span[i] = 0.0;
        }
    }
    
    auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
    if (req_grad) {
        out_node->children.push_back(a.node());
        auto a_node = a.node();
        
        // Capture mask by value into closure
        out_node->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out_node),
                                 a_node_w = std::weak_ptr<TensorNode>(a_node),
                                 mask = std::move(mask)]() {
            auto out = out_node_w.lock();
            auto a = a_node_w.lock();
            if (!out || !out->grad || !a || !a->requires_grad) return;
            Matrix a_grad = *(out->grad);
            auto& ag_span = a_grad.data();
            for (std::size_t i = 0; i < a_grad.size(); ++i) {
                ag_span[i] *= mask[i];
            }
            a->accumulate_grad(a_grad);
        };
    }
    return Tensor(out_node);
}


// BatchNorm1d
inline Tensor batchnorm1d(const Tensor& input, const Tensor& gamma, const Tensor& beta,
                          Matrix& running_mean, Matrix& running_var,
                          int64_t& num_batches_tracked,
                          double momentum, double eps, bool training, bool affine) {
    bool req_grad = input.requires_grad() || (affine && (gamma.requires_grad() || beta.requires_grad()));
    std::size_t num_features = input.rows();
    std::size_t batch_size = input.cols();
    
    if (batch_size == 0) return Tensor(Matrix(num_features, 0), req_grad);
    
    Matrix batch_mean(num_features, 1, 0.0);
    Matrix batch_var(num_features, 1, 0.0);
    Matrix normalized(num_features, batch_size);
    
    if (training) {
        for (std::size_t i = 0; i < num_features; ++i) {
            double sum = 0.0;
            for (std::size_t j = 0; j < batch_size; ++j) {
                sum += input.data().at_unchecked(i, j);
            }
            double mu = sum / batch_size;
            batch_mean.set_value_unchecked(i, 0, mu);
            
            double var_sum = 0.0;
            for (std::size_t j = 0; j < batch_size; ++j) {
                double d = input.data().at_unchecked(i, j) - mu;
                var_sum += d * d;
            }
            batch_var.set_value_unchecked(i, 0, var_sum / batch_size);
            
            running_mean.set_value_unchecked(i, 0, (1.0 - momentum) * running_mean.at_unchecked(i, 0) + momentum * mu);
            running_var.set_value_unchecked(i, 0, (1.0 - momentum) * running_var.at_unchecked(i, 0) + momentum * (var_sum / batch_size));
        }
        num_batches_tracked++;
    }
    
    const Matrix& mean_src = training ? batch_mean : running_mean;
    const Matrix& var_src = training ? batch_var : running_var;
    
    std::vector<double> inv_std(num_features);
    for (std::size_t i = 0; i < num_features; ++i) {
        inv_std[i] = 1.0 / std::sqrt(var_src.at_unchecked(i, 0) + eps);
    }
    
    Matrix out_data(num_features, batch_size);
    auto& in_span = input.data().data();
    auto& out_span = out_data.data();
    auto& norm_span = normalized.data();
    
    for (std::size_t idx = 0; idx < input.data().size(); ++idx) {
        std::size_t i = idx / batch_size;
        double mu = mean_src.at_unchecked(i, 0);
        double norm = (in_span[idx] - mu) * inv_std[i];
        norm_span[idx] = norm;
        if (affine) {
            out_span[idx] = norm * gamma.data().at_unchecked(i, 0) + beta.data().at_unchecked(i, 0);
        } else {
            out_span[idx] = norm;
        }
    }
    
    auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
    if (req_grad) {
        out_node->children.push_back(input.node());
        if (affine) {
            out_node->children.push_back(gamma.node());
            out_node->children.push_back(beta.node());
        }
        
        auto in_node = input.node();
        auto g_node = gamma.node();
        auto b_node = beta.node();
        
        out_node->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out_node),
                                 in_node_w = std::weak_ptr<TensorNode>(in_node),
                                 g_node_w = std::weak_ptr<TensorNode>(g_node),
                                 b_node_w = std::weak_ptr<TensorNode>(b_node),
                                 normalized = std::move(normalized),
                                 inv_std = std::move(inv_std),
                                 training, affine, batch_size, num_features]() {
            auto out = out_node_w.lock();
            auto in = in_node_w.lock();
            if (!out || !out->grad) return;
            
            Matrix go = *(out->grad);
            
            if (affine) {
                auto g = g_node_w.lock();
                auto b = b_node_w.lock();
                if (g && g->requires_grad) {
                    Matrix dg(num_features, 1, 0.0);
                    for (std::size_t i = 0; i < num_features; ++i) {
                        double s = 0.0;
                        for (std::size_t j = 0; j < batch_size; ++j) {
                            s += go.at_unchecked(i, j) * normalized.at_unchecked(i, j);
                        }
                        dg.set_value_unchecked(i, 0, s);
                    }
                    g->accumulate_grad(dg);
                }
                if (b && b->requires_grad) {
                    Matrix db(num_features, 1, 0.0);
                    for (std::size_t i = 0; i < num_features; ++i) {
                        double s = 0.0;
                        for (std::size_t j = 0; j < batch_size; ++j) {
                            s += go.at_unchecked(i, j);
                        }
                        db.set_value_unchecked(i, 0, s);
                    }
                    b->accumulate_grad(db);
                }
                
                if (in && in->requires_grad && g) {
                    for (std::size_t i = 0; i < num_features; ++i) {
                        double g_val = g->data.at_unchecked(i, 0);
                        for (std::size_t j = 0; j < batch_size; ++j) {
                            go.set_value_unchecked(i, j, go.at_unchecked(i, j) * g_val);
                        }
                    }
                }
            }
            
            if (in && in->requires_grad) {
                Matrix dx(num_features, batch_size);
                if (training) {
                    for (std::size_t i = 0; i < num_features; ++i) {
                        double sum_go = 0.0;
                        double sum_go_norm = 0.0;
                        for (std::size_t j = 0; j < batch_size; ++j) {
                            sum_go += go.at_unchecked(i, j);
                            sum_go_norm += go.at_unchecked(i, j) * normalized.at_unchecked(i, j);
                        }
                        double mean_go = sum_go / batch_size;
                        double mean_go_norm = sum_go_norm / batch_size;
                        for (std::size_t j = 0; j < batch_size; ++j) {
                            double val = inv_std[i] * (go.at_unchecked(i, j) - mean_go - normalized.at_unchecked(i, j) * mean_go_norm);
                            dx.set_value_unchecked(i, j, val);
                        }
                    }
                } else {
                    for (std::size_t i = 0; i < num_features; ++i) {
                        for (std::size_t j = 0; j < batch_size; ++j) {
                            dx.set_value_unchecked(i, j, go.at_unchecked(i, j) * inv_std[i]);
                        }
                    }
                }
                in->accumulate_grad(dx);
            }
        };
    }
    
    return Tensor(out_node);
}

// LayerNorm
inline Tensor layernorm(const Tensor& input, const Tensor& gamma, const Tensor& beta, double eps) {
    bool req_grad = input.requires_grad() || gamma.requires_grad() || beta.requires_grad();
    std::size_t feat = input.rows();
    std::size_t batch = input.cols();
    
    Matrix normalized_cache(feat, batch);
    std::vector<double> inv_std_cache(batch);
    Matrix out_data(feat, batch);
    
    for (std::size_t j = 0; j < batch; ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < feat; ++i) {
            sum += input.data().at_unchecked(i, j);
        }
        double mu = sum / feat;
        double var_sum = 0.0;
        for (std::size_t i = 0; i < feat; ++i) {
            double d = input.data().at_unchecked(i, j) - mu;
            var_sum += d * d;
        }
        double inv_std = 1.0 / std::sqrt(var_sum / feat + eps);
        inv_std_cache[j] = inv_std;
        
        for (std::size_t i = 0; i < feat; ++i) {
            double norm = (input.data().at_unchecked(i, j) - mu) * inv_std;
            normalized_cache.set_value_unchecked(i, j, norm);
            out_data.set_value_unchecked(i, j, norm * gamma.data().at_unchecked(i, 0) + beta.data().at_unchecked(i, 0));
        }
    }
    
    auto out_node = std::make_shared<TensorNode>(out_data, req_grad, false);
    if (req_grad) {
        out_node->children.push_back(input.node());
        out_node->children.push_back(gamma.node());
        out_node->children.push_back(beta.node());
        
        auto in_node = input.node();
        auto g_node = gamma.node();
        auto b_node = beta.node();
        
        out_node->backward_op = [out_node_w = std::weak_ptr<TensorNode>(out_node),
                                 in_node_w = std::weak_ptr<TensorNode>(in_node),
                                 g_node_w = std::weak_ptr<TensorNode>(g_node),
                                 b_node_w = std::weak_ptr<TensorNode>(b_node),
                                 normalized_cache = std::move(normalized_cache),
                                 inv_std_cache = std::move(inv_std_cache),
                                 feat, batch]() {
            auto out = out_node_w.lock();
            auto in = in_node_w.lock();
            auto g = g_node_w.lock();
            auto b = b_node_w.lock();
            if (!out || !out->grad) return;
            
            Matrix go = *(out->grad);
            
            if (g && g->requires_grad) {
                Matrix dg(feat, 1, 0.0);
                for (std::size_t i = 0; i < feat; ++i) {
                    double s = 0.0;
                    for (std::size_t j = 0; j < batch; ++j) {
                        s += go.at_unchecked(i, j) * normalized_cache.at_unchecked(i, j);
                    }
                    dg.set_value_unchecked(i, 0, s);
                }
                g->accumulate_grad(dg);
            }
            if (b && b->requires_grad) {
                Matrix db(feat, 1, 0.0);
                for (std::size_t i = 0; i < feat; ++i) {
                    double s = 0.0;
                    for (std::size_t j = 0; j < batch; ++j) {
                        s += go.at_unchecked(i, j);
                    }
                    db.set_value_unchecked(i, 0, s);
                }
                b->accumulate_grad(db);
            }
            
            if (in && in->requires_grad && g) {
                Matrix dx(feat, batch);
                for (std::size_t j = 0; j < batch; ++j) {
                    double inv_std = inv_std_cache[j];
                    double sum_dx_hat = 0.0;
                    double sum_dx_hat_x_hat = 0.0;
                    
                    for (std::size_t i = 0; i < feat; ++i) {
                        double dx_hat = go.at_unchecked(i, j) * g->data.at_unchecked(i, 0);
                        sum_dx_hat += dx_hat;
                        sum_dx_hat_x_hat += dx_hat * normalized_cache.at_unchecked(i, j);
                    }
                    
                    double mean_dx_hat = sum_dx_hat / feat;
                    double mean_dx_hat_x_hat = sum_dx_hat_x_hat / feat;
                    
                    for (std::size_t i = 0; i < feat; ++i) {
                        double dx_hat = go.at_unchecked(i, j) * g->data.at_unchecked(i, 0);
                        double val = inv_std * (dx_hat - mean_dx_hat - normalized_cache.at_unchecked(i, j) * mean_dx_hat_x_hat);
                        dx.set_value_unchecked(i, j, val);
                    }
                }
                in->accumulate_grad(dx);
            }
        };
    }
    
    return Tensor(out_node);
}


} // namespace nn
