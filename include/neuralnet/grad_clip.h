#ifndef GRAD_CLIP_HPP
#define GRAD_CLIP_HPP

// ── clip_grad_norm_ ───────────────────────────────────────────────────────────
// F3: 梯度范数裁剪（in-place，按 max_norm 缩放）
// PyTorch 风格：若 ||g||_2 > max_norm，则 g *= max_norm / (||g||_2 + eps)。
// 返回 pre-clip 总范数（不缩放后的值）。

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <execution>
#include <functional>
#include <vector>

#include <neuralnet/matrix.h>
#include <neuralnet/nn/config.h>

namespace nn
{
    inline double clip_grad_norm_(
        std::vector<std::reference_wrapper<Matrix>> grads,
        double max_norm,
        double eps = 1e-6)
    {
        // ── 计算 pre-clip 总范数 ──
        // sum(grad.norm()^2) over all grads
        const double total_sq = nn::transform_reduce(
            grads.size(), // 迭代数=参数个数；每次迭代的 norm() 内部再自适应
            grads.begin(), grads.end(),
            0.0,
            std::plus<>{},
            [](const std::reference_wrapper<Matrix> &g_ref) noexcept {
                return g_ref.get().norm() * g_ref.get().norm();
            });

        const double total_norm = std::sqrt(total_sq);

        // 非有限 total_norm 抛错（NaN 静默传播 / Inf 把梯度归零），与 PyTorch 对齐
        if (!std::isfinite(total_norm))
        {
            throw std::runtime_error("clip_grad_norm_: gradient contains NaN or Inf");
        }

        // ── 若超阈值，缩放所有梯度 ──
        if (total_norm > max_norm && total_sq > 0.0)
        {
            const double clip_coef = max_norm / (total_norm + eps);
            for (auto &g_ref : grads)
            {
                Matrix &g = g_ref.get();
                nn::transform(g.size(),
                               g.data().begin(), g.data().end(),
                               g.data().begin(),
                               [clip_coef](double v) noexcept { return v * clip_coef; });
            }
        }

        return total_norm;
    }
} // namespace nn

#endif // GRAD_CLIP_HPP
