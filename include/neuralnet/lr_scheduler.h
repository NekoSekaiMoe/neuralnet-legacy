#ifndef LR_SCHEDULER_HPP
#define LR_SCHEDULER_HPP

#include <cmath>
#include <cstddef>
#include <vector>

namespace nn
{
    // ── 学习率调度器基类 ──────────────────────────────────────────────────────
    class LRScheduler
    {
    public:
        virtual ~LRScheduler() = default;
        virtual double get_lr() const = 0;
        virtual void step() = 0;
    };

    // ── StepLR：每 step_size 个 epoch 乘以 gamma ──────────────────────────────
    class StepLR : public LRScheduler
    {
    private:
        double lr_;
        std::size_t step_size_;
        double gamma_;
        std::size_t epoch_count_;

    public:
        StepLR(double lr, std::size_t step_size, double gamma = 0.1)
            : lr_(lr), step_size_(step_size), gamma_(gamma), epoch_count_(0) {}

        double get_lr() const override
        {
            const std::size_t num_decays = epoch_count_ / step_size_;
            return lr_ * std::pow(gamma_, static_cast<double>(num_decays));
        }

        void step() override { ++epoch_count_; }
    };

    // ── CosineAnnealingLR：余弦退火 ──────────────────────────────────────────
    // lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * epoch / T_max))
    class CosineAnnealingLR : public LRScheduler
    {
    private:
        double lr_max_;
        double lr_min_;
        std::size_t T_max_;
        std::size_t epoch_count_;

    public:
        CosineAnnealingLR(double lr_max, std::size_t T_max, double lr_min = 0.0)
            : lr_max_(lr_max), lr_min_(lr_min), T_max_(T_max), epoch_count_(0) {}

        double get_lr() const override
        {
            const double progress = static_cast<double>(epoch_count_) / static_cast<double>(T_max_);
            return lr_min_ + 0.5 * (lr_max_ - lr_min_) * (1.0 + std::cos(M_PI * progress));
        }

        void step() override { ++epoch_count_; }
    };

    // ── ExponentialLR：每个 epoch 乘以 gamma ─────────────────────────────────
    class ExponentialLR : public LRScheduler
    {
    private:
        double lr_;
        double gamma_;
        std::size_t epoch_count_;

    public:
        ExponentialLR(double lr, double gamma)
            : lr_(lr), gamma_(gamma), epoch_count_(0) {}

        double get_lr() const override
        {
            return lr_ * std::pow(gamma_, static_cast<double>(epoch_count_));
        }

        void step() override { ++epoch_count_; }
    };

} // namespace nn

#endif // LR_SCHEDULER_HPP
