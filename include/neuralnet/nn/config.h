#ifndef NN_CONFIG_HPP
#define NN_CONFIG_HPP

#include <cstddef>
#include <execution>
#include <iterator>

// ── 执行策略 ────────────────────────────────────────────────────────────────
// 默认并行+向量化；编译时可通过 -DNN_EXEC_POLICY=std::execution::seq 覆盖
#ifndef NN_EXEC_POLICY
#define NN_EXEC_POLICY std::execution::par_unseq
#endif

// ── 缓存分块大小 ─────────────────────────────────────────────────────────────
// 32×32×8 = 8 KB，安全装入大多数 CPU 的 L1 缓存
// 矩阵乘法、转置等所有分块操作共用此值
// 修改时需同步评估 b_block 栈占用（BLOCK_SIZE² × 8 字节）
namespace nn
{
    inline constexpr std::size_t BLOCK_SIZE = 64;

    // ── 数值常量 ─────────────────────────────────────────────────────────────
    // 使用 constexpr 避免运行时计算
    inline constexpr double EPSILON = 1e-8;

    // ── C++17 兼容的 counting_iterator ──────────────────────────────────────
    // 替代 C++20 std::views::iota，用于 std::for_each 等并行算法
    template <typename Integral>
    class counting_iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = Integral;
        using difference_type = std::ptrdiff_t;
        using pointer = const Integral *;
        using reference = Integral;

        constexpr explicit counting_iterator(Integral value) noexcept : value_(value) {}

        // InputIterator
        constexpr reference operator*() const noexcept { return value_; }
        constexpr counting_iterator &operator++() noexcept
        {
            ++value_;
            return *this;
        }
        constexpr counting_iterator operator++(int) noexcept
        {
            auto tmp = *this;
            ++value_;
            return tmp;
        }

        // ForwardIterator
        constexpr bool operator==(const counting_iterator &other) const noexcept
        {
            return value_ == other.value_;
        }
        constexpr bool operator!=(const counting_iterator &other) const noexcept
        {
            return value_ != other.value_;
        }

        // BidirectionalIterator
        constexpr counting_iterator &operator--() noexcept
        {
            --value_;
            return *this;
        }
        constexpr counting_iterator operator--(int) noexcept
        {
            auto tmp = *this;
            --value_;
            return tmp;
        }

        // RandomAccessIterator
        constexpr counting_iterator &operator+=(difference_type n) noexcept
        {
            value_ += static_cast<Integral>(n);
            return *this;
        }
        constexpr counting_iterator &operator-=(difference_type n) noexcept
        {
            value_ -= static_cast<Integral>(n);
            return *this;
        }
        constexpr counting_iterator operator+(difference_type n) const noexcept
        {
            return counting_iterator(value_ + static_cast<Integral>(n));
        }
        constexpr counting_iterator operator-(difference_type n) const noexcept
        {
            return counting_iterator(value_ - static_cast<Integral>(n));
        }
        constexpr difference_type operator-(const counting_iterator &other) const noexcept
        {
            return static_cast<difference_type>(value_) - static_cast<difference_type>(other.value_);
        }
        constexpr reference operator[](difference_type n) const noexcept
        {
            return value_ + static_cast<Integral>(n);
        }
        constexpr bool operator<(const counting_iterator &other) const noexcept
        {
            return value_ < other.value_;
        }
        constexpr bool operator>(const counting_iterator &other) const noexcept
        {
            return value_ > other.value_;
        }
        constexpr bool operator<=(const counting_iterator &other) const noexcept
        {
            return value_ <= other.value_;
        }
        constexpr bool operator>=(const counting_iterator &other) const noexcept
        {
            return value_ >= other.value_;
        }

    private:
        Integral value_;
    };

    // 非成员 operator+ (n + it)
    template <typename Integral>
    constexpr counting_iterator<Integral> operator+(
        typename counting_iterator<Integral>::difference_type n,
        const counting_iterator<Integral> &it) noexcept
    {
        return it + n;
    }

} // namespace nn

#endif // NN_CONFIG_HPP
