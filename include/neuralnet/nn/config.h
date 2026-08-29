#ifndef NN_CONFIG_HPP
#define NN_CONFIG_HPP

#include <cstddef>
#include <execution>
#include <iterator>
#include <numeric>

// ── 执行策略 ────────────────────────────────────────────────────────────────
// 默认并行+向量化；编译时可通过 -DNN_EXEC_POLICY=std::execution::seq 覆盖
#ifndef NN_EXEC_POLICY
#define NN_EXEC_POLICY std::execution::par_unseq
#endif

// ── 缓存分块大小 ─────────────────────────────────────────────────────────────
// 64×64×8 字节 = 32 KB，装入多数 CPU 的 L1/L2 缓存
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
    /**
     * @brief C++17-compatible counting iterator (replacement for C++20 std::views::iota).
     *
     * Random access iterator that yields sequential integer values.
     * Used with std::for_each and other parallel algorithms to iterate over index ranges.
     *
     * @tparam Integral Integer type for the counter.
     */
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

        // ── 自适应执行策略（SmartPolicy，对标上游 neuralnet.cpp）──────────
        // std::execution 并行算法的线程池调度开销在中小矩阵上远超并行收益
        //（实测 4096 元素时 par_unseq 比串行慢 ~6x，见 bug.md）。所有逐元素
        // 运算改走下方 nn::for_each / nn::transform / nn::transform_reduce：
        // 工作单元数 ≥ PARALLEL_THRESHOLD 才进入并行（仍受 NN_EXEC_POLICY
        // 编译期覆盖控制），否则使用纯串行循环，零调度开销。
        /**
         * @brief Adaptive execution policy (SmartPolicy) to avoid parallel overhead on small workloads.
         *
         * Thread pool scheduling overhead in std::execution parallel algorithms can exceed parallel gains
         * for small to medium matrices (e.g., ~6x slower for 4096 elements).
         * All element-wise operations use threshold-based dispatch:
         * - work >= PARALLEL_THRESHOLD: parallel execution (par_unseq)
         * - work < PARALLEL_THRESHOLD: serial loop with zero scheduling overhead
         *
         * Threshold can be overridden at compile time via NN_PARALLEL_THRESHOLD macro.
         */
#ifndef NN_PARALLEL_THRESHOLD
#define NN_PARALLEL_THRESHOLD 524288 // 512K 元素；上游 32 核实测此值首次稳定获益
#endif
        inline constexpr std::size_t PARALLEL_THRESHOLD = NN_PARALLEL_THRESHOLD;

        // ── 迭代器版：work = 总工作单元数（通常为元素数） ──────────────────
        /**
         * @brief Adaptive for_each with automatic parallel/serial dispatch.
         *
         * Executes parallel (par_unseq) if work >= PARALLEL_THRESHOLD, otherwise serial.
         * Avoids thread pool overhead for small workloads.
         *
         * @param work Total work units (typically element count).
         * @param first Beginning of the input range.
         * @param last End of the input range.
         * @param f Unary function to apply to each element.
         */
        template <typename Iter, typename Fn>
        inline void for_each(std::size_t work, Iter first, Iter last, Fn f)
        {
            if (work >= PARALLEL_THRESHOLD)
                std::for_each(NN_EXEC_POLICY, first, last, std::move(f));
            else
                for (; first != last; ++first)
                    f(*first);
        }

        /**
         * @brief Adaptive unary transform with automatic parallel/serial dispatch.
         *
         * @param work Total work units (typically element count).
         * @param first Beginning of the input range.
         * @param last End of the input range.
         * @param d_first Beginning of the output range.
         * @param f Unary transformation function.
         */
        template <typename InIter, typename OutIter, typename Fn>
        inline void transform(std::size_t work, InIter first, InIter last,
                              OutIter d_first, Fn f)
        {
            if (work >= PARALLEL_THRESHOLD)
                std::transform(NN_EXEC_POLICY, first, last, d_first, std::move(f));
            else
                for (; first != last; ++first, ++d_first)
                    *d_first = f(*first);
        }

        /**
         * @brief Adaptive binary transform with automatic parallel/serial dispatch.
         *
         * @param work Total work units (typically element count).
         * @param first1 Beginning of the first input range.
         * @param last1 End of the first input range.
         * @param first2 Beginning of the second input range.
         * @param d_first Beginning of the output range.
         * @param f Binary transformation function.
         */
        template <typename InIter1, typename InIter2, typename OutIter, typename Fn>
        inline void transform(std::size_t work, InIter1 first1, InIter1 last1,
                              InIter2 first2, OutIter d_first, Fn f)
        {
            if (work >= PARALLEL_THRESHOLD)
                std::transform(NN_EXEC_POLICY, first1, last1, first2, d_first, std::move(f));
            else
                for (; first1 != last1; ++first1, ++first2, ++d_first)
                    *d_first = f(*first1, *first2);
        }

        /**
         * @brief Adaptive unary transform_reduce with automatic parallel/serial dispatch.
         *
         * @param work Total work units (typically element count).
         * @param first Beginning of the input range.
         * @param last End of the input range.
         * @param init Initial value for reduction.
         * @param reduce Binary reduction function.
         * @param transform Unary transformation function applied before reduction.
         * @return Reduced result.
         */
        template <typename Iter, typename T, typename Reduce, typename Transform>
        inline T transform_reduce(std::size_t work, Iter first, Iter last,
                                  T init, Reduce reduce, Transform transform)
        {
            if (work >= PARALLEL_THRESHOLD)
                return std::transform_reduce(NN_EXEC_POLICY, first, last, init, reduce, transform);
            for (; first != last; ++first)
                init = reduce(init, transform(*first));
            return init;
        }

        /**
         * @brief Adaptive binary transform_reduce with automatic parallel/serial dispatch.
         *
         * @param work Total work units (typically element count).
         * @param first1 Beginning of the first input range.
         * @param last1 End of the first input range.
         * @param first2 Beginning of the second input range.
         * @param init Initial value for reduction.
         * @param reduce Binary reduction function.
         * @param transform Binary transformation function applied before reduction.
         * @return Reduced result.
         */
        template <typename InIter1, typename InIter2, typename T, typename Reduce, typename Transform>
        inline T transform_reduce(std::size_t work, InIter1 first1, InIter1 last1,
                                  InIter2 first2, T init, Reduce reduce, Transform transform)
        {
            if (work >= PARALLEL_THRESHOLD)
                return std::transform_reduce(NN_EXEC_POLICY, first1, last1, first2,
                                             init, reduce, transform);
            for (; first1 != last1; ++first1, ++first2)
                init = reduce(init, transform(*first1, *first2));
            return init;
        }

        /**
         * @brief Adaptive reduce (sum) with automatic parallel/serial dispatch.
         *
         * @param work Total work units (typically element count).
         * @param first Beginning of the input range.
         * @param last End of the input range.
         * @param init Initial value for reduction.
         * @return Reduced result using addition.
         */
        template <typename Iter, typename T>
        inline T reduce(std::size_t work, Iter first, Iter last, T init)
        {
            if (work >= PARALLEL_THRESHOLD)
                return std::reduce(NN_EXEC_POLICY, first, last, init);
            for (; first != last; ++first)
                init = init + *first;
            return init;
        }

        /**
         * @brief Adaptive reduce with custom binary operation and automatic parallel/serial dispatch.
         *
         * @param work Total work units (typically element count).
         * @param first Beginning of the input range.
         * @param last End of the input range.
         * @param init Initial value for reduction.
         * @param binop Binary operation for reduction.
         * @return Reduced result.
         */
        template <typename Iter, typename T, typename BinOp>
        inline T reduce(std::size_t work, Iter first, Iter last, T init, BinOp binop)
        {
            if (work >= PARALLEL_THRESHOLD)
                return std::reduce(NN_EXEC_POLICY, first, last, init, binop);
            for (; first != last; ++first)
                init = binop(init, *first);
            return init;
        }

        // ── 下标版：等价于旧式 counting_iterator 循环 ────────────────────
        // 调用方声明的 work 是总工作量（如逐元素规模），不一定等于迭代次数 n。
        /**
         * @brief Adaptive indexed loop with automatic parallel/serial dispatch.
         *
         * Executes f(i) for i in [0, n). Dispatches to parallel or serial based on work threshold.
         *
         * @param work Total work units (element-wise scale, may differ from n).
         * @param n Number of iterations.
         * @param f Function taking index i as parameter.
         */
        template <typename Fn>
        inline void for_range(std::size_t work, std::size_t n, Fn f)
        {
            if (work >= PARALLEL_THRESHOLD)
                std::for_each(NN_EXEC_POLICY,
                              counting_iterator<std::size_t>(0),
                              counting_iterator<std::size_t>(n), std::move(f));
            else
                for (std::size_t i = 0; i < n; ++i)
                    f(i);
        }

        // ── GEMM/转置块循环专用（对标上游 parallel_for_blocks）──────────
        // 每个迭代处理一个 64×64 输出块（≥ BLOCK_SIZE² × K 次乘加），
        // 计算密度高，恒定并行，不适用元素数阈值。
        /**
         * @brief Parallel block loop for GEMM and transpose operations.
         *
         * Always executes in parallel (no threshold check). Each iteration processes a BLOCK_SIZE x BLOCK_SIZE
         * tile with high computational density (>= BLOCK_SIZE^2 * K multiply-adds).
         * Used for cache-blocked matrix operations.
         *
         * @param n_blocks Number of blocks to process.
         * @param f Function taking block index as parameter.
         */
        template <typename Fn>
        inline void for_blocks(std::size_t n_blocks, Fn f)
        {
            std::for_each(NN_EXEC_POLICY,
                          counting_iterator<std::size_t>(0),
                          counting_iterator<std::size_t>(n_blocks), std::move(f));
        }

} // namespace nn

#endif // NN_CONFIG_HPP
