#ifndef NN_THREADPOOL_HPP
#define NN_THREADPOOL_HPP

// ══════════════════════════════════════════════════════════════════════════
//  threadpool.h — 自包含线程池（移植自上游 core_threadpool 的调度策略）
//
//  动机：std::execution::par_unseq 需要 TBB 运行时（libtbb-dev）依赖；
//  本头文件提供等价的并行原语，编译 -DNN_EXEC_POOL 后 nn::for_each /
//  nn::transform / nn::for_blocks 等全部切到本池，摆脱外部依赖。
//
//  调度策略（与上游一致）：
//    · 固定 worker + 原子块抢取：任务切成 ~4×线程数的块，worker 通过
//      fetch_add 抢块执行，天然负载均衡（快块做完抢慢块的活）；
//    · 调用线程也参与执行（不空等），少量块时近似串行开销；
//    · 队列支持多个并发 parallel_* 调用（FIFO 串行化，不嵌套不死锁）；
//    · 归约类原语按块部分归约 + 投递线程按块序串行合并 → 确定性结果
//      （比 TBB 的非确定归约树更可复现）。
// ══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace nn
{

    class ThreadPool
    {
    private:
        // 一次 parallel_* 调用的共享状态（存活于调用方栈上）。
        // invoke(start, end) 执行 [start, end) 区间；next 为抢块游标；
        // remaining 为未完成块数，归零后任务结束。
        struct Job
        {
            std::atomic<std::size_t> next{0};
            std::atomic<std::size_t> remaining{0};
            std::size_t n_chunks = 0;
            std::size_t chunk = 1;
            std::size_t n = 0;
            std::function<void(std::size_t, std::size_t)> invoke;
            std::atomic<bool> error_set{false}; // 仅首个异常写入者胜出
            std::exception_ptr error{nullptr};
        };

        std::vector<std::thread> workers_;
        std::vector<std::shared_ptr<Job>> queue_;
        std::mutex mtx_;
        std::condition_variable cv_;
        bool stop_ = false;

        static std::size_t default_thread_count() noexcept
        {
            const unsigned hw = std::thread::hardware_concurrency();
            return hw == 0 ? 1u : static_cast<std::size_t>(hw);
        }

        // 执行一个 Job 的所有剩余块（worker 与投递线程共用）。
        void run_chunks_(Job &job)
        {
            for (;;)
            {
                const std::size_t c = job.next.fetch_add(1, std::memory_order_relaxed);
                if (c >= job.n_chunks)
                    return;
                const std::size_t start = c * job.chunk;
                const std::size_t end = std::min(start + job.chunk, job.n);
                try
                {
                    job.invoke(start, end);
                }
                catch (...)
                {
                    bool expected = false;
                    if (job.error_set.compare_exchange_strong(expected, true))
                        job.error = std::current_exception();
                    // 让其他块快速收敛（错误优先抛出）
                    job.next.store(job.n_chunks, std::memory_order_relaxed);
                }
                job.remaining.fetch_sub(1, std::memory_order_acq_rel);
            }
        }

        void worker_loop_()
        {
            for (;;)
            {
                std::shared_ptr<Job> job;
                {
                    std::unique_lock<std::mutex> lock(mtx_);
                    cv_.wait(lock, [this]
                             { return stop_ || !queue_.empty(); });
                    if (stop_ && queue_.empty())
                        return;
                    job = queue_.front(); // shared_ptr 拷贝：任务对象生命周期安全
                    if (job->remaining.load(std::memory_order_acquire) == 0)
                    {
                        // 已完成的头部任务：代投递者清理（其出队时可能不在队头）
                        queue_.erase(queue_.begin());
                        cv_.notify_all();
                        continue;
                    }
                }
                run_chunks_(*job);
            }
        }

        explicit ThreadPool(std::size_t threads)
        {
            if (threads == 0)
                threads = 1;
            // 调用线程充当一个 worker，避免空转等待
            for (std::size_t i = 0; i + 1 < threads; ++i)
                workers_.emplace_back([this] { worker_loop_(); });
        }

        // 入队 → 本线程参与执行 → 等待完成 → 出队（若在队头）→ 重抛异常。
        // job 以 shared_ptr 持有：即使本函数返回后任务仍在队列里（非队头），
        // worker 持有的引用也保持对象存活。
        void dispatch_(std::shared_ptr<Job> job)
        {
            {
                std::lock_guard<std::mutex> lock(mtx_);
                queue_.push_back(job);
            }
            cv_.notify_all();

            run_chunks_(*job);

            while (job->remaining.load(std::memory_order_acquire) != 0)
                std::this_thread::yield();

            {
                std::lock_guard<std::mutex> lock(mtx_);
                if (!queue_.empty() && queue_.front() == job)
                    queue_.erase(queue_.begin());
                // 非队头（前面还有别人的任务）：留给 worker 清理路径
            }
            cv_.notify_all();

            if (job->error)
                std::rethrow_exception(job->error);
        }

        [[nodiscard]] std::size_t chunk_size_(std::size_t n) const noexcept
        {
            const std::size_t target = num_workers() * 4;
            return std::max<std::size_t>(1, (n + target - 1) / target);
        }

        // 通用分块归约：local_fn(start, end) 串行计算块内部分和，
        // combine 按块序合并 partials（确定性）。
        template <typename T, typename LocalFn, typename CombineFn>
        T reduce_chunks_(std::size_t n, T init, LocalFn &&local_fn, CombineFn &&combine)
        {
            const std::size_t chunk = chunk_size_(n);
            const std::size_t n_chunks = (n + chunk - 1) / chunk;
            auto partials = std::make_shared<std::vector<T>>(n_chunks);

            auto job = std::make_shared<Job>();
            job->n = n;
            job->chunk = chunk;
            job->n_chunks = n_chunks;
            job->remaining.store(n_chunks, std::memory_order_relaxed);
            job->invoke = [&local_fn, partials, chunk](std::size_t start, std::size_t end)
            { (*partials)[start / chunk] = local_fn(start, end); };

            dispatch_(std::move(job));

            for (std::size_t c = 0; c < n_chunks; ++c)
                init = combine(init, (*partials)[c]);
            return init;
        }

    public:
        ThreadPool(const ThreadPool &) = delete;
        ThreadPool &operator=(const ThreadPool &) = delete;

        // 进程级全局池（惰性初始化，首次使用时创建，退出时自动 join）。
        static ThreadPool &global()
        {
            static ThreadPool pool(default_thread_count());
            return pool;
        }

        ~ThreadPool()
        {
            {
                std::lock_guard<std::mutex> lock(mtx_);
                stop_ = true;
            }
            cv_.notify_all();
            for (auto &t : workers_)
                if (t.joinable())
                    t.join();
        }

        [[nodiscard]] std::size_t num_workers() const noexcept
        {
            return workers_.size() + 1;
        }

        /**
         * @brief Executes f(i) for every i in [0, n) across all workers.
         *
         * The range is split into ~4 chunks per worker; chunks are grabbed via
         * an atomic cursor (dynamic load balancing). The calling thread also
         * executes chunks. The first exception thrown by f is re-thrown to the
         * caller after all chunks complete.
         */
        template <typename Fn>
        void parallel_for(std::size_t n, Fn &&f)
        {
            if (n == 0)
                return;
            if (workers_.empty() || n == 1)
            {
                for (std::size_t i = 0; i < n; ++i)
                    f(i);
                return;
            }

            const std::size_t chunk = chunk_size_(n);
            auto job = std::make_shared<Job>();
            job->n = n;
            job->chunk = chunk;
            job->n_chunks = (n + chunk - 1) / chunk;
            job->remaining.store(job->n_chunks, std::memory_order_relaxed);
            job->invoke = [&f](std::size_t start, std::size_t end)
            {
                for (std::size_t i = start; i < end; ++i)
                    f(i);
            };

            dispatch_(std::move(job));
        }

        /**
         * @brief Parallel unary transform: d_first[i] = op(first[i]) for i in [0, n).
         */
        template <typename InIter, typename OutIter, typename UnaryOp>
        void parallel_transform(std::size_t n, InIter first, OutIter d_first, UnaryOp op)
        {
            parallel_for(n, [&](std::size_t i) { d_first[i] = op(first[i]); });
        }

        /**
         * @brief Parallel binary transform:
         *        d_first[i] = op(first1[i], first2[i]) for i in [0, n).
         */
        template <typename InIter1, typename InIter2, typename OutIter, typename BinaryOp>
        void parallel_transform(std::size_t n, InIter1 first1, InIter2 first2,
                                OutIter d_first, BinaryOp op)
        {
            parallel_for(n, [&](std::size_t i) { d_first[i] = op(first1[i], first2[i]); });
        }

        /**
         * @brief Unary transform-reduce over [0, n): per-chunk partials are
         *        combined in chunk order (deterministic).
         */
        template <typename InIter, typename T, typename Reduce, typename Transform>
        T parallel_transform_reduce(std::size_t n, InIter first, T init,
                                    Reduce reduce, Transform transform)
        {
            if (n == 0)
                return init;
            return reduce_chunks_<T>(
                n, std::move(init),
                [&](std::size_t start, std::size_t end)
                {
                    T local = transform(first[start]);
                    for (std::size_t i = start + 1; i < end; ++i)
                        local = reduce(local, transform(first[i]));
                    return local;
                },
                reduce);
        }

        /**
         * @brief Binary transform-reduce over [0, n):
         *        partials combined in chunk order (deterministic).
         */
        template <typename InIter1, typename InIter2, typename T, typename Reduce, typename Transform>
        T parallel_transform_reduce2(std::size_t n, InIter1 first1, InIter2 first2,
                                     T init, Reduce reduce, Transform transform)
        {
            if (n == 0)
                return init;
            return reduce_chunks_<T>(
                n, std::move(init),
                [&](std::size_t start, std::size_t end)
                {
                    T local = transform(first1[start], first2[start]);
                    for (std::size_t i = start + 1; i < end; ++i)
                        local = reduce(local, transform(first1[i], first2[i]));
                    return local;
                },
                reduce);
        }

        /**
         * @brief Parallel reduce (sum with custom op) over [0, n).
         */
        template <typename InIter, typename T, typename Reduce>
        T parallel_reduce(std::size_t n, InIter first, T init, Reduce reduce)
        {
            if (n == 0)
                return init;
            return reduce_chunks_<T>(
                n, std::move(init),
                [&](std::size_t start, std::size_t end)
                {
                    T local = first[start];
                    for (std::size_t i = start + 1; i < end; ++i)
                        local = reduce(local, first[i]);
                    return local;
                },
                reduce);
        }
    };

} // namespace nn

#endif // NN_THREADPOOL_HPP
