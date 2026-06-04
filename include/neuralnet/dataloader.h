#ifndef DATALOADER_HPP
#define DATALOADER_HPP

// ── DataLoader + Dataset (F1) ────────────────────────────────────────────────
// 内存版数据加载器：抽象 Dataset 接口 + TensorDataset（dense 矩阵版）+ DataLoader 批迭代器。
// 行为对齐 PyTorch DataLoader：可选 shuffle / drop_last；用 std::mt19937 做随机洗牌。
// 输入输出约定：features_ shape = (feature_dim, N)，labels_ shape = (label_dim, N)。

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <neuralnet/matrix.h>

namespace nn
{
    // ── Dataset 抽象基类 ─────────────────────────────────────────────────────
    class Dataset
    {
    public:
        virtual ~Dataset() = default;
        [[nodiscard]] virtual std::size_t size() const = 0;
        [[nodiscard]] virtual std::pair<Matrix, Matrix> get(std::size_t idx) const = 0;
    };

    // ── TensorDataset：dense 矩阵版数据集 ─────────────────────────────────────
    // features shape (feature_dim, N)，labels shape (label_dim, N)。
    // get(i) 返回第 i 个样本（feature_dim, 1）+ (label_dim, 1)。
    class TensorDataset final : public Dataset
    {
    private:
        Matrix features_;
        Matrix labels_;

    public:
        TensorDataset(Matrix features, Matrix labels)
            : features_(std::move(features)), labels_(std::move(labels)) {}

        [[nodiscard]] std::size_t size() const override
        {
            return features_.cols();
        }

        [[nodiscard]] std::pair<Matrix, Matrix> get(std::size_t idx) const override
        {
            if (idx >= features_.cols())
            {
                throw std::out_of_range("TensorDataset::get index out of range");
            }

            Matrix single_feat(features_.rows(), 1);
            for (std::size_t i = 0; i < features_.rows(); ++i)
            {
                single_feat.set_value_unchecked(i, 0, features_.at_unchecked(i, idx));
            }

            Matrix single_label(labels_.rows(), 1);
            for (std::size_t i = 0; i < labels_.rows(); ++i)
            {
                single_label.set_value_unchecked(i, 0, labels_.at_unchecked(i, idx));
            }

            return {std::move(single_feat), std::move(single_label)};
        }
    };

    // ── DataLoader：批迭代器 ─────────────────────────────────────────────────
    // 用法：`for (auto [x, y] : loader) { ... }`。
    // 实现要点：
    //   * begin() 初始化 indices_ = [0..N-1]，可选 shuffle，重置 pos_=0。
    //   * end() 位置的语义：drop_last ? (N/batch)*batch : N
    //     （用 pos < end_pos 风格的等值比较）。
    //   * operator++ 截断到 N（避免越界访问 ds_），N/batch 满批后停止。
    class DataLoader
    {
    private:
        const Dataset &ds_;
        std::size_t batch_size_;
        bool shuffle_;
        bool drop_last_;
        std::mt19937 rng_;
        std::vector<std::size_t> indices_;
        std::size_t pos_{0};

    public:
        DataLoader(const Dataset &ds, std::size_t batch_size, bool shuffle = false,
                   bool drop_last = false, unsigned seed = 0)
            : ds_(ds), batch_size_(batch_size), shuffle_(shuffle), drop_last_(drop_last),
              rng_(seed != 0 ? seed : std::random_device{}())
        {
            if (batch_size_ == 0)
            {
                throw std::invalid_argument("DataLoader: batch_size must be > 0");
            }
        }

        // ── 嵌套迭代器 ──
        class Iterator
        {
        public:
            DataLoader *loader_;
            std::size_t pos_;

            Iterator(DataLoader *loader, std::size_t pos) : loader_(loader), pos_(pos) {}

            // ++ 截断到 N，避免越界访问
            Iterator &operator++()
            {
                pos_ = std::min(pos_ + loader_->batch_size_, loader_->ds_.size());
                return *this;
            }

            std::pair<Matrix, Matrix> operator*() const
            {
                const std::size_t N = loader_->ds_.size();
                const std::size_t actual_batch =
                    (pos_ < N) ? std::min(loader_->batch_size_, N - pos_) : 0;

                // 用第一个样本确定 feat_dim / label_dim
                std::pair<Matrix, Matrix> first =
                    loader_->ds_.get(loader_->indices_[pos_]);
                const std::size_t feat_dim = first.first.rows();
                const std::size_t label_dim = first.second.rows();

                Matrix batch_x(feat_dim, actual_batch);
                Matrix batch_y(label_dim, actual_batch);

                // 写入第 0 个样本
                for (std::size_t i = 0; i < feat_dim; ++i)
                    batch_x.set_value_unchecked(i, 0, first.first.at_unchecked(i, 0));
                for (std::size_t i = 0; i < label_dim; ++i)
                    batch_y.set_value_unchecked(i, 0, first.second.at_unchecked(i, 0));

                // 写入剩余样本
                for (std::size_t j = 1; j < actual_batch; ++j)
                {
                    auto sample = loader_->ds_.get(loader_->indices_[pos_ + j]);
                    for (std::size_t i = 0; i < feat_dim; ++i)
                    {
                        batch_x.set_value_unchecked(
                            i, j, sample.first.at_unchecked(i, 0));
                    }
                    for (std::size_t i = 0; i < label_dim; ++i)
                    {
                        batch_y.set_value_unchecked(
                            i, j, sample.second.at_unchecked(i, 0));
                    }
                }

                return {std::move(batch_x), std::move(batch_y)};
            }

            bool operator!=(const Iterator &other) const
            {
                return pos_ != other.pos_;
            }
        };

        Iterator begin()
        {
            indices_.resize(ds_.size());
            std::iota(indices_.begin(), indices_.end(), std::size_t{0});
            if (shuffle_)
            {
                std::shuffle(indices_.begin(), indices_.end(), rng_);
            }
            pos_ = 0;
            return Iterator(this, 0);
        }

        Iterator end()
        {
            const std::size_t N = ds_.size();
            const std::size_t end_pos = drop_last_
                ? (N / batch_size_) * batch_size_
                : N;
            return Iterator(this, end_pos);
        }
    };
} // namespace nn

#endif // DATALOADER_HPP
