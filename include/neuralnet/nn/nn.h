#ifndef NN_HPP
#define NN_HPP

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <neuralnet/nn/config.h>
#include <neuralnet/layer.h>
#include <neuralnet/loss.h>
#include <neuralnet/model/model.h>
#include <neuralnet/optimizer.h>

// ── Wave 2 新增模块（必须在 model.h 之后以提供 Model::summary 实现） ──
#include <neuralnet/dataloader.h>
#include <neuralnet/grad_clip.h>
#include <neuralnet/summary.h>

namespace nn
{
    [[nodiscard]] inline Matrix one_hot(const std::vector<std::size_t> &true_i, std::size_t mat_size)
    {
        const std::size_t batch_size = true_i.size();
        Matrix result(mat_size, batch_size);

        for (std::size_t i = 0; i < batch_size; ++i)
        {
            if (true_i[i] >= mat_size)
            {
                throw std::out_of_range("one_hot index out of range");
            }
            result.set_value_unchecked(true_i[i], i, 1.0);
        }

        return result;
    }
} // namespace nn

#endif // NN_HPP