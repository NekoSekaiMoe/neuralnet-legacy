#ifndef MODEL_IO_HPP
#define MODEL_IO_HPP

#include <neuralnet/model/model.h>

namespace nn
{
    // ── 便捷自由函数（Model 版本）──────────────────────────────────────────────
    // 实际实现委托给 Model::save / Model::load，避免代码重复。
    inline void save_model(const std::string &filename, Model &model)
    {
        model.save(filename);
    }

    inline void load_model(const std::string &filename, Model &model)
    {
        model.load(filename);
    }

} // namespace nn

#endif // MODEL_IO_HPP
