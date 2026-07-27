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

    inline ModelSpec read_model_spec(const std::string &filename)
    {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs)
        {
            throw std::runtime_error("Cannot read model file: " + filename);
        }
        uint32_t magic, version;
        ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        ifs.read(reinterpret_cast<char *>(&version), sizeof(version));
        if (magic != 0x4E4E4E4E)
        {
            throw std::runtime_error("Invalid model file format");
        }
        if (version < 3)
        {
            ModelSpec spec;
            spec.type = ModelType::Sequential;
            return spec;
        }

        ModelSpec spec;
        ifs.read(reinterpret_cast<char *>(&spec.type), sizeof(spec.type));
        ifs.read(reinterpret_cast<char *>(&spec.vocab_size), sizeof(std::size_t));
        ifs.read(reinterpret_cast<char *>(&spec.d_model), sizeof(std::size_t));
        ifs.read(reinterpret_cast<char *>(&spec.seq_len), sizeof(std::size_t));
        ifs.read(reinterpret_cast<char *>(&spec.num_heads), sizeof(std::size_t));
        ifs.read(reinterpret_cast<char *>(&spec.d_ff), sizeof(std::size_t));
        ifs.read(reinterpret_cast<char *>(&spec.num_layers), sizeof(std::size_t));
        return spec;
    }

    inline Model build_gpt_model(
        std::size_t vocab_size,
        std::size_t d_model,
        std::size_t seq_len,
        std::size_t num_heads,
        std::size_t d_ff,
        std::size_t num_layers)
    {
        Model model;
        model.add<GPTModel>(vocab_size, d_model, seq_len, num_heads, d_ff, num_layers);
        return model;
    }

    inline Model build_gpt_model_from_spec(const ModelSpec &spec)
    {
        if (spec.type != ModelType::GPT)
            throw std::runtime_error("Invalid ModelSpec type for GPT");

        return build_gpt_model(
            spec.vocab_size, spec.d_model, spec.seq_len,
            spec.num_heads, spec.d_ff, spec.num_layers);
    }

    inline ModelSpec make_gpt_spec(
        std::size_t vocab_size,
        std::size_t d_model,
        std::size_t seq_len,
        std::size_t num_heads,
        std::size_t d_ff,
        std::size_t num_layers)
    {
        ModelSpec spec;
        spec.type       = ModelType::GPT;
        spec.vocab_size = vocab_size;
        spec.d_model    = d_model;
        spec.seq_len    = seq_len;
        spec.num_heads  = num_heads;
        spec.d_ff       = d_ff;
        spec.num_layers = num_layers;
        return spec;
    }

} // namespace nn

#endif // MODEL_IO_HPP
