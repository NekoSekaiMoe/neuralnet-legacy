#ifndef MODEL_HPP
#define MODEL_HPP

#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <neuralnet/nn/config.h>
#include <neuralnet/keyvalue_record.h>
#include <neuralnet/layer.h>
#include <optional>

namespace nn
{
    enum class ModelType : uint32_t {
        Unknown = 0,
        Sequential = 1,
        GPT = 2
    };

    struct ModelSpec {
        ModelType type = ModelType::Unknown;
        std::size_t vocab_size = 0;
        std::size_t d_model = 0;
        std::size_t seq_len = 0;
        std::size_t num_heads = 0;
        std::size_t d_ff = 0;
        std::size_t num_layers = 0;
    };
    class Model
    {
    private:
        std::vector<std::unique_ptr<Layer>> layers_;
        bool is_training_{true};

    public:
        Model() = default;

        // 不可拷贝（unique_ptr 语义），只能移动
        Model(const Model &) = delete;
        Model &operator=(const Model &) = delete;
        Model(Model &&) noexcept = default;
        Model &operator=(Model &&) noexcept = default;

        // ── 构建网络 ────────────────────────────────────────────────────────
        // 用法：model.add<nn::Linear>(784, 64).add<nn::ReLU>()...
        template <typename LayerType, typename... Args>
        Model &add(Args &&...args)
        {
            layers_.emplace_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
            return *this;
        }

        [[nodiscard]] std::size_t num_layers() const noexcept { return layers_.size(); }

        // ── 层访问 ──
        // 暴露给测试与 summary() 用；唯一_ptr 不可拷贝。
        [[nodiscard]] const std::vector<std::unique_ptr<Layer>> &get_layers() const noexcept
        {
            return layers_;
        }

        [[nodiscard]] Layer& layer_at(std::size_t index)
        {
            if (index >= layers_.size()) throw std::out_of_range("Layer index out of range");
            return *layers_[index];
        }

        // ── summary (F4) ──
        // 完整定义在 <neuralnet/summary.h> 中；本声明使调用点能编译，
        // 链接期由 summary.h 的 inline 定义提供实现。
        // out != nullptr 时同步写入流（不附加换行）。
        [[nodiscard]] std::string summary(std::ostream *out = nullptr) const;

        // ── 模式 (F6) ────────────────────────────────────────────────────────
        // 切换 train/eval 模式，并向每层传播 on_mode_change()。
        // 默认所有层 on_mode_change 是空操作；Dropout / BatchNorm 后续可覆盖。
        void train()
        {
            is_training_ = true;
            for (auto &layer : layers_)
            {
                layer->on_mode_change(true);
            }
        }

        void eval()
        {
            is_training_ = false;
            for (auto &layer : layers_)
            {
                layer->on_mode_change(false);
            }
        }

        [[nodiscard]] bool is_training() const noexcept { return is_training_; }

        // ── 前向传播 ────────────────────────────────────────────────────────
        [[nodiscard]] Matrix forward(const Matrix &input)
        {
            if (layers_.empty())
            {
                throw std::runtime_error("Model has no layers");
            }
            Matrix out = layers_.front()->forward(input);
            for (std::size_t i = 1; i < layers_.size(); ++i)
            {
                out = layers_[i]->forward(out);
            }
            return out;
        }

        // ── 反向传播 ────────────────────────────────────────────────────────
        // 传入 loss 对最后一层输出的梯度，返回对输入的梯度（通常不需要）
        Matrix backward(const Matrix &grad_output)
        {
            if (layers_.empty())
            {
                throw std::runtime_error("Model has no layers");
            }
            Matrix grad = layers_.back()->backward(grad_output);
            for (std::size_t i = layers_.size() - 1; i-- > 0;)
            {
                grad = layers_[i]->backward(grad);
            }
            return grad;
        }

        // ── 参数收集 ────────────────────────────────────────────────────────
        // 聚合所有层的可训练参数，供 Optimizer 使用
        [[nodiscard]] std::vector<std::reference_wrapper<Matrix>> parameters()
        {
            std::vector<std::reference_wrapper<Matrix>> result;
            for (auto &layer : layers_)
            {
                for (auto &p : layer->parameters())
                {
                    result.push_back(p);
                }
            }
            return result;
        }

        [[nodiscard]] std::vector<std::reference_wrapper<Matrix>> param_gradients()
        {
            std::vector<std::reference_wrapper<Matrix>> result;
            for (auto &layer : layers_)
            {
                for (auto &g : layer->param_gradients())
                {
                    result.push_back(g);
                }
            }
            return result;
        }

        // ── I/O ─────────────────────────────────────────────────────────────
        // 文件格式（binary，little-endian）：
        //   magic    (uint32, 4B) = 0x4E4E4E4E
        //   version  (uint32, 4B) = 1 | 2 | 3 | 4
        //
        // v1:
        //   for each parameter Matrix in parameters() order:
        //     size_t rows, cols
        //     rows*cols doubles（行主序，与 Matrix 内存布局一致）
        //
        // v2:
        //   size_t n_params
        //   for i in [0, n_params):
        //     size_t rows, cols
        //     rows*cols doubles
        //   size_t n_state_layers
        //   for each layer in layers_ order where has_state()==true:
        //     layer->save_state(os)
        //
        // v3: v2 前置原始 ModelSpec（type + 6×size_t，结构需与加载方一致）
        //
        // v4（移植自上游 KeyValueRecord，推荐）: 自描述键值格式：
        //   spec 段  : [len u32][KeyValueRecord bytes]（len=0 表示无 spec；
        //              字段 type/vocab_size/d_model/seq_len/num_heads/d_ff/num_layers）
        //   参数段   : 同 v2
        //   状态段   : size_t n_state_layers，每层 [len u32][KeyValueRecord bytes]
        //              （字段名/类型/长度自带，未知字段可跳过 → 向前兼容）
        //
        // 注意：网络结构（各层维度）必须与加载时一致。load() 按版本分发；未知版本抛错。
        // save(format_version=0) 自动选择（有 spec→3，无→2）；传 4 显式写 v4。
        void save(const std::string &filename, const std::optional<ModelSpec>& spec = std::nullopt,
                  uint32_t format_version = 0)
        {
            std::ofstream ofs(filename, std::ios::binary);
            if (!ofs)
            {
                throw std::runtime_error("Cannot write model file: " + filename);
            }
            save(ofs, spec, format_version);
            std::cout << "Model saved to " << filename << std::endl;
        }

        void save(std::ostream &os, const std::optional<ModelSpec>& spec = std::nullopt,
                  uint32_t format_version = 0)
        {
            const uint32_t magic = 0x4E4E4E4E;
            const uint32_t version = format_version != 0
                ? format_version
                : (spec.has_value() ? 3u : 2u);
            // 只接受已知版本；v3 需要 spec（否则下面会解引用空 optional，UB）
            if (version < 1 || version > 4)
                throw std::invalid_argument(
                    "Model::save: unsupported format_version (supported: 1-4)");
            if (version == 3 && !spec.has_value())
                throw std::invalid_argument(
                    "Model::save: format_version 3 requires a ModelSpec");
            os.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
            os.write(reinterpret_cast<const char *>(&version), sizeof(version));

            auto write_record = [&](const KeyValueRecord &rec)
            {
                const std::string bytes = rec.serialize();
                const uint32_t len = static_cast<uint32_t>(bytes.size());
                os.write(reinterpret_cast<const char *>(&len), sizeof(len));
                os.write(bytes.data(), bytes.size());
            };

            if (version == 3)
            {
                ModelType type = spec->type;
                os.write(reinterpret_cast<const char *>(&type), sizeof(type));
                os.write(reinterpret_cast<const char *>(&spec->vocab_size), sizeof(std::size_t));
                os.write(reinterpret_cast<const char *>(&spec->d_model), sizeof(std::size_t));
                os.write(reinterpret_cast<const char *>(&spec->seq_len), sizeof(std::size_t));
                os.write(reinterpret_cast<const char *>(&spec->num_heads), sizeof(std::size_t));
                os.write(reinterpret_cast<const char *>(&spec->d_ff), sizeof(std::size_t));
                os.write(reinterpret_cast<const char *>(&spec->num_layers), sizeof(std::size_t));
            }
            else if (version == 4)
            {
                if (spec.has_value())
                {
                    KeyValueRecord rec;
                    rec.set("type", static_cast<uint64_t>(spec->type))
                        .set("vocab_size", spec->vocab_size)
                        .set("d_model", spec->d_model)
                        .set("seq_len", spec->seq_len)
                        .set("num_heads", spec->num_heads)
                        .set("d_ff", spec->d_ff)
                        .set("num_layers", spec->num_layers);
                    write_record(rec);
                }
                else
                {
                    const uint32_t zero = 0;
                    os.write(reinterpret_cast<const char *>(&zero), sizeof(zero));
                }
            }

            auto write_matrix = [&](const Matrix &m)
            {
                std::size_t rows = m.rows(), cols = m.cols();
                os.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
                os.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
                const auto &data = m.data();
                os.write(reinterpret_cast<const char *>(data.data()),
                          data.size() * sizeof(double));
            };

            // ── 参数段 ──
            auto params = parameters();
            const std::size_t n_params = params.size();
            os.write(reinterpret_cast<const char *>(&n_params), sizeof(n_params));
            for (auto &p_ref : params)
            {
                write_matrix(p_ref.get());
            }

            // ── 状态段 ──
            std::size_t n_state_layers = 0;
            for (const auto &layer : layers_)
            {
                if (layer->has_state()) ++n_state_layers;
            }
            os.write(reinterpret_cast<const char *>(&n_state_layers), sizeof(n_state_layers));
            for (const auto &layer : layers_)
            {
                if (layer->has_state())
                {
                    if (version == 4)
                    {
                        KeyValueRecord rec;
                        layer->save_state_kv(rec);
                        write_record(rec);
                    }
                    else
                    {
                        layer->save_state(os);
                    }
                }
            }
        }

        void load(const std::string &filename)
        {
            std::ifstream ifs(filename, std::ios::binary);
            if (!ifs)
            {
                throw std::runtime_error("Cannot read model file: " + filename);
            }
            load(ifs);
            std::cout << "Model loaded from " << filename << std::endl;
        }

        void load(std::istream &is)
        {
            uint32_t magic, version;
            is.read(reinterpret_cast<char *>(&magic), sizeof(magic));
            is.read(reinterpret_cast<char *>(&version), sizeof(version));
            if (magic != 0x4E4E4E4E)
            {
                throw std::runtime_error("Invalid model file format");
            }
            if (version != 1 && version != 2 && version != 3 && version != 4)
            {
                throw std::runtime_error("Unsupported model file version");
            }

            // v4 记录读取：[len u32][KeyValueRecord bytes]（len=0 → 空记录）。
            // 每次读取后检查流状态，拒绝截断；长度上限防损坏文件触发巨额分配。
            auto read_record = [&]() -> KeyValueRecord
            {
                uint32_t len = 0;
                is.read(reinterpret_cast<char *>(&len), sizeof(len));
                if (!is)
                    throw std::runtime_error("Truncated v4 record length");
                if (len == 0)
                    return KeyValueRecord{};
                constexpr uint32_t kMaxRecordLen = 1u << 28; // 256 MiB
                if (len > kMaxRecordLen)
                    throw std::runtime_error("Implausible v4 record length");
                std::string buf(len, '\0');
                is.read(buf.data(), len);
                if (!is)
                    throw std::runtime_error("Truncated v4 record body");
                return KeyValueRecord::parse(buf);
            };

            if (version == 3)
            {
                ModelSpec spec;
                is.read(reinterpret_cast<char *>(&spec.type), sizeof(spec.type));
                is.read(reinterpret_cast<char *>(&spec.vocab_size), sizeof(std::size_t));
                is.read(reinterpret_cast<char *>(&spec.d_model), sizeof(std::size_t));
                is.read(reinterpret_cast<char *>(&spec.seq_len), sizeof(std::size_t));
                is.read(reinterpret_cast<char *>(&spec.num_heads), sizeof(std::size_t));
                is.read(reinterpret_cast<char *>(&spec.d_ff), sizeof(std::size_t));
                is.read(reinterpret_cast<char *>(&spec.num_layers), sizeof(std::size_t));
            }
            else if (version == 4)
            {
                // 自描述 spec（解析校验其完整性；结构一致性由参数段形状检查保证）
                read_record(); // spec 记录（内容与 read_model_spec 一致）
            }

            auto read_matrix = [&](Matrix &m)
            {
                std::size_t rows, cols;
                is.read(reinterpret_cast<char *>(&rows), sizeof(rows));
                is.read(reinterpret_cast<char *>(&cols), sizeof(cols));
                if (rows != m.rows() || cols != m.cols())
                {
                    throw std::runtime_error("Model shape mismatch in loaded file");
                }
                auto &data = m.data();
                is.read(reinterpret_cast<char *>(data.data()),
                         data.size() * sizeof(double));
            };

            auto params = parameters();

            if (version == 1)
            {
                for (auto &p_ref : params)
                {
                    read_matrix(p_ref.get());
                }
            }
            else
            {
                // ── 参数段 ──
                std::size_t n_params = 0;
                is.read(reinterpret_cast<char *>(&n_params), sizeof(n_params));
                if (n_params != params.size())
                {
                    throw std::runtime_error("Model parameter count mismatch in v2 file");
                }
                for (auto &p_ref : params)
                {
                    read_matrix(p_ref.get());
                }

                // ── 状态段 ──
                std::size_t n_state_layers = 0;
                is.read(reinterpret_cast<char *>(&n_state_layers), sizeof(n_state_layers));
                std::size_t actual_state_layers = 0;
                for (const auto &layer : layers_)
                {
                    if (layer->has_state()) ++actual_state_layers;
                }
                if (n_state_layers != actual_state_layers)
                {
                    throw std::runtime_error(
                        "Model state layer count mismatch in v2 file: file="
                        + std::to_string(n_state_layers)
                        + ", model=" + std::to_string(actual_state_layers));
                }
                for (const auto &layer : layers_)
                {
                    if (layer->has_state())
                    {
                        if (version == 4)
                        {
                            layer->load_state_kv(read_record());
                        }
                        else
                        {
                            layer->load_state(is);
                        }
                    }
                }
            }
        }
    };

} // namespace nn

#endif // MODEL_HPP
