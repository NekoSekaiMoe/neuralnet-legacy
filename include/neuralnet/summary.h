#ifndef SUMMARY_HPP
#define SUMMARY_HPP

// ── Model::summary() (F4) ────────────────────────────────────────────────────
// 输出每层的 name、Output Shape、Params；对 BatchNorm 还显示 running stats 维度。
// 末尾给出 Total params。
// 接口：summary(model) 返回字符串；summary(model, &ostream) 同时写入流。

#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <neuralnet/layer.h>
#include <neuralnet/matrix.h>
#include <neuralnet/model/model.h>

namespace nn
{
    namespace detail
    {
        inline std::string format_param_count(std::size_t n)
        {
            return std::to_string(n);
        }

        inline std::string make_separator()
        {
            return std::string(60, '=');
        }
    } // namespace detail

    inline std::string summary(const Model &model, std::ostream *out = nullptr)
    {
        std::ostringstream oss;
        const std::string sep = detail::make_separator();

        oss << sep << "\n";
        oss << std::left
            << std::setw(20) << "Layer (name)"
            << std::setw(20) << "Output Shape"
            << std::setw(10) << "Params"
            << "\n";
        oss << sep << "\n";

        std::size_t total_params = 0;
        for (std::size_t i = 0; i < model.num_layers(); ++i)
        {
            const Layer &layer = *model.get_layers()[i];
            const std::string nm = layer.name();
            const std::size_t pc = layer.param_count();
            total_params += pc;

            oss << std::left << std::setw(20) << nm
                << std::setw(20) << "(?)"
                << std::setw(10) << detail::format_param_count(pc);

            // 对 BatchNorm，附加 running stats 维度信息
            if (nm == std::string("BatchNorm1d") || nm == std::string("BatchNorm2d"))
            {
                const auto &bn = static_cast<const BatchNorm1d &>(layer);
                oss << "  running_mean ("
                    << bn.running_mean().rows() << ", "
                    << bn.running_mean().cols() << ")"
                    << " running_var ("
                    << bn.running_var().rows() << ", "
                    << bn.running_var().cols() << ")";
            }
            oss << "\n";
        }

        oss << sep << "\n";
        oss << "Total params: " << total_params << "\n";

        std::string result = oss.str();
        if (out != nullptr)
        {
            *out << result;
        }
        return result;
    }

    // Model::summary() 委托给上面的自由函数。定义为 inline 以便在
    // 头文件被多 TU 包含时不违反 ODR；summary.h 必须先于本实现被包含。
    inline std::string Model::summary(std::ostream *out) const
    {
        return nn::summary(*this, out);
    }
} // namespace nn

#endif // SUMMARY_HPP
