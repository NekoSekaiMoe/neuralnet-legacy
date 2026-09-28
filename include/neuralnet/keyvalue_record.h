#ifndef NN_KEYVALUE_RECORD_HPP
#define NN_KEYVALUE_RECORD_HPP

// ══════════════════════════════════════════════════════════════════════════
//  keyvalue_record.h — 自描述键值记录（移植自上游 model_keyvalue_record）
//
//  设计目标：
//    1. 容易解析 —— 长度前缀 + 显式类型 + 值长度前缀，无状态机/偏移量假设。
//    2. 自描述   —— 每条记录自带 key + type + value，未知字段可按长度跳过。
//    3. 面向 C++ —— set/get 直接对应 uint64_t / double / std::string / 数组。
//    4. 版本友好 —— 缺失字段由上层按版本记录默认值；新增字段只需加一条
//       set/get；出现未知类型时解析器按 value_len 安全跳过（向前兼容）。
//
//  字节布局（小端）：
//    [field_count u32]
//    field := [key_len u32][key bytes][type u8][value_len u32][value bytes]
//      type 0 (UInt)       : value = 8 字节 uint64
//      type 1 (Double)     : value = 8 字节 IEEE754 双精度
//      type 2 (Str)        : value = 原始字符串字节
//      type 3 (UIntArray)  : value = count×8 字节的 uint64 数组
//      type 4 (DoubleArray): value = count×8 字节的 double 数组
// ══════════════════════════════════════════════════════════════════════════

#include <cstddef>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nn
{

    class KeyValueRecord
    {
    public:
        enum class Type : uint8_t
        {
            UInt = 0,
            Double = 1,
            Str = 2,
            UIntArray = 3,
            DoubleArray = 4,
        };

        // ── 写入（builder 风格）──────────────────────────────────────────
        KeyValueRecord &set(const std::string &key, uint64_t v)
        {
            Field f;
            f.key = key;
            f.type = Type::UInt;
            f.u = v;
            fields_.push_back(std::move(f));
            return *this;
        }

        KeyValueRecord &set(const std::string &key, double v)
        {
            Field f;
            f.key = key;
            f.type = Type::Double;
            f.d = v;
            fields_.push_back(std::move(f));
            return *this;
        }

        KeyValueRecord &set(const std::string &key, int v)
        {
            return set(key, static_cast<uint64_t>(static_cast<int64_t>(v)));
        }

        KeyValueRecord &set(const std::string &key, const std::string &v)
        {
            Field f;
            f.key = key;
            f.type = Type::Str;
            f.s = v;
            fields_.push_back(std::move(f));
            return *this;
        }

        KeyValueRecord &set(const std::string &key, const std::vector<uint64_t> &v)
        {
            Field f;
            f.key = key;
            f.type = Type::UIntArray;
            f.uarr = v;
            fields_.push_back(std::move(f));
            return *this;
        }

        KeyValueRecord &set(const std::string &key, const std::vector<double> &v)
        {
            Field f;
            f.key = key;
            f.type = Type::DoubleArray;
            f.darr = v;
            fields_.push_back(std::move(f));
            return *this;
        }

        // 序列化为字节串（不含总长度前缀；总长度由文件格式负责）。
        // 与 parse() 对称的硬限制：超限直接抛异常，避免静默截断长度前缀、
        // 写出加载不回的文件。
        [[nodiscard]] std::string serialize() const
        {
            constexpr std::size_t kMaxFields = 4096;
            constexpr std::size_t kMaxKeyLen = 256;
            std::string out;
            if (fields_.size() > kMaxFields)
                throw std::length_error("KeyValueRecord::serialize: too many fields");
            append_u32(out, static_cast<uint32_t>(fields_.size()));
            for (const Field &f : fields_)
            {
                if (f.key.size() > kMaxKeyLen)
                    throw std::length_error("KeyValueRecord::serialize: key too long");
                append_u32(out, static_cast<uint32_t>(f.key.size()));
                out.append(f.key);
                out.push_back(static_cast<char>(f.type));
                std::string value;
                switch (f.type)
                {
                case Type::UInt:
                    append_u64(value, f.u);
                    break;
                case Type::Double:
                {
                    uint64_t bits = 0;
                    static_assert(sizeof(bits) == sizeof(f.d), "double must be 8 bytes");
                    std::memcpy(&bits, &f.d, sizeof(bits));
                    append_u64(value, bits);
                    break;
                }
                case Type::Str:
                    value = f.s;
                    break;
                case Type::UIntArray:
                    for (const uint64_t x : f.uarr)
                        append_u64(value, x);
                    break;
                case Type::DoubleArray:
                    for (const double x : f.darr)
                    {
                        uint64_t bits = 0;
                        std::memcpy(&bits, &x, sizeof(bits));
                        append_u64(value, bits);
                    }
                    break;
                }
                if (value.size() > std::numeric_limits<uint32_t>::max())
                    throw std::length_error("KeyValueRecord::serialize: value too long");
                append_u32(out, static_cast<uint32_t>(value.size()));
                out.append(value);
            }
            return out;
        }

        // ── 解析：整块字节串 → KeyValueRecord ─────────────────────────────
        // 非法输入（截断/类型不符）抛 std::runtime_error；未知类型按长度跳过。
        [[nodiscard]] static KeyValueRecord parse(std::string_view bytes)
        {
            KeyValueRecord rec;
            uint32_t field_count = 0;
            if (!take_u32(bytes, field_count))
                throw std::runtime_error("KeyValueRecord::parse: truncated header");
            constexpr uint32_t kMaxFields = 4096;
            if (field_count > kMaxFields)
                throw std::runtime_error("KeyValueRecord::parse: implausible field count");

            for (uint32_t i = 0; i < field_count; ++i)
            {
                uint32_t key_len = 0;
                if (!take_u32(bytes, key_len) || key_len > bytes.size())
                    throw std::runtime_error("KeyValueRecord::parse: truncated key");
                if (key_len > 256)
                    throw std::runtime_error("KeyValueRecord::parse: implausible key length");
                std::string key(bytes.substr(0, key_len));
                bytes.remove_prefix(key_len);

                if (bytes.empty())
                    throw std::runtime_error("KeyValueRecord::parse: truncated type tag");
                const auto type = static_cast<Type>(static_cast<uint8_t>(bytes.front()));
                bytes.remove_prefix(1);

                uint32_t value_len = 0;
                if (!take_u32(bytes, value_len) || value_len > bytes.size())
                    throw std::runtime_error("KeyValueRecord::parse: truncated value");

                Field f;
                f.key = std::move(key);
                f.type = type;
                bool unknown = false;
                switch (type)
                {
                case Type::UInt:
                    if (value_len != 8)
                        throw std::runtime_error("KeyValueRecord::parse: bad UInt length");
                    f.u = take_u64(bytes);
                    break;
                case Type::Double:
                {
                    if (value_len != 8)
                        throw std::runtime_error("KeyValueRecord::parse: bad Double length");
                    const uint64_t bits = take_u64(bytes);
                    std::memcpy(&f.d, &bits, sizeof(f.d));
                    break;
                }
                case Type::Str:
                    f.s = std::string(bytes.substr(0, value_len));
                    bytes.remove_prefix(value_len);
                    break;
                case Type::UIntArray:
                {
                    if (value_len % 8 != 0)
                        throw std::runtime_error("KeyValueRecord::parse: bad UIntArray length");
                    const std::size_t count = value_len / 8;
                    f.uarr.resize(count);
                    for (std::size_t k = 0; k < count; ++k)
                        f.uarr[k] = take_u64(bytes);
                    break;
                }
                case Type::DoubleArray:
                {
                    if (value_len % 8 != 0)
                        throw std::runtime_error("KeyValueRecord::parse: bad DoubleArray length");
                    const std::size_t count = value_len / 8;
                    f.darr.resize(count);
                    for (std::size_t k = 0; k < count; ++k)
                    {
                        const uint64_t bits = take_u64(bytes);
                        std::memcpy(&f.darr[k], &bits, sizeof(double));
                    }
                    break;
                }
                default:
                    // 未知类型：按 value_len 安全跳过（向前兼容，不入库）
                    bytes.remove_prefix(value_len);
                    unknown = true;
                    break;
                }

                if (!unknown)
                    rec.fields_.push_back(std::move(f));
            }
            return rec;
        }

        // ── 读取（返回 false 表示缺失或类型不符）─────────────────────────
        [[nodiscard]] bool has(const std::string &key) const
        {
            for (const Field &f : fields_)
                if (f.key == key)
                    return true;
            return false;
        }

        [[nodiscard]] bool get(const std::string &key, uint64_t &out) const
        {
            for (const Field &f : fields_)
                if (f.key == key && f.type == Type::UInt)
                {
                    out = f.u;
                    return true;
                }
            return false;
        }

        [[nodiscard]] bool get(const std::string &key, double &out) const
        {
            for (const Field &f : fields_)
                if (f.key == key && f.type == Type::Double)
                {
                    out = f.d;
                    return true;
                }
            return false;
        }

        [[nodiscard]] bool get(const std::string &key, std::string &out) const
        {
            for (const Field &f : fields_)
                if (f.key == key && f.type == Type::Str)
                {
                    out = f.s;
                    return true;
                }
            return false;
        }

        [[nodiscard]] bool get(const std::string &key, std::vector<uint64_t> &out) const
        {
            for (const Field &f : fields_)
                if (f.key == key && f.type == Type::UIntArray)
                {
                    out = f.uarr;
                    return true;
                }
            return false;
        }

        [[nodiscard]] bool get(const std::string &key, std::vector<double> &out) const
        {
            for (const Field &f : fields_)
                if (f.key == key && f.type == Type::DoubleArray)
                {
                    out = f.darr;
                    return true;
                }
            return false;
        }

        [[nodiscard]] std::size_t size() const noexcept { return fields_.size(); }

    private:
        struct Field
        {
            std::string key;
            Type type = Type::UInt;
            uint64_t u = 0;
            double d = 0.0;
            std::string s;
            std::vector<uint64_t> uarr;
            std::vector<double> darr;
        };

        std::vector<Field> fields_;

        // ── 小端编码/解码工具 ───────────────────────────────────────────
        static void append_u32(std::string &out, uint32_t v)
        {
            out.push_back(static_cast<char>(v & 0xFF));
            out.push_back(static_cast<char>((v >> 8) & 0xFF));
            out.push_back(static_cast<char>((v >> 16) & 0xFF));
            out.push_back(static_cast<char>((v >> 24) & 0xFF));
        }

        static void append_u64(std::string &out, uint64_t v)
        {
            for (int i = 0; i < 8; ++i)
                out.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
        }

        static bool take_u32(std::string_view &s, uint32_t &out)
        {
            if (s.size() < 4)
                return false;
            out = static_cast<uint32_t>(static_cast<unsigned char>(s[0]))
                | (static_cast<uint32_t>(static_cast<unsigned char>(s[1])) << 8)
                | (static_cast<uint32_t>(static_cast<unsigned char>(s[2])) << 16)
                | (static_cast<uint32_t>(static_cast<unsigned char>(s[3])) << 24);
            s.remove_prefix(4);
            return true;
        }

        static uint64_t take_u64(std::string_view &s)
        {
            uint64_t out = 0;
            for (int i = 0; i < 8; ++i)
                out |= static_cast<uint64_t>(static_cast<unsigned char>(s[i])) << (8 * i);
            s.remove_prefix(8);
            return out;
        }
    };

} // namespace nn

#endif // NN_KEYVALUE_RECORD_HPP
