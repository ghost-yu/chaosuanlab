// ============================================================================
// rope_cpu.cpp —— rope 算子的 CPU 实现
// ----------------------------------------------------------------------------
// 【RoPE 是什么？】
// Transformer 的注意力本身"不感知位置"（把句子打乱顺序，结果一样），
// 所以必须把"位置信息"编码进向量。RoPE（Rotary Position Embedding，
// 旋转位置编码）的做法是：根据位置 pos，把向量的每一对分量
// （第 j 个和第 d/2+j 个）旋转一个角度 angle = pos / theta^(2j/d)。
// 旋转公式（二维旋转）：
//   a' = a*cos(angle) - b*sin(angle)
//   b' = b*cos(angle) + a*sin(angle)
//
// 为什么旋转能编码位置？两个向量旋转后做点积，点积值恰好等于
// 它们"原始夹角"加上"位置差"的旋转角——位置差越大，点积越小，
// 注意力自然"偏好相邻位置的 token"。这是 RoPE 的精妙之处。
//
// 实现要点：
//   频率向量 freq[j] = theta^(2j/d)，j = 0..d/2-1：只依赖 theta 和 d，
//   与数据无关，所以用全局缓存避免重复计算（重复调用时省时间）。
//   对每个 token s（位置 pos = pos_ids[s]）、每对分量 j、每个 head h：
//     angle = pos / freq[j]
//     同一 (s, j) 下所有 head 的 angle 相同 → 提前算好 cos/sin，内层复用。
// ============================================================================

#include "rope_cpu.hpp"  // 本文件函数的声明。

#include <cmath>          // 标准库：std::pow / std::cos / std::sin。
#include <cstring>        // 标准库：std::memcpy（构造缓存 key 用）。
#include <unordered_map>  // 标准库：std::unordered_map（哈希表缓存）。
#include <vector>         // 标准库：std::vector（动态数组）。

#include "../../../utils.hpp"  // 项目公共工具（cast、异常宏）。

namespace {

// 构造缓存的 key：把 (theta 的位模式, d) 打包成一个 64 位整数。
// 为什么不用 pair<float,size_t> 做 key？浮点数做哈希键不稳定
// （NaN/±0 等边界情况），打包成整数更可靠。
uint64_t make_freq_key(float theta, size_t d) {
    uint32_t theta_bits;
    std::memcpy(&theta_bits, &theta, sizeof(theta_bits));   // float 位模式 -> uint32
    return (static_cast<uint64_t>(theta_bits) << 32) | static_cast<uint64_t>(d);
    // 高 32 位放 theta 的位模式，低 32 位放 d，拼成一个唯一 key。
}

// 获取频率向量 freq[j] = theta^(2j/d)，带全局缓存。
// 返回 const 引用，避免每次调用都复制一份 vector。
// 注意：static 局部变量在 C++11 起是线程安全初始化的。
const std::vector<float> &get_freq(float theta, size_t d) {
    static std::unordered_map<uint64_t, std::vector<float>> cache;  // 全局缓存表

    uint64_t key = make_freq_key(theta, d);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;      // 命中缓存：直接返回
    }

    // 未命中：现场计算频率向量。
    size_t half_d = d / 2;      // 频率向量长度 = d/2（每对分量一个频率）
    std::vector<float> freq(half_d);
    for (size_t j = 0; j < half_d; j++) {
        freq[j] = std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(d));
        // freq[j] = theta^(2j/d)：2j/d 作为指数
    }
    // 存入缓存并返回。std::move 避免拷贝；emplace 返回 (迭代器, 是否插入成功)。
    auto [new_it, _] = cache.emplace(key, std::move(freq));
    return new_it->second;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// 模板实现：T 为 float / fp16_t / bf16_t。
// ---------------------------------------------------------------------------
template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta,
           size_t seqlen, size_t nheads, size_t d) {
    size_t half_d = d / 2;

    const auto &freq = get_freq(theta, d);   // 取频率向量（可能命中缓存）

    for (size_t s = 0; s < seqlen; s++) {            // 遍历 token
        float pos = static_cast<float>(pos_ids[s]);  // 当前 token 的位置编号
        for (size_t j = 0; j < half_d; j++) {        // 遍历 d/2 对分量
            float angle = pos / freq[j];             // 旋转角 = pos / theta^(2j/d)
            float cos_val = std::cos(angle);         // 提前算好 cos/sin，
            float sin_val = std::sin(angle);         // 下面 nheads 个 head 复用
            for (size_t h = 0; h < nheads; h++) {    // 遍历 head（共享同一角度）
                // 当前 (token, head) 向量在扁平内存中的起始下标。
                // 内存排布：[seqlen][nheads][d]，所以
                // base_idx = s*nheads*d + h*d。
                size_t base_idx = s * nheads * d + h * d;

                // 取该向量第 j 对分量：前半 [j]，后半 [half_d + j]。
                float a, b;
                if constexpr (std::is_same_v<T, chaosuan::bf16_t> || std::is_same_v<T, chaosuan::fp16_t>) {
                    a = chaosuan::utils::cast<float>(in[base_idx + j]);
                    b = chaosuan::utils::cast<float>(in[base_idx + half_d + j]);
                } else {
                    a = in[base_idx + j];
                    b = in[base_idx + half_d + j];
                }

                // 二维旋转公式（RoPE 的核心）：
                float ra = a * cos_val - b * sin_val;  // a' = a*cos - b*sin
                float rb = b * cos_val + a * sin_val;  // b' = b*cos + a*sin

                // 写回输出（低精度类型要转回原类型）。
                if constexpr (std::is_same_v<T, chaosuan::bf16_t> || std::is_same_v<T, chaosuan::fp16_t>) {
                    out[base_idx + j] = chaosuan::utils::cast<T>(ra);
                    out[base_idx + half_d + j] = chaosuan::utils::cast<T>(rb);
                } else {
                    out[base_idx + j] = ra;
                    out[base_idx + half_d + j] = rb;
                }
            }
        }
    }
}

namespace chaosuan::ops::cpu {

// 对外入口：按数据类型选择模板实例。
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, chaosuanDataType_t dtype, size_t seqlen, size_t nheads, size_t d) {
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        return rope_<float>(reinterpret_cast<float *>(out),
                            reinterpret_cast<const float *>(in),
                            reinterpret_cast<const int64_t *>(pos_ids),
                            theta, seqlen, nheads, d);
    case CHAOSUAN_DTYPE_BF16:
        return rope_<chaosuan::bf16_t>(reinterpret_cast<chaosuan::bf16_t *>(out),
                                       reinterpret_cast<const chaosuan::bf16_t *>(in),
                                       reinterpret_cast<const int64_t *>(pos_ids),
                                       theta, seqlen, nheads, d);
    case CHAOSUAN_DTYPE_F16:
        return rope_<chaosuan::fp16_t>(reinterpret_cast<chaosuan::fp16_t *>(out),
                                       reinterpret_cast<const chaosuan::fp16_t *>(in),
                                       reinterpret_cast<const int64_t *>(pos_ids),
                                       theta, seqlen, nheads, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);  // 其它类型不支持
    }
}
} // namespace chaosuan::ops::cpu
