// ============================================================================
// rms_norm_cpu.cpp —— rms_norm 算子的 CPU 实现
// ----------------------------------------------------------------------------
// 【RMSNorm 是什么？】
// 大模型每一层网络之后都会做"归一化"，让数据保持稳定（不爆炸不消失）。
// RMSNorm（Root Mean Square Normalization，均方根归一化）是 Llama/Qwen
// 等模型用的归一化方法，比 LayerNorm 少了"减均值"这一步，更简单更快。
//
// 算法（每行独立，共 m 行，每行 d 个元素）：
//   第 1 遍：累加这一行所有元素的平方和 sum_sq；
//   计算 rms = sqrt(sum_sq / d + eps)；
//   第 2 遍：out[j] = in[j] * weight[j] / rms。
// 需要扫两遍数据，因为第一遍的结果（rms）要等整行扫完才能算出。
//
// 【为什么叫"均方根"？】
// 平方 → 求平均（除以 d）→ 开根号，就是"root mean square"。
// 每一行除以自己的 rms 后，整行的"能量"就被拉回到 1 左右。
// ============================================================================

#include "rms_norm_cpu.hpp"  // 本文件函数的声明。

#include <cmath>      // 标准库：std::sqrt（开平方）。

#include "../../../utils.hpp"  // 项目公共工具（cast、异常宏）。

// ---------------------------------------------------------------------------
// 模板实现：T 为 float / fp16_t / bf16_t。
// fp16/bf16 分支：CPU 上先转 float 做计算，再转回原类型存输出。
// 注意：无论输入是什么类型，**累加器/中间计算一律用 float**，
// 这是数值稳定性的常规做法——用低精度累加会放大舍入误差。
// ---------------------------------------------------------------------------
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t m, size_t d) {
    for (size_t i = 0; i < m; i++) {          // 逐行处理
        // ---- 第 1 遍：求平方和 ----
        float sum_sq = 0.0f;                  // 累加器，用 float
        for (size_t j = 0; j < d; j++) {
            if constexpr (std::is_same_v<T, chaosuan::bf16_t> || std::is_same_v<T, chaosuan::fp16_t>) {
                float val = chaosuan::utils::cast<float>(in[i * d + j]);  // 低精度 -> float
                sum_sq += val * val;          // 累加平方
            } else {
                sum_sq += in[i * d + j] * in[i * d + j];
            }
        }
        // 均方根：sqrt(平方和/长度 + eps)。eps 在根号**里面**加，
        // 这是 RMSNorm 的标准公式（保证分母不为 0）。
        float rms = std::sqrt(sum_sq / d + eps);

        // ---- 第 2 遍：归一化并乘上可学习权重 ----
        for (size_t j = 0; j < d; j++) {
            if constexpr (std::is_same_v<T, chaosuan::bf16_t> || std::is_same_v<T, chaosuan::fp16_t>) {
                float val = chaosuan::utils::cast<float>(in[i * d + j]);
                float w = chaosuan::utils::cast<float>(weight[j]);
                out[i * d + j] = chaosuan::utils::cast<T>(val * w / rms);  // 算完转回低精度
            } else {
                out[i * d + j] = in[i * d + j] * weight[j] / rms;
            }
        }
    }
}

namespace chaosuan::ops::cpu {

// 对外入口：按数据类型选择模板实例。
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, chaosuanDataType_t dtype, size_t m, size_t d) {
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        return rms_norm_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                                reinterpret_cast<const float *>(weight), eps, m, d);
    case CHAOSUAN_DTYPE_BF16:
        return rms_norm_<chaosuan::bf16_t>(reinterpret_cast<chaosuan::bf16_t *>(out),
                                           reinterpret_cast<const chaosuan::bf16_t *>(in),
                                           reinterpret_cast<const chaosuan::bf16_t *>(weight), eps, m, d);
    case CHAOSUAN_DTYPE_F16:
        return rms_norm_<chaosuan::fp16_t>(reinterpret_cast<chaosuan::fp16_t *>(out),
                                           reinterpret_cast<const chaosuan::fp16_t *>(in),
                                           reinterpret_cast<const chaosuan::fp16_t *>(weight), eps, m, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);  // 其它类型不支持
    }
}
} // namespace chaosuan::ops::cpu
