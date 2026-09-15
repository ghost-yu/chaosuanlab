// ============================================================================
// self_attention_cpu.cpp —— self_attention 算子的 CPU 实现
// ----------------------------------------------------------------------------
// 【self_attention 是什么？】
// Transformer 的核心："每个 token 在生成时，回头看看所有历史 token，
// 决定自己该重点关注谁"。三个角色：
//   q（Query 查询）  ："我在找什么"；
//   k（Key 键）      ："我有什么特征"；
//   v（Value 值）    ："我能提供什么内容"。
// 流程：q 和每个 k 做点积得到"匹配分数" → softmax 变成权重
//       → 用权重对 v 加权求和，得到注意力输出。
//
// 这是所有算子中最复杂的一个。算法分三步（对每个 head h、每个位置 i）：
//
//   1. 算注意力分数：scores[j] = (q[i] · k[j]) * scale
//      注意因果掩码：j 只能取到 i 对应的历史位置（j <= i + offset），
//      超出部分直接设为 -INFINITY（softmax 里 e^(-∞)=0，等价于不看它）。
//      offset = total_len - seqlen：KV 缓存里比当前序列多出的历史 token 数。
//
//   2. 数值稳定的 softmax：
//      先找出本行最大值 max_score，再算 exp(scores[j] - max_score)。
//      为什么减 max？直接 exp 大数会溢出成 Inf，减掉最大值后指数 <= 0，
//      永远不会溢出。这是 softmax 的标准做法。
//
//   3. 加权求和：attn_val[i] = Σ_j softmax_scores[j] * v[j]。
//
// GQA：nrep = nhead / nkvhead；第 h 个查询头使用第 h/nrep 个键值头。
// （GQA = Grouped Query Attention，多个查询头共享一个键值头，省显存。）
// ============================================================================

#include "self_attention_cpu.hpp"  // 本文件函数的声明。

#include "../../../utils.hpp"  // 项目公共工具（cast、异常宏）。

#include <cmath>    // 标准库：std::exp / std::isinf / INFINITY。
#include <vector>   // 标准库：std::vector（动态数组，存中间分数）。

// ---------------------------------------------------------------------------
// 模板实现：T 为 float / fp16_t / bf16_t。
// ---------------------------------------------------------------------------
template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     float scale, size_t seqlen, size_t total_len, size_t nhead,
                     size_t nkvhead, size_t d, size_t dv) {
    // GQA 分组：每组多少个查询头共享一个键值头。
    // 第 h 个查询头对应的键值头编号 = h / nrep（向下取整）。
    size_t nrep = nhead / nkvhead;
    // 每行（一个 (位置, 头) 组合）的注意力分数，长度 = total_len。
    // 用 float 存中间分数：不管输入是 fp16/bf16，注意力计算都在 float 上做。
    std::vector<float> scores(total_len);

    for (size_t h = 0; h < nhead; h++) {        // 遍历所有查询头
        size_t kh = h / nrep;                   // 本头对应的键值头编号（GQA 映射）

        for (size_t i = 0; i < seqlen; i++) {   // 遍历序列位置
            float max_score = -INFINITY;        // 本行最大分数（softmax 稳定化用）

            // ---- 第 1 步：计算分数（含因果掩码）----
            for (size_t j = 0; j < total_len; j++) {
                // 因果掩码：当前第 i 个位置只能"看"到历史上 j 的位置。
                // 关键理解：k/v 里前 offset 个是"以前生成的 token"（KV 缓存），
                // 后面 seqlen 个是"当前的 token"。所以第 i 个查询位置
                // 能看的键位置上限 = i + offset。
                if (j > i + (total_len - seqlen)) {
                    scores[j] = -INFINITY;      // 掩码：设为负无穷（softmax 后权重为 0）
                    continue;
                }

                // 点积：q 的第 i 行与 k 的第 j 行（同头 kh）做内积。
                float dot = 0.0f;
                for (size_t dim = 0; dim < d; dim++) {
                    if constexpr (std::is_same_v<T, chaosuan::bf16_t> || std::is_same_v<T, chaosuan::fp16_t>) {
                        float qv = chaosuan::utils::cast<float>(q[i * nhead * d + h * d + dim]);
                        float kval = chaosuan::utils::cast<float>(k[j * nkvhead * d + kh * d + dim]);
                        dot += qv * kval;
                    } else {
                        dot += q[i * nhead * d + h * d + dim] *
                               k[j * nkvhead * d + kh * d + dim];
                    }
                }
                dot *= scale;                   // 乘缩放因子（通常 1/sqrt(d)）
                scores[j] = dot;
                if (dot > max_score) max_score = dot;   // 记录本行最大值
            }

            // ---- 第 2 步：数值稳定的 softmax ----
            float sum_exp = 0.0f;               // 指数和（softmax 分母）
            for (size_t j = 0; j < total_len; j++) {
                // 被掩码的位置是 -INFINITY：exp(-Inf - max) 会出问题，
                // 所以用 std::isinf 判断，直接给 0（e^(-∞) = 0）。
                float exp_val = std::isinf(scores[j]) ? 0.0f : std::exp(scores[j] - max_score);
                scores[j] = exp_val;            // 原地覆盖为 exp 值（后面还要用）
                sum_exp += exp_val;
            }

            // ---- 第 3 步：用 softmax 权重对 v 加权求和 ----
            for (size_t dim = 0; dim < dv; dim++) {
                float val = 0.0f;
                for (size_t j = 0; j < total_len; j++) {
                    if (scores[j] > 0.0f) {     // 掩码位置权重为 0，跳过（省计算）
                        if constexpr (std::is_same_v<T, chaosuan::bf16_t> || std::is_same_v<T, chaosuan::fp16_t>) {
                            float vv = chaosuan::utils::cast<float>(v[j * nkvhead * dv + kh * dv + dim]);
                            val += scores[j] * vv;
                        } else {
                            val += scores[j] * v[j * nkvhead * dv + kh * dv + dim];
                        }
                    }
                }
                // 归一化：除以指数和。注意若整行都被掩码（sum_exp=0），
                // 这里会出现除零；本项目输入保证了 total_len>=seqlen 且
                // 至少 j=0 未被掩码，所以 sum_exp>0（除非 d 为 0）。
                if constexpr (std::is_same_v<T, chaosuan::bf16_t> || std::is_same_v<T, chaosuan::fp16_t>) {
                    attn_val[i * nhead * dv + h * dv + dim] = chaosuan::utils::cast<T>(val / sum_exp);
                } else {
                    attn_val[i * nhead * dv + h * dv + dim] = val / sum_exp;
                }
            }
        }
    }
}

namespace chaosuan::ops::cpu {

// 对外入口：按数据类型选择模板实例。
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, chaosuanDataType_t dtype,
                    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        return self_attention_<float>(reinterpret_cast<float *>(attn_val),
                                      reinterpret_cast<const float *>(q),
                                      reinterpret_cast<const float *>(k),
                                      reinterpret_cast<const float *>(v),
                                      scale, seqlen, total_len, nhead, nkvhead, d, dv);
    case CHAOSUAN_DTYPE_BF16:
        return self_attention_<chaosuan::bf16_t>(reinterpret_cast<chaosuan::bf16_t *>(attn_val),
                                                 reinterpret_cast<const chaosuan::bf16_t *>(q),
                                                 reinterpret_cast<const chaosuan::bf16_t *>(k),
                                                 reinterpret_cast<const chaosuan::bf16_t *>(v),
                                                 scale, seqlen, total_len, nhead, nkvhead, d, dv);
    case CHAOSUAN_DTYPE_F16:
        return self_attention_<chaosuan::fp16_t>(reinterpret_cast<chaosuan::fp16_t *>(attn_val),
                                                 reinterpret_cast<const chaosuan::fp16_t *>(q),
                                                 reinterpret_cast<const chaosuan::fp16_t *>(k),
                                                 reinterpret_cast<const chaosuan::fp16_t *>(v),
                                                 scale, seqlen, total_len, nhead, nkvhead, d, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);  // 其它类型不支持
    }
}
} // namespace chaosuan::ops::cpu
