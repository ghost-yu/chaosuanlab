// ============================================================================
// self_attention_cpu.hpp —— self_attention 算子的 CPU 实现【头文件/声明】
// ----------------------------------------------------------------------------
// 头文件只写函数"长什么样"，实现写在同名的 .cpp 里。
// ============================================================================

#pragma once  // 防止头文件被重复包含导致"重复定义"错误。

#include "chaosuan.h"  // 数据类型枚举 chaosuanDataType_t。

#include <cstddef>    // std::byte、size_t。

namespace chaosuan::ops::cpu {

// 因果自注意力（带 KV 缓存支持 + GQA 分组查询注意力）：
// attn_val[qlen, nhead, dv] = softmax(q·k^T * scale + 因果掩码) · v
//   attn_val —— 输出：形状 [seqlen, nhead, dv]；
//   q        —— 输入：查询，形状 [seqlen, nhead, d]；
//   k        —— 输入：键，形状 [total_len, nkvhead, d]（可含历史 KV 缓存）；
//   v        —— 输入：值，形状 [total_len, nkvhead, dv]（可含历史 KV 缓存）；
//   scale    —— 缩放系数（通常 1/sqrt(d)）；
//   dtype    —— 数据类型枚举（chaosuanDataType_t）；
//   seqlen   —— 当前序列长度（q 的行数）；
//   total_len —— k/v 总长度（历史缓存 + 当前序列）；
//   nhead    —— 查询头数；
//   nkvhead  —— 键值头数（GQA：nhead 必须是 nkvhead 的整数倍）；
//   d, dv    —— 每个头的查询/键维度、值维度。
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, chaosuanDataType_t dtype,
                    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv);
}
