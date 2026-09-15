// ============================================================================
// rope_cpu.hpp —— rope 算子的 CPU 实现【头文件/声明】
// ----------------------------------------------------------------------------
// 头文件只写函数"长什么样"，实现写在同名的 .cpp 里。
// ============================================================================

#pragma once  // 防止头文件被重复包含导致"重复定义"错误。

#include "chaosuan.h"  // 数据类型枚举 chaosuanDataType_t。

#include <cstddef>    // std::byte、size_t。

namespace chaosuan::ops::cpu {

// 旋转位置编码（RoPE）：对每个 token 的每个 head 向量做二维旋转变换，
// 把"位置信息"编码进向量。
//   out    —— 输出：形状 [seqlen, nheads, d]，与 in 同形状；
//   in     —— 输入：形状 [seqlen, nheads, d]；
//   pos_ids —— 输入：每个 token 的位置编号，长度 seqlen（固定 int64_t）；
//   theta  —— 频率基（Qwen 默认 10000.0）；
//   dtype  —— 数据类型枚举（chaosuanDataType_t）；
//   seqlen, nheads, d —— 序列长度、头数、每个 head 的向量维度（d 必须为偶数）。
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, chaosuanDataType_t dtype, size_t seqlen, size_t nheads, size_t d);
}
