// ============================================================================
// rms_norm_cpu.hpp —— rms_norm 算子的 CPU 实现【头文件/声明】
// ----------------------------------------------------------------------------
// 头文件只写函数"长什么样"，实现写在同名的 .cpp 里。
// ============================================================================

#pragma once  // 防止头文件被重复包含导致"重复定义"错误。

#include "chaosuan.h"  // 数据类型枚举 chaosuanDataType_t。

#include <cstddef>    // std::byte、size_t。

namespace chaosuan::ops::cpu {

// RMSNorm 归一化：out[i][j] = in[i][j] * weight[j] / sqrt(mean(in[i]^2) + eps)
//   out    —— 输出：形状 [m, d]；
//   in     —— 输入：形状 [m, d]；
//   weight —— 输入：可学习的缩放权重，形状 [d]（每个维度一个）；
//   eps    —— 防除零的小常数；
//   dtype  —— 数据类型枚举（chaosuanDataType_t）；
//   m, d   —— 行数（样本数）和每行长度。
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, chaosuanDataType_t dtype, size_t m, size_t d);
}
