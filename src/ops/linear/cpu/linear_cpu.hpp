// ============================================================================
// linear_cpu.hpp —— linear 算子的 CPU 实现【头文件/声明】
// ----------------------------------------------------------------------------
// 头文件只写函数"长什么样"，实现写在同名的 .cpp 里。
// ============================================================================

#pragma once  // 防止头文件被重复包含导致"重复定义"错误。

#include "chaosuan.h"  // 数据类型枚举 chaosuanDataType_t。

#include <cstddef>    // std::byte、size_t。

namespace chaosuan::ops::cpu {

// 全连接层：out[m,n] = in[m,k] · weight[n,k]^T + bias[n]
//   out    —— 输出：形状 [m, n] 的结果矩阵；
//   in     —— 输入：形状 [m, k]；
//   weight —— 输入：形状 [n, k]（未转置，第 j 行就是输出第 j 列的权重）；
//   bias   —— 输入：形状 [n] 的偏置（可为 nullptr = 没有偏置）；
//   dtype  —— 数据类型枚举（chaosuanDataType_t）；
//   m, n, k —— 三个维度大小。
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            chaosuanDataType_t dtype, size_t m, size_t n, size_t k);
}
