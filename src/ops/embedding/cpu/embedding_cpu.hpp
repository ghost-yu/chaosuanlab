// ============================================================================
// embedding_cpu.hpp —— embedding 算子的 CPU 实现【头文件/声明】
// ----------------------------------------------------------------------------
// 头文件只写函数"长什么样"，实现写在同名的 .cpp 里。
// ============================================================================

#pragma once  // 防止头文件被重复包含导致"重复定义"错误。

#include "chaosuan.h"  // 数据类型枚举 chaosuanDataType_t。

#include <cstddef>    // std::byte、size_t。

namespace chaosuan::ops::cpu {

// 按 index 里的编号，把 weight 的对应行拷贝到 out（词向量查表）。
//   out            —— 输出：拼接后的向量表（行数 = numel，每行 embedding_dim 个元素）；
//   index          —— 输入：词的编号数组（固定 int64_t）；
//   weight         —— 输入：词向量表（weight[行号][列号]）；
//   dtype          —— weight / out 的数据类型；
//   numel          —— index 长度（输出行数）；
//   embedding_dim  —— 每行向量长度。
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               chaosuanDataType_t dtype, size_t numel, size_t embedding_dim);
}
