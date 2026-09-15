// ============================================================================
// self_attention_nvidia.cuh —— self_attention（注意力） 的 CUDA 实现声明（作业 4）
// ----------------------------------------------------------------------------
// 这个头文件只声明"函数长什么样"，真正的 CUDA kernel 在
// 同目录的 self_attention_nvidia.cu 里（nvcc 编译）。
// op.cpp 在 ENABLE_NVIDIA_API 下 include 本文件并调用这些函数。
// ============================================================================

#pragma once

#include "chaosuan.h"  // 数据类型枚举。

#include <cstddef>     // std::byte、size_t。

namespace chaosuan::ops::nvidia {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, chaosuanDataType_t dtype, size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv);
} // namespace chaosuan::ops::nvidia
