// ============================================================================
// embedding_nvidia.cuh —— embedding（查表） 的 CUDA 实现声明（作业 4）
// ----------------------------------------------------------------------------
// 这个头文件只声明"函数长什么样"，真正的 CUDA kernel 在
// 同目录的 embedding_nvidia.cu 里（nvcc 编译）。
// op.cpp 在 ENABLE_NVIDIA_API 下 include 本文件并调用这些函数。
// ============================================================================

#pragma once

#include "chaosuan.h"  // 数据类型枚举。

#include <cstddef>     // std::byte、size_t。

namespace chaosuan::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, chaosuanDataType_t dtype, size_t numel, size_t embedding_dim);
} // namespace chaosuan::ops::nvidia
