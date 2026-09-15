// ============================================================================
// argmax_nvidia.cuh —— argmax（最大值下标） 的 CUDA 实现声明（作业 4）
// ----------------------------------------------------------------------------
// 这个头文件只声明"函数长什么样"，真正的 CUDA kernel 在
// 同目录的 argmax_nvidia.cu 里（nvcc 编译）。
// op.cpp 在 ENABLE_NVIDIA_API 下 include 本文件并调用这些函数。
// ============================================================================

#pragma once

#include "chaosuan.h"  // 数据类型枚举。

#include <cstddef>     // std::byte、size_t。

namespace chaosuan::ops::nvidia {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, chaosuanDataType_t dtype, size_t numel);
} // namespace chaosuan::ops::nvidia
