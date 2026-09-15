// ============================================================================
// rms_norm_nvidia.cuh —— rms_norm（归一化） 的 CUDA 实现声明（作业 4）
// ----------------------------------------------------------------------------
// 这个头文件只声明"函数长什么样"，真正的 CUDA kernel 在
// 同目录的 rms_norm_nvidia.cu 里（nvcc 编译）。
// op.cpp 在 ENABLE_NVIDIA_API 下 include 本文件并调用这些函数。
// ============================================================================

#pragma once

#include "chaosuan.h"  // 数据类型枚举。

#include <cstddef>     // std::byte、size_t。

namespace chaosuan::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, chaosuanDataType_t dtype, size_t m, size_t d);
} // namespace chaosuan::ops::nvidia
