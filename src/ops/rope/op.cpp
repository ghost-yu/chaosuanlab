// ============================================================================
// op.cpp —— rope 算子的实现：参数校验 + 按设备分发
// ----------------------------------------------------------------------------
// 校验重点：3 维张量、d 必须为偶数（要切两半）、pos_ids 是 int64
// 且长度等于 seqlen。
// 真正算数的是 cpu/rope_cpu.cpp。
// ============================================================================

#include "op.hpp"  // 本算子声明。

#include "../../core/chaosuan_core.hpp"  // 核心上下文（切换设备用）。
#include "../../utils.hpp"               // 项目公共工具（校验宏）。

#include "cpu/rope_cpu.hpp"              // CPU 实现头文件。
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"        // CUDA 实现头文件（作业 4 再加）。
#endif

namespace chaosuan::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // -----------------------------------------------------------------------
    // 1. 参数校验
    // -----------------------------------------------------------------------
    CHECK_SAME_DEVICE(out, in);                                  // out 与 in 同设备
                                                                 // （pos_ids 也在同设备，见下）
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());                 // 同类型
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");             // 连续内存
    ASSERT(out->ndim() == 3, "RoPE: out must be 3D [seqlen, nhead, d].");
    ASSERT(in->ndim() == 3, "RoPE: in must be 3D [seqlen, nhead, d].");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D [seqlen].");
    ASSERT(pos_ids->dtype() == CHAOSUAN_DTYPE_I64, "RoPE: pos_ids must be int64.");

    size_t seqlen = in->shape()[0];   // 序列长度（token 数）
    size_t nheads = in->shape()[1];   // head 数
    size_t d = in->shape()[2];        // 每个 head 的向量维度

    ASSERT(d % 2 == 0, "RoPE: head_dim must be even.");  // d 必须偶数（要切 a/b 两半）
    ASSERT(out->shape()[0] == seqlen && out->shape()[1] == nheads && out->shape()[2] == d,
           "RoPE: out shape mismatch.");                  // out 与 in 同形状
    ASSERT(pos_ids->shape()[0] == seqlen, "RoPE: pos_ids shape mismatch.");  // 每 token 一个位置

    // -----------------------------------------------------------------------
    // 2. 按设备分发
    // -----------------------------------------------------------------------
    if (out->deviceType() == CHAOSUAN_DEVICE_CPU) {   // CPU 快速路径
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(),
                         seqlen, nheads, d);
    }

    chaosuan::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case CHAOSUAN_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(),
                         seqlen, nheads, d);
#ifdef ENABLE_NVIDIA_API
    case CHAOSUAN_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace chaosuan::ops
