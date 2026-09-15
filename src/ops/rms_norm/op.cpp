// ============================================================================
// op.cpp —— rms_norm 算子的实现：参数校验 + 按设备分发
// ----------------------------------------------------------------------------
// 调度入口套路：校验 → 取维度 → 按设备交给后端。
// 真正算数的是 cpu/rms_norm_cpu.cpp。
// ============================================================================

#include "op.hpp"  // 本算子声明。

#include "../../core/chaosuan_core.hpp"  // 核心上下文（切换设备用）。
#include "../../utils.hpp"               // 项目公共工具（校验宏）。

#include "cpu/rms_norm_cpu.hpp"          // CPU 实现头文件。
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"    // CUDA 实现头文件（作业 4 再加）。
#endif

namespace chaosuan::ops {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // -----------------------------------------------------------------------
    // 1. 参数校验
    // -----------------------------------------------------------------------
    CHECK_SAME_DEVICE(out, in, weight);                              // 三个张量同设备
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());    // 同类型
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RmsNorm: all tensors must be contiguous.");              // 连续内存
    ASSERT(in->ndim() == 2, "RmsNorm: in must be 2D [m, d].");       // 输入 [m, d]
    ASSERT(weight->ndim() == 1, "RmsNorm: weight must be 1D [d].");  // 权重 [d]
    ASSERT(out->ndim() == 2, "RmsNorm: out must be 2D [m, d].");

    size_t m = in->shape()[0];   // 行数（样本数）
    size_t d = in->shape()[1];   // 每行长度（维度数）

    // 形状约束：out 与 in 同形状；weight 长度 = 每行维度数
    ASSERT(out->shape()[0] == m && out->shape()[1] == d, "RmsNorm: out shape mismatch.");
    ASSERT(weight->shape()[0] == d, "RmsNorm: weight length must match in dim.");

    // -----------------------------------------------------------------------
    // 2. 按设备分发
    // -----------------------------------------------------------------------
    if (out->deviceType() == CHAOSUAN_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), m, d);
    }

    chaosuan::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case CHAOSUAN_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), m, d);
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
