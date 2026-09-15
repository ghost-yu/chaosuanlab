// ============================================================================
// op.cpp —— swiglu 算子的实现：参数校验 + 按设备分发
// ----------------------------------------------------------------------------
// 调度入口套路：校验 → 取元素个数 → 按设备交给后端。
// 真正算数的是 cpu/swiglu_cpu.cpp。
// ============================================================================

#include "op.hpp"  // 本算子声明。

#include "../../core/chaosuan_core.hpp"  // 核心上下文（切换设备用）。
#include "../../utils.hpp"               // 项目公共工具（校验宏）。

#include "cpu/swiglu_cpu.hpp"            // CPU 实现头文件。
#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.cuh"      // CUDA 实现头文件（作业 4 再加）。
#endif

namespace chaosuan::ops {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // -----------------------------------------------------------------------
    // 1. 参数校验
    // -----------------------------------------------------------------------
    CHECK_SAME_DEVICE(out, gate, up);                          // 三个张量同设备
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype()); // 同类型
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: all tensors must be contiguous.");         // 连续内存
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape()); // 三个张量同形状

    size_t numel = out->numel();  // 元素总个数

    // -----------------------------------------------------------------------
    // 2. 按设备分发
    // -----------------------------------------------------------------------
    if (out->deviceType() == CHAOSUAN_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
    }

    chaosuan::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case CHAOSUAN_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
#ifdef ENABLE_NVIDIA_API
    case CHAOSUAN_DEVICE_NVIDIA:
        return nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace chaosuan::ops
