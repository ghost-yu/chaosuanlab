// ============================================================================
// op.cpp —— argmax 算子的实现：参数校验 + 按设备分发
// ----------------------------------------------------------------------------
// 每个算子的"调度入口"都长这样：
//   1. 校验参数（同设备、同类型、连续内存、形状正确）；
//   2. 从张量元信息里取出计算需要的维度数字；
//   3. 按设备类型把"字节指针"交给对应的后端（CPU / CUDA）。
// 真正算数的是 cpu/argmax_cpu.cpp（本算子 CPU 后端）。
// ============================================================================

#include "op.hpp"  // 本算子声明。

#include "../../core/chaosuan_core.hpp"  // 核心上下文（切换设备用）。
#include "../../utils.hpp"               // 项目公共工具（校验宏）。

#include "cpu/argmax_cpu.hpp"            // CPU 实现头文件。
#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"      // CUDA 实现头文件（作业 4 再加）。
#endif

namespace chaosuan::ops {

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // -----------------------------------------------------------------------
    // 1. 参数校验
    // -----------------------------------------------------------------------
    CHECK_SAME_DEVICE(max_idx, max_val, vals);                          // 三个张量在同一设备
    CHECK_SAME_DTYPE(vals->dtype(), max_val->dtype());                  // vals 与 max_val 同类型
    ASSERT(max_idx->dtype() == CHAOSUAN_DTYPE_I64,
           "Argmax: max_idx must be int64.");                           // 下标固定用 int64
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "Argmax: all tensors must be contiguous.");                  // 连续内存
    ASSERT(vals->ndim() == 1, "Argmax: vals must be 1D.");              // 一维输入
    ASSERT(max_idx->ndim() == 1 && max_val->ndim() == 1,
           "Argmax: max_idx and max_val must be 1D.");                  // 输出都是单元素
    ASSERT(max_idx->shape()[0] == 1 && max_val->shape()[0] == 1,
           "Argmax: max_idx and max_val must have shape [1].");

    size_t numel = vals->shape()[0];  // 元素个数 = 一维数组长度

    // -----------------------------------------------------------------------
    // 2. 按设备分发
    // -----------------------------------------------------------------------
    // CPU 快速路径：张量就在 CPU 上，直接调 CPU 实现。
    if (max_idx->deviceType() == CHAOSUAN_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), numel);
    }

    // 非 CPU：先切换设备上下文，再走 switch 分发（作业 4 会补 NVIDIA 分支）。
    chaosuan::core::context().setDevice(max_idx->deviceType(), max_idx->deviceId());

    switch (max_idx->deviceType()) {
    case CHAOSUAN_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), numel);
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
