// ============================================================================
// op.cpp —— linear 算子的实现：参数校验 + 按设备分发
// ----------------------------------------------------------------------------
// 与前面的算子结构一致，但多了一个"可选参数"bias：
//   - bias 可能为 nullptr（没有偏置），此时不再要求它满足设备/类型/形状检查；
//   - 传给后端时，bias 有值就传数据指针，没有就传 nullptr。
// ============================================================================

#include "op.hpp"  // 本算子声明。

#include "../../core/chaosuan_core.hpp"  // 核心上下文（切换设备用）。
#include "../../utils.hpp"               // 项目公共工具（校验宏）。

#include "cpu/linear_cpu.hpp"            // CPU 实现头文件。
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"      // CUDA 实现头文件（作业 4 再加）。
#endif

namespace chaosuan::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // -----------------------------------------------------------------------
    // 1. 参数校验
    // -----------------------------------------------------------------------
    CHECK_SAME_DEVICE(out, in, weight);                                 // 三个必选张量同设备
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());       // 同类型
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: all tensors must be contiguous.");                  // 连续内存
    ASSERT(in->ndim() == 2, "Linear: in must be 2D.");       // 输入 [m, k]
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D.");  // 权重 [n, k]
    ASSERT(out->ndim() == 2, "Linear: out must be 2D.");     // 输出 [m, n]
    // bias 是可选参数：只有非空时才做完整检查（设备/类型/连续/维度）。
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
        ASSERT(bias->ndim() == 1, "Linear: bias must be 1D.");  // 偏置 [n]
    }

    size_t m = in->shape()[0];      // 样本数 / 行数
    size_t k = in->shape()[1];      // 输入特征数（weight 的列数必须等于它）
    size_t n = weight->shape()[0];  // 输出特征数（weight 的行数 = 输出列数）
    size_t kk = weight->shape()[1];

    // 矩阵乘法维度约束：in 的列数 == weight 的列数（这就是 k）。
    ASSERT(k == kk, "Linear: input last dim must match weight last dim.");
    ASSERT(out->shape()[0] == m && out->shape()[1] == n, "Linear: out shape mismatch.");
    if (bias) {
        ASSERT(bias->shape()[0] == n, "Linear: bias shape mismatch.");  // 偏置长度 = 输出特征数
    }

    // -----------------------------------------------------------------------
    // 2. 按设备分发
    // -----------------------------------------------------------------------
    // 注意 bias ? bias->data() : nullptr —— 没有偏置就传 nullptr 给后端。
    if (out->deviceType() == CHAOSUAN_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), m, n, k);
    }

    chaosuan::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case CHAOSUAN_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), m, n, k);
#ifdef ENABLE_NVIDIA_API
    case CHAOSUAN_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), m, n, k);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace chaosuan::ops
