// ============================================================================
// op.cpp —— embedding 算子的实现：参数校验 + 按设备分发
// ----------------------------------------------------------------------------
// 调度入口套路：校验 → 取维度 → 按设备交给后端。
// 真正算数的是 cpu/embedding_cpu.cpp。
// ============================================================================

#include "op.hpp"  // 本算子声明。

#include "../../core/chaosuan_core.hpp"  // 核心上下文（切换设备用）。
#include "../../utils.hpp"               // 项目公共工具（校验宏）。

#include "cpu/embedding_cpu.hpp"         // CPU 实现头文件。
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"   // CUDA 实现头文件（作业 4 再加）。
#endif

namespace chaosuan::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // -----------------------------------------------------------------------
    // 1. 参数校验
    // -----------------------------------------------------------------------
    CHECK_SAME_DEVICE(out, index, weight);                              // 三个张量同设备
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());                    // out 与 weight 同类型
    ASSERT(index->dtype() == CHAOSUAN_DTYPE_I64,
           "Embedding: index must be int64.");                          // 词编号用 int64
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");               // 连续内存
    ASSERT(index->ndim() == 1, "Embedding: index must be 1D.");         // 编号是一维
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D [vocab, dim].");
    ASSERT(out->ndim() == 2, "Embedding: out must be 2D [seq, dim].");

    size_t numel = index->shape()[0];               // 有几个编号（输出几行）
    size_t embedding_dim = weight->shape()[1];      // 每行向量长度
    // 形状约束：输出行数 = 编号个数；输出列数 = 向量维度；编号不能越界（< 词表行数）
    ASSERT(out->shape()[0] == numel, "Embedding: out rows must match index length.");
    ASSERT(out->shape()[1] == embedding_dim, "Embedding: out cols must match weight cols.");
    ASSERT(weight->shape()[0] >= 1, "Embedding: weight must be non-empty.");

    // -----------------------------------------------------------------------
    // 2. 按设备分发
    // -----------------------------------------------------------------------
    if (out->deviceType() == CHAOSUAN_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), numel, embedding_dim);
    }

    chaosuan::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case CHAOSUAN_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), numel, embedding_dim);
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
