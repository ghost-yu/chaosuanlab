// ============================================================================
// op.cpp —— self_attention 算子的实现：参数校验 + 按设备分发
// ----------------------------------------------------------------------------
// 校验重点：q/k/v 都是 3 维；GQA 要求 nhead 是 nkvhead 的整数倍；
// total_len（k 的行数）>= seqlen（q 的行数）。
// 真正算数的是 cpu/self_attention_cpu.cpp。
// ============================================================================

#include "op.hpp"  // 本算子声明。

#include "../../core/chaosuan_core.hpp"  // 核心上下文（切换设备用）。
#include "../../utils.hpp"               // 项目公共工具（校验宏）。

#include "cpu/self_attention_cpu.hpp"    // CPU 实现头文件。
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"  // CUDA 实现头文件（作业 4 再加）。
#endif

namespace chaosuan::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // -----------------------------------------------------------------------
    // 1. 参数校验
    // -----------------------------------------------------------------------
    CHECK_SAME_DEVICE(attn_val, q, k, v);                                // 四个张量同设备
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());  // 同类型
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");            // 连续内存
    ASSERT(q->ndim() == 3, "SelfAttention: q must be 3D [seqlen, nhead, d].");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be 3D [total_len, nkvhead, d].");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be 3D [total_len, nkvhead, dv].");
    ASSERT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3D [seqlen, nhead, dv].");

    size_t seqlen = q->shape()[0];    // 当前序列长度（q 的行数）
    size_t total_len = k->shape()[0]; // k/v 总长度（历史 KV 缓存 + 当前序列）
    size_t nhead = q->shape()[1];     // 查询头数
    size_t nkvhead = k->shape()[1];   // 键值头数
    size_t d = q->shape()[2];         // 每个查询头的维度（= k 的维度）
    size_t dv = v->shape()[2];        // 每个值头的维度

    // GQA：查询头数必须是键值头数的整数倍（nrep >= 1）
    ASSERT(nhead % nkvhead == 0, "SelfAttention: nhead must be a multiple of nkvhead.");
    ASSERT(k->shape()[2] == d, "SelfAttention: k head_dim must match q head_dim.");
    ASSERT(total_len >= seqlen, "SelfAttention: kv length must be >= query length.");
    // 形状约束：attn_val 形状 = q 的形状但最后一维换成 dv
    ASSERT(attn_val->shape()[0] == seqlen && attn_val->shape()[1] == nhead && attn_val->shape()[2] == dv,
           "SelfAttention: attn_val shape mismatch.");
    ASSERT(v->shape()[0] == total_len && v->shape()[1] == nkvhead,
           "SelfAttention: v shape must match k shape except last dim.");

    // -----------------------------------------------------------------------
    // 2. 按设备分发
    // -----------------------------------------------------------------------
    if (attn_val->deviceType() == CHAOSUAN_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
                                   seqlen, total_len, nhead, nkvhead, d, dv);
    }

    chaosuan::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case CHAOSUAN_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
                                   seqlen, total_len, nhead, nkvhead, d, dv);
#ifdef ENABLE_NVIDIA_API
    case CHAOSUAN_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
                                   seqlen, total_len, nhead, nkvhead, d, dv);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace chaosuan::ops
