import io
import os

# 每个算子的占位头文件：声明 nvidia 命名空间函数（4.2 填真实 kernel）。
# 头文件同时被 op.cpp（g++）和 .cu（nvcc）包含，所以只写声明。
PLACEHOLDER = """// ============================================================================
// {name}_nvidia.cuh —— {desc} 的 CUDA 实现声明（作业 4）
// ----------------------------------------------------------------------------
// 这个头文件只声明"函数长什么样"，真正的 CUDA kernel 在
// 同目录的 {name}_nvidia.cu 里（nvcc 编译）。
// op.cpp 在 ENABLE_NVIDIA_API 下 include 本文件并调用这些函数。
// ============================================================================

#pragma once

#include "chaosuan.h"  // 数据类型枚举。

#include <cstddef>     // std::byte、size_t。

namespace chaosuan::ops::nvidia {{
{decls}
}} // namespace chaosuan::ops::nvidia
"""

# 各算子的函数声明（参数与 CPU 版一致）
DECLS = {
    "add": ("add（逐元素相加）", "void add(std::byte *c, const std::byte *a, const std::byte *b, chaosuanDataType_t dtype, size_t numel);"),
    "argmax": ("argmax（最大值下标）", "void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, chaosuanDataType_t dtype, size_t numel);"),
    "embedding": ("embedding（查表）", "void embedding(std::byte *out, const std::byte *index, const std::byte *weight, chaosuanDataType_t dtype, size_t numel, size_t embedding_dim);"),
    "linear": ("linear（全连接）", "void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, chaosuanDataType_t dtype, size_t m, size_t n, size_t k);"),
    "rms_norm": ("rms_norm（归一化）", "void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, chaosuanDataType_t dtype, size_t m, size_t d);"),
    "rope": ("rope（旋转位置编码）", "void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, chaosuanDataType_t dtype, size_t seqlen, size_t nheads, size_t d);"),
    "self_attention": ("self_attention（注意力）", "void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, chaosuanDataType_t dtype, size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv);"),
    "swiglu": ("swiglu（门控激活）", "void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, chaosuanDataType_t dtype, size_t numel);"),
}

ROOT = r"G:\shi\chaosuan\src\ops"

for op, (desc, decl) in DECLS.items():
    d = os.path.join(ROOT, op, "nvidia")
    os.makedirs(d, exist_ok=True)
    content = PLACEHOLDER.format(name=op, desc=desc, decls=decl)
    with io.open(os.path.join(d, f"{op}_nvidia.cuh"), "w", encoding="utf-8") as f:
        f.write(content)
    print("created", op, "_nvidia.cuh")

# add/op.cpp 没有 include nvidia 头，补上 include + 分支调用
p = os.path.join(ROOT, "add", "op.cpp")
s = io.open(p, encoding="utf-8").read()
old = '#include "cpu/add_cpu.hpp"\n'
new = '#include "cpu/add_cpu.hpp"\n#ifdef ENABLE_NVIDIA_API\n#include "nvidia/add_nvidia.cuh"\n#endif\n'
assert old in s
s = s.replace(old, new)
old2 = """#ifdef ENABLE_NVIDIA_API
    case CHAOSUAN_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif"""
new2 = """#ifdef ENABLE_NVIDIA_API
    case CHAOSUAN_DEVICE_NVIDIA:
        return nvidia::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
#endif"""
assert old2 in s
s = s.replace(old2, new2)
io.open(p, "w", encoding="utf-8").write(s)
print("patched add/op.cpp")
