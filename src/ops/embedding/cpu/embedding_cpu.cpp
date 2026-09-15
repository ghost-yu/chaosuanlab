// ============================================================================
// embedding_cpu.cpp —— embedding（嵌入）算子的 CPU 实现
// ----------------------------------------------------------------------------
// 【embedding 是什么？】
// 大模型里，每个"词"不是直接喂给网络的，而是先查一张"词向量表"
// （weight）：表里每一行是一个词的向量表示。
// embedding 操作 = 给定一串词的编号（index），把每个编号对应的那行
// 向量取出来，按顺序拼成输出（out）。
//   例：index = [2, 0, 5]，weight 有 6 行，每行 4 个数
//       → out = weight[2] 接 weight[0] 接 weight[5]（3 行，每行 4 个数）。
//
// 本质是"内存搬运"：根据编号把 weight 的某一行整段拷贝到 out。
// 用标准库 std::memcpy 一次性拷一行，比逐元素 for 循环赋值快很多
// （memcpy 底层按大块搬运，CPU 有专门优化）。
// ============================================================================

#include "embedding_cpu.hpp"  // 本文件函数的声明（在头文件里）。

#include "../../../utils.hpp" // 项目公共工具（cast、异常宏）。

#include <cstring>    // C++ 标准库：std::memcpy（内存块拷贝函数）。

// ---------------------------------------------------------------------------
// 【模板函数】template <typename T>：T 是类型参数，调用时被替换成
// float / bf16_t / fp16_t，一份代码供多种数据类型共用。
//
// 参数：
//   T *out                —— 输出：拼接好的向量表（行数 = numel，每行 embedding_dim 个元素）；
//   const int64_t *index  —— 输入：词的编号数组（每个编号 64 位整数）；
//                           注意 index 的类型固定是 int64_t，与 T 无关；
//   const T *weight       —— 输入：词向量表（weight[行号][列号]）；
//   size_t numel          —— index 里有多少个编号（即输出有几行）；
//   size_t embedding_dim  —— 每行向量的长度（多少个数）。
// ---------------------------------------------------------------------------
template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t numel, size_t embedding_dim) {
    for (size_t i = 0; i < numel; i++) {            // 遍历每个待查找的行号
        int64_t idx = index[i];                     // 取出第 i 个索引：要从 weight 第 idx 行取数据

        // 把 weight 的第 idx 行整体拷贝到 out 的第 i 行位置：
        //   源地址   = &weight[idx * embedding_dim]（第 idx 行的开头）；
        //   目标地址 = &out[i * embedding_dim]（输出第 i 行的开头）；
        //   字节数   = embedding_dim * sizeof(T)（一行有 embedding_dim 个元素，
        //             每个 sizeof(T) 字节）。
        // 指针算术：weight + idx * embedding_dim 表示"往后跳 idx*embedding_dim
        // 个 T 元素"，就是第 idx 行开头（每行正好 embedding_dim 个元素）。
        // 为什么用 memcpy 而不是 for 循环？memcpy 是标准库高度优化的内存拷贝，
        // 按大块搬运，比逐元素赋值快得多；embedding 本质就是大批量行拷贝。
        std::memcpy(&out[i * embedding_dim], &weight[idx * embedding_dim], embedding_dim * sizeof(T));
    }
}

// ---------------------------------------------------------------------------
// 【命名空间嵌套】chaosuan::ops::cpu = 项目::算子::CPU 实现（防重名）。
// ---------------------------------------------------------------------------
namespace chaosuan::ops::cpu {

// ---------------------------------------------------------------------------
// 对外统一入口 embedding()：
// 数据在内存里只是"一串字节"，按 float 看还是按 bf16 看取决于 dtype。
// 函数用 switch 选对模板实例，再把"字节指针" reinterpret_cast
// （重新解释，见 tensor.cpp 导读）成"具体类型指针"。
//
// 参数（注意都是 std::byte* —— 原始字节指针）：
//   out    —— 输出：拼接后的向量表；
//   index  —— 输入：词的编号数组（内部按 int64 解释）；
//   weight —— 输入：词向量表；
//   dtype  —— 数据类型枚举（chaosuanDataType_t）；
//   numel  —— index 长度（输出行数）；
//   embedding_dim —— 每行向量长度。
// ---------------------------------------------------------------------------
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               chaosuanDataType_t dtype, size_t numel, size_t embedding_dim) {
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        // reinterpret_cast：字节指针 -> 具体类型指针（内容不变，只改"看待方式"）。
        // index 固定是 int64_t*，与 T 无关。
        return embedding_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index),
                                 reinterpret_cast<const float *>(weight), numel, embedding_dim);
    case CHAOSUAN_DTYPE_BF16:
        return embedding_<chaosuan::bf16_t>(reinterpret_cast<chaosuan::bf16_t *>(out), reinterpret_cast<const int64_t *>(index),
                                            reinterpret_cast<const chaosuan::bf16_t *>(weight), numel, embedding_dim);
    case CHAOSUAN_DTYPE_F16:
        return embedding_<chaosuan::fp16_t>(reinterpret_cast<chaosuan::fp16_t *>(out), reinterpret_cast<const int64_t *>(index),
                                            reinterpret_cast<const chaosuan::fp16_t *>(weight), numel, embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);  // 其它类型不支持：报错
    }
}
} // namespace chaosuan::ops::cpu
