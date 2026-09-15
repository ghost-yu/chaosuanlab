// ============================================================================
// swiglu_cpu.cpp —— swiglu 算子的 CPU 实现
// ----------------------------------------------------------------------------
// 【SwiGLU 是什么？】
// 大模型 MLP（多层感知机）里的激活函数，Qwen/Llama 都用它。
// MLP 有两路输入：
//   gate（门控）：先过 sigmoid 变成 0~1 的"开关"；
//   up（上升） ：要放大的值。
// 公式：out[i] = up[i] * gate[i] * sigmoid(gate[i])
//      = up[i] * gate[i] / (1 + e^(-gate[i]))
// 直观理解：gate 值大 → sigmoid 接近 1 → up 几乎原样通过；
//          gate 值小（负）→ sigmoid 接近 0 → 输出被"关掉"。
// 这比 ReLU 更平滑（可微），训练更稳。
//
// 纯逐元素计算：每个下标 i 独立，互不影响（天然适合并行/GPU）。
// ============================================================================

#include "swiglu_cpu.hpp"  // 本文件函数的声明（在头文件里）。

#include <cmath>      // C++ 标准库：std::exp（自然指数 e^x）。

#include "../../../utils.hpp"  // 项目公共工具（cast、异常宏）。

// ---------------------------------------------------------------------------
// 【模板函数】template <typename T>：T 是类型参数（float / fp16_t / bf16_t）。
// 参数：
//   out   —— 输出指针（逐元素结果，长度 numel）；
//   gate  —— 门控输入指针（长度 numel）；
//   up    —— 上升输入指针（长度 numel）；
//   numel —— 元素总个数（numel = number of elements）。
// ---------------------------------------------------------------------------
template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        // if constexpr：编译期判断 T 是否低精度类型。
        if constexpr (std::is_same_v<T, chaosuan::bf16_t> || std::is_same_v<T, chaosuan::fp16_t>) {
            // 低精度分支：先转 float 算（CPU 没有低精度指数指令，且精度差），再转回。
            float g = chaosuan::utils::cast<float>(gate[i]);  // 门控值
            float u = chaosuan::utils::cast<float>(up[i]);    // 上升值
            // sigmoid(g) = 1 / (1 + e^(-g))。
            // 除以 (1 + e^(-g)) 等价于乘以 sigmoid(g)。
            out[i] = chaosuan::utils::cast<T>(u * g / (1.0f + std::exp(-g)));
        } else {
            // float 分支：直接算（float 精度足够，不用转换）。
            out[i] = up[i] * gate[i] / (1.0f + std::exp(-gate[i]));
        }
    }
}

// ---------------------------------------------------------------------------
// 【命名空间嵌套】chaosuan::ops::cpu = 项目::算子::CPU 实现（防重名）。
// ---------------------------------------------------------------------------
namespace chaosuan::ops::cpu {

// ---------------------------------------------------------------------------
// 对外统一入口 swiglu()：按 dtype 选模板实例，字节指针转具体类型指针。
// 参数：
//   out   —— 输出指针（std::byte* 原始字节）；
//   gate  —— 门控输入指针；
//   up    —— 上升输入指针；
//   dtype —— 数据类型枚举（chaosuanDataType_t）；
//   numel —— 元素总个数。
// ---------------------------------------------------------------------------
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            chaosuanDataType_t dtype, size_t numel) {
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        // reinterpret_cast：字节指针 -> float 指针（内容不变，只改"看待方式"）。
        return swiglu_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                              reinterpret_cast<const float *>(up), numel);
    case CHAOSUAN_DTYPE_BF16:
        return swiglu_<chaosuan::bf16_t>(reinterpret_cast<chaosuan::bf16_t *>(out),
                                         reinterpret_cast<const chaosuan::bf16_t *>(gate),
                                         reinterpret_cast<const chaosuan::bf16_t *>(up), numel);
    case CHAOSUAN_DTYPE_F16:
        return swiglu_<chaosuan::fp16_t>(reinterpret_cast<chaosuan::fp16_t *>(out),
                                         reinterpret_cast<const chaosuan::fp16_t *>(gate),
                                         reinterpret_cast<const chaosuan::fp16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);  // 其它类型不支持
    }
}
} // namespace chaosuan::ops::cpu
