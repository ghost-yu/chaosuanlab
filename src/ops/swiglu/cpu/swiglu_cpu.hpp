// ============================================================================
// swiglu_cpu.hpp —— swiglu 算子的 CPU 实现【头文件/声明】
// ----------------------------------------------------------------------------
// 头文件只写函数"长什么样"，实现写在同名的 .cpp 里。
// ============================================================================

#pragma once  // 防止头文件被重复包含导致"重复定义"错误。

#include "chaosuan.h"  // 数据类型枚举 chaosuanDataType_t。

#include <cstddef>    // std::byte、size_t。

namespace chaosuan::ops::cpu {

// SwiGLU 门控激活：out[i] = up[i] * gate[i] / (1 + e^(-gate[i]))
//   out   —— 输出指针（逐元素结果，长度 numel）；
//   gate  —— 门控输入指针（长度 numel）；
//   up    —— 上升输入指针（长度 numel）；
//   dtype —— 数据类型枚举（chaosuanDataType_t）；
//   numel —— 元素总个数。
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            chaosuanDataType_t dtype, size_t numel);
}
