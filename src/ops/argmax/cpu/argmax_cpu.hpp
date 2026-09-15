// ============================================================================
// argmax_cpu.hpp —— argmax 算子的 CPU 实现【头文件/声明】
// ----------------------------------------------------------------------------
// C++ 惯例：头文件（.hpp）只写"这个函数长什么样"（签名/声明），
// 不写"怎么实现"；实现放在同名的 .cpp 文件里。
// 这样其他文件只要 #include 这个头文件，就能调用 argmax()，
// 编译器只需要知道参数和返回类型，不需要看到实现细节。
// ============================================================================

#pragma once  // 预处理指令：保证这个头文件即使被多个 .cpp include 多次，
              // 内容也只展开一次，避免"重复定义"错误。

#include "chaosuan.h"  // C API 公共头文件：里面定义了 chaosuanDataType_t
                       // （数据类型枚举，见 tensor.cpp 里对该类型的详解）。

#include <cstddef>    // C++ 标准库：std::byte（原始字节）、size_t（无符号大小）。

// 命名空间：chaosuan::ops::cpu（项目名::算子::CPU 实现）。
namespace chaosuan::ops::cpu {

// 计算 vals 的最大值及其下标，分别写入 max_val 和 max_idx。
// 参数说明（全部是"裸指针"，即 C 风格的指针，方便被 C API 调用）：
//   max_idx：输出指针，指向 int64 类型的单元素 —— 最大值的下标
//   max_val：输出指针，指向与 vals 同类型的单元素 —— 最大值
//   vals   ：输入指针，指向一维连续数据（本算子的数据来源）
//   dtype  ：vals / max_val 的数据类型（chaosuanDataType_t 枚举）
//   numel  ：vals 的元素个数（size_t = 无符号整数）
// 注意：max_idx / max_val 都是"单元素"（下标只有 1 个、最大值只有 1 个）。
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, chaosuanDataType_t dtype, size_t numel);
}
