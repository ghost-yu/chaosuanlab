// ============================================================================
// argmax_cpu.cpp —— argmax 算子的 CPU 实现
// ----------------------------------------------------------------------------
// 【argmax 是什么？】
// argmax = "argument of the maximum"（最大值的自变量/下标）。
// 给定一串数，argmax 找出"最大的数是第几个"以及"它本身是多少"。
// 例：vals = [3, 7, 2, 9, 1] → max_idx = 3（第 3 个元素最大），max_val = 9。
// 大模型里用它做"贪心解码"：模型输出每个词的概率，取概率最大的那个词。
//
// 本文件是 CPU 版：单线程顺序扫描，从第 0 个元素开始依次比较，
// 遇到更大的就更新"当前最大值 + 当前最大下标"。一趟遍历结束即得结果。
// （CUDA 版 argmax_nvidia.cu 用成千上万个线程并行做"归约"，原理不同，
//   那是作业 4 的内容。）
// ============================================================================

#include "argmax_cpu.hpp"  // 本文件函数的声明（在头文件里）。

#include "../../../utils.hpp"  // 项目公共工具：utils::cast（类型转换）、异常宏。

#include <cstdint>    // C++ 标准库：int64_t（64 位有符号整数，即 long long）。
                      // 下标可能很大（数亿个元素），所以用 64 位整数存。

// ---------------------------------------------------------------------------
// 【模板函数】（C++ 语法，见 tensor.cpp 文件头的导读）
// template <typename T> 表示"T 是一个类型参数"。
// 调用 argmax_<float>(...) 时，编译器用 float 替换 T，生成一份专门处理
// float 的代码；调用 argmax_<chaosuan::bf16_t>(...) 时再生成一份 bf16 的。
// 这样"一份代码，多种类型共用"，不用为每种类型复制粘贴一遍。
//
// 参数：
//   int64_t *max_idx —— 输出：最大值的下标（指针 = 通过它把结果写回外面）；
//   T *max_val        —— 输出：最大值本身（T 类型，与输入元素同类型）；
//   const T *vals     —— 输入：一维数据数组（const 表示函数内不能改它）；
//   size_t numel      —— 输入：数组元素个数（size_t = 无符号整数，装大小）。
// ---------------------------------------------------------------------------
template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    // 初始化：先假设第 0 个元素就是最大的（后面从第 1 个开始比）。
    T best_val = vals[0];   // best = 当前找到的最大值（先拿第 0 个当"种子"）
    int64_t best_idx = 0;   // best_idx = 当前最大值的下标（第 0 个 → 0）

    // -----------------------------------------------------------------------
    // 【if constexpr】编译期 if：
    // std::is_same_v<T, chaosuan::bf16_t> 是一个"类型判断"，编译时就能确定
    // T 到底是不是 bf16。如果是，编译器只保留 true 分支的代码；
    // 否则只保留 else 分支。不会两个分支都编译。
    //
    // 为什么要分两支？
    //   bf16_t / fp16_t 是"低精度浮点"（各占 2 字节），CPU 上没有直接比较
    //   它们的硬件指令，直接写 vals[i] > best_val 无法编译/结果不对。
    //   所以先把每个元素转成 float（utils::cast<float>），用 float 比较，
    //   比较完再把"原始类型"的最大值记下来（写回时就不需要再转了）。
    // -----------------------------------------------------------------------
    if constexpr (std::is_same_v<T, chaosuan::bf16_t> || std::is_same_v<T, chaosuan::fp16_t>) {
        // best_f：用 float 形式维护"当前最大值"（方便比较）。
        float best_f = chaosuan::utils::cast<float>(best_val);

        // 从第 1 个元素开始逐个扫描（第 0 个已经当种子了）。
        for (size_t i = 1; i < numel; i++) {
            // 把当前元素转成 float 再比较。
            float cur_f = chaosuan::utils::cast<float>(vals[i]);
            if (cur_f > best_f) {          // 发现更大的
                best_f = cur_f;            //   更新 float 版最大值
                best_val = vals[i];        //   同步更新原始类型版最大值（bf16/fp16）
                best_idx = static_cast<int64_t>(i);   //   记录下标（i 是 size_t，转成 int64_t）
            }
        }
    } else {
        // 普通类型（float 等）：CPU 直接支持比较，不用转换。
        for (size_t i = 1; i < numel; i++) {
            // 严格大于（>）才更新：如果相等不更新，
            // 效果是"多个最大值并列时，取下标最小的那个"（稳定）。
            if (vals[i] > best_val) {
                best_val = vals[i];
                best_idx = static_cast<int64_t>(i);
            }
        }
    }

    // 把结果写回调用者：max_idx / max_val 是指针，
    // *max_idx = ... 表示"往指针指向的位置写入"，这就是 C 里常见的
    // "输出参数"写法（函数返回 void，靠指针把结果带出去）。
    *max_idx = best_idx;
    *max_val = best_val;
}

// ---------------------------------------------------------------------------
// 【命名空间嵌套】namespace chaosuan::ops::cpu 是三个命名空间套在一起：
// chaosuan::ops::cpu，等价于 namespace chaosuan { namespace ops { namespace cpu { ... } } }
// 它的作用是：这个函数全名是 chaosuan::ops::cpu::argmax，不会和别处重名。
// ---------------------------------------------------------------------------
namespace chaosuan::ops::cpu {

// ---------------------------------------------------------------------------
// 对外统一入口 argmax()：
// 数据在内存里只是"一串字节"，到底当 float 看还是当 bf16 看，取决于 dtype。
// 这个函数根据 dtype 用 switch 选对模板实例，然后把"字节指针" reinterpret_cast
// （重新解释：内容不变，只改"看待方式"）成"具体类型指针"再调用 argmax_。
//
// 参数（注意都是 std::byte* —— 原始字节指针）：
//   max_idx —— 输出：最大值下标（内部按 int64 解释，占 1 个元素）；
//   max_val —— 输出：最大值（按 dtype 解释，占 1 个元素）；
//   vals    —— 输入：一维数据（按 dtype 解释）；
//   dtype   —— 数据类型枚举（chaosuanDataType_t，见 chaosuan.h）；
//   numel   —— 元素个数。
// ---------------------------------------------------------------------------
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, chaosuanDataType_t dtype, size_t numel) {
    // switch：按数据类型分发。每个 case 调用一次模板，参数全部 reinterpret_cast。
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        // max_idx 解释成 int64_t*，max_val 解释成 float*，vals 解释成 const float*。
        return argmax_<float>(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val),
                              reinterpret_cast<const float *>(vals), numel);
    case CHAOSUAN_DTYPE_BF16:
        return argmax_<chaosuan::bf16_t>(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<chaosuan::bf16_t *>(max_val),
                                         reinterpret_cast<const chaosuan::bf16_t *>(vals), numel);
    case CHAOSUAN_DTYPE_F16:
        return argmax_<chaosuan::fp16_t>(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<chaosuan::fp16_t *>(max_val),
                                         reinterpret_cast<const chaosuan::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);  // 其它类型（如 I32）暂不支持，抛异常
    }
}
} // namespace chaosuan::ops::cpu
