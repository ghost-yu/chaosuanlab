// ============================================================================
// linear_cpu.cpp —— linear（全连接层）算子的 CPU 实现
// ----------------------------------------------------------------------------
// 【linear 是什么？】
// 神经网络里最常见的层：y = x·W^T + b
//   x：输入，形状 [m, k]（m 个样本，每个 k 个特征）；
//   W：权重，形状 [n, k]（n 个输出神经元，每个连 k 个输入）；
//   b：偏置，形状 [n]（每个输出一个偏置，可选）；
//   y：输出，形状 [m, n]（每个样本得到 n 个输出值）。
// 数学上 y[i][j] = Σ_kk (x[i][kk] * W[j][kk]) + b[j]，
// 即"输入第 i 行" 与 "权重第 j 行" 做点积，再加偏置。
//
// 本文件用**三重循环的朴素矩阵乘法**实现（最简单，但 CPU 上不是最快的；
// 优化版会用 BLAS/多线程/向量化，那是性能课的范畴）。
//   for i in m:          // 遍历输出行（每个样本）
//     for j in n:        // 遍历输出列（每个输出神经元）
//       sum = 点积(输入第 i 行, 权重第 j 行)
//       sum += bias[j]
//       out[i][j] = sum
//
// 注意索引关系：weight 是 [n, k] 且**未转置**，我们要的是 W^T 的效果，
// 所以取 weight[j][kk]（第 j 行第 kk 列），而不是 weight[kk][j]。
// ============================================================================

#include "linear_cpu.hpp"  // 本文件函数的声明（在头文件里）。

#include "../../../utils.hpp"  // 项目公共工具（cast、异常宏）。

// ---------------------------------------------------------------------------
// 【模板函数】template <typename T>：T 是类型参数（float / fp16_t / bf16_t）。
// 关键设计：无论输入是什么类型，**累加器 sum 一律用 float**！
// 原因：
//   1. 低精度类型（fp16/bf16）没有 CPU 累加指令，直接加会出错/变慢；
//   2. float 精度更高，累加过程中误差小（数值稳定性）。
// ===========================================================================
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t n, size_t k) {
    // fp16/bf16 分支：每次乘法前把两个数转成 float，用 float 累加。
    if constexpr (std::is_same_v<T, chaosuan::bf16_t> || std::is_same_v<T, chaosuan::fp16_t>) {
        for (size_t i = 0; i < m; i++) {              // i：遍历输出行（样本）
            for (size_t j = 0; j < n; j++) {          // j：遍历输出列（神经元）
                float sum = 0.0f;                     // 点积累加器（float，见上面说明）
                for (size_t kk = 0; kk < k; kk++) {   // kk：遍历输入特征
                    // 输入第 i 行第 kk 个 × 权重第 j 行第 kk 个，转 float 相乘再累加。
                    // in[i * k + kk]：第 i 行偏移 i*k，再偏移 kk（每行 k 个元素）。
                    // weight[j * k + kk]：第 j 行偏移 j*k，再偏移 kk。
                    sum += chaosuan::utils::cast<float>(in[i * k + kk]) * chaosuan::utils::cast<float>(weight[j * k + kk]);
                }
                if (bias) sum += chaosuan::utils::cast<float>(bias[j]);  // 加偏置（若有）
                out[i * n + j] = chaosuan::utils::cast<T>(sum);          // 结果转回低精度类型写回
            }
        }
    } else {
        // float 分支：float 与 float 相乘还是 float，直接算（不用转换）。
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                float sum = 0.0f;
                for (size_t kk = 0; kk < k; kk++) {
                    sum += in[i * k + kk] * weight[j * k + kk];
                }
                if (bias) sum += bias[j];
                out[i * n + j] = sum;   // float 写回 float
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 【命名空间嵌套】chaosuan::ops::cpu = 项目::算子::CPU 实现（防重名）。
// ---------------------------------------------------------------------------
namespace chaosuan::ops::cpu {

// ---------------------------------------------------------------------------
// 对外统一入口 linear()：按 dtype 选模板实例，字节指针转具体类型指针。
// 参数：
//   out    —— 输出：形状 [m, n] 的结果矩阵；
//   in     —— 输入：形状 [m, k]；
//   weight —— 输入：形状 [n, k]（未转置）；
//   bias   —— 输入：形状 [n] 的偏置（可为 nullptr = 没有偏置）；
//   dtype  —— 数据类型枚举（chaosuanDataType_t）；
//   m, n, k —— 三个维度大小（见上方数学说明）。
// ---------------------------------------------------------------------------
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            chaosuanDataType_t dtype, size_t m, size_t n, size_t k) {
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        // reinterpret_cast：字节指针 -> float 指针（内容不变，只改"看待方式"）。
        return linear_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                              reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), m, n, k);
    case CHAOSUAN_DTYPE_BF16:
        return linear_<chaosuan::bf16_t>(reinterpret_cast<chaosuan::bf16_t *>(out), reinterpret_cast<const chaosuan::bf16_t *>(in),
                                         reinterpret_cast<const chaosuan::bf16_t *>(weight),
                                         reinterpret_cast<const chaosuan::bf16_t *>(bias), m, n, k);
    case CHAOSUAN_DTYPE_F16:
        return linear_<chaosuan::fp16_t>(reinterpret_cast<chaosuan::fp16_t *>(out), reinterpret_cast<const chaosuan::fp16_t *>(in),
                                         reinterpret_cast<const chaosuan::fp16_t *>(weight),
                                         reinterpret_cast<const chaosuan::fp16_t *>(bias), m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);  // 其它类型不支持
    }
}
} // namespace chaosuan::ops::cpu
