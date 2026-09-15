// ============================================================================
// rms_norm_nvidia.cu —— rms_norm 算子的 CUDA 实现（作业 4，Task 4.2）
// ----------------------------------------------------------------------------
// 【RMSNorm 是什么？】见 CPU 版注释：对每一行做"均方根归一化"：
//   rms = sqrt(平方和 / d + eps)；out[j] = in[j] * weight[j] / rms。
// 需要扫两遍：第一遍求平方和，第二遍归一化。
//
// 【并行设计】一行一个 block（gridDim = m 行）。
//   1. 块内每个线程累加自己负责的那些列的平方 → 共享内存；
//   2. 树形归约出整行平方和 → 线程 0 算 rms 存共享内存；
//   3. __syncthreads() 等 rms 就绪 → 每个线程归一化自己负责的列。
//
// 【为什么用共享内存 + __syncthreads？】
//   共享内存（shared memory）是"块内共享、极快"的内存，比全局内存快几十倍。
//   __syncthreads() 是"块内路障"：等所有线程都执行到这里才放行，
//   防止"线程 A 还没写完共享内存，线程 B 就开始读"的竞态。
//   这是 CUDA 并行最核心的两个概念，必须吃透。
// ============================================================================

#include "rms_norm_nvidia.cuh"  // 本算子函数声明。

#include <cuda_runtime.h>  // CUDA 运行时。
#include <cuda_fp16.h>     // __half。
#include <cuda_bf16.h>     // __nv_bfloat16.

#include <stdexcept>       // std::runtime_error。

namespace chaosuan::ops::nvidia {

namespace {

template <typename T>
__device__ float to_float(T x);

template <>
__device__ float to_float<float>(float x) { return x; }

template <>
__device__ float to_float<__half>(__half x) { return __half2float(x); }

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ T from_float(float x);

template <>
__device__ float from_float<float>(float x) { return x; }

template <>
__device__ __half from_float<__half>(float x) { return __float2half(x); }

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// 每行一个 block；kThreads = 每块线程数（编译期常量，数组大小确定）。
template <typename T, int kThreads>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight,
                                float eps, size_t m, size_t d) {
    const size_t row = blockIdx.x;          // 本 block 负责的行
    if (row >= m) return;
    const T *in_row = in + row * d;         // 输入行起点
    T *out_row = out + row * d;             // 输出行起点

    __shared__ float s_sum[kThreads];       // 每个线程的局部平方和

    // ---- 第 1 遍：累加平方和（stride 步长访问，线程间负载均衡） ----
    float sum = 0.0f;
    for (size_t j = threadIdx.x; j < d; j += kThreads) {
        float v = to_float<T>(in_row[j]);
        sum += v * v;
    }
    s_sum[threadIdx.x] = sum;
    __syncthreads();

    // 树形归约：块内求和（和 argmax 的归约同套路，这里是"和"不是"最大"）。
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // 线程 0 算 rms，写进共享内存广播给全块。
    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = sqrtf(s_sum[0] / static_cast<float>(d) + eps);
        // sqrtf 是 float 版开方（GPU 上 sqrt 双精度慢，float 场景用 sqrtf）
    }
    __syncthreads();

    // ---- 第 2 遍：归一化 + 乘权重 ----
    float rms = s_rms;
    for (size_t j = threadIdx.x; j < d; j += kThreads) {
        float w = to_float<T>(weight[j]);   // 权重对每行是共享的（shape [d]）
        out_row[j] = from_float<T>(to_float<T>(in_row[j]) * w / rms);
    }
}

constexpr int kBlockSize = 256;

template <typename T>
void launch_rms_norm(void *out, const void *in, const void *weight, float eps,
                     size_t m, size_t d) {
    // grid = m 行；每行一个 block（每行 d 个元素，d=128 < 256，够用）。
    rms_norm_kernel<T, kBlockSize><<<static_cast<unsigned int>(m), kBlockSize>>>(
        static_cast<T *>(out), static_cast<const T *>(in),
        static_cast<const T *>(weight), eps, m, d);
    cudaDeviceSynchronize();
}

} // anonymous namespace

// 对外入口：与 CPU 版 rms_norm 参数一致。
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, chaosuanDataType_t dtype, size_t m, size_t d) {
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        launch_rms_norm<float>(out, in, weight, eps, m, d);
        break;
    case CHAOSUAN_DTYPE_F16:
        launch_rms_norm<__half>(out, in, weight, eps, m, d);
        break;
    case CHAOSUAN_DTYPE_BF16:
        launch_rms_norm<__nv_bfloat16>(out, in, weight, eps, m, d);
        break;
    default:
        throw std::runtime_error("RmsNorm nvidia: unsupported dtype");
    }
}

} // namespace chaosuan::ops::nvidia
