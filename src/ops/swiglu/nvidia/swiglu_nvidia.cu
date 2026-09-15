// ============================================================================
// swiglu_nvidia.cu —— swiglu 算子的 CUDA 实现（作业 4，Task 4.2）
// ----------------------------------------------------------------------------
// 【SwiGLU 是什么？】
// 大模型 MLP 里用的激活函数（Llama/Qwen 都是）：
//   out = gate * SiLU(up) = gate * (up / (1 + exp(-up)))
// 其中 SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))。
// gate 和 up 是同一个输入矩阵经过两个不同 linear 的结果。
//
// 【为什么用 float 算？】
// 中间计算（exp、除法）统一提升到 float 精度：
//   - fp16/bf16 的动态范围小，直接算 exp 容易溢出；
//   - GPU 的 float 计算单元又快又准。
// 只在最后写回时转回原类型。
// ============================================================================

#include "swiglu_nvidia.cuh"  // 本算子函数声明。

#include <cuda_runtime.h>  // CUDA 运行时。
#include <cuda_fp16.h>     // __half。
#include <cuda_bf16.h>     // __nv_bfloat16。

#include <stdexcept>       // std::runtime_error。

namespace chaosuan::ops::nvidia {

namespace {

// 类型到 float 的转换辅助（__device__ = 跑在 GPU 上的函数）。
// float 直接返回；__half / __nv_bfloat16 用 CUDA 自带转换指令。
template <typename T>
__device__ float to_float(T x);

template <>
__device__ float to_float<float>(float x) { return x; }

template <>
__device__ float to_float<__half>(__half x) { return __half2float(x); }

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

// float 转回原类型。
template <typename T>
__device__ T from_float(float x);

template <>
__device__ float from_float<float>(float x) { return x; }

template <>
__device__ __half from_float<__half>(float x) { return __float2half(x); }

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// 模板 kernel：每个线程算一个输出元素。
template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numel) {
        float g = to_float<T>(gate[i]);     // 门控值
        float u = to_float<T>(up[i]);       // 上游值
        // SiLU(u) = u / (1 + e^(-u))：注意 e^(-u) 用 expf（float 版 exp）。
        float silu = u / (1.0f + expf(-u));
        out[i] = from_float<T>(g * silu);   // gate * SiLU(up)
    }
}

constexpr int kBlockSize = 256;

void launch_swiglu(void *out, const void *gate, const void *up, size_t numel, chaosuanDataType_t dtype) {
    int grid = static_cast<int>((numel + kBlockSize - 1) / kBlockSize);
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        swiglu_kernel<float><<<grid, kBlockSize>>>(static_cast<float *>(out), static_cast<const float *>(gate),
                                                   static_cast<const float *>(up), numel);
        break;
    case CHAOSUAN_DTYPE_F16:
        swiglu_kernel<__half><<<grid, kBlockSize>>>(static_cast<__half *>(out), static_cast<const __half *>(gate),
                                                    static_cast<const __half *>(up), numel);
        break;
    case CHAOSUAN_DTYPE_BF16:
        swiglu_kernel<__nv_bfloat16><<<grid, kBlockSize>>>(static_cast<__nv_bfloat16 *>(out),
                                                           static_cast<const __nv_bfloat16 *>(gate),
                                                           static_cast<const __nv_bfloat16 *>(up), numel);
        break;
    default:
        throw std::runtime_error("SwiGLU nvidia: unsupported dtype");
    }
    cudaDeviceSynchronize();
}

} // anonymous namespace

// 对外入口：与 CPU 版 swiglu 参数一致。
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            chaosuanDataType_t dtype, size_t numel) {
    launch_swiglu(out, gate, up, numel, dtype);
}

} // namespace chaosuan::ops::nvidia
