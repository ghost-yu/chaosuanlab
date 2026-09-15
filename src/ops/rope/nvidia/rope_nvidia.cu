// ============================================================================
// rope_nvidia.cu —— rope 算子的 CUDA 实现（作业 4，Task 4.2）
// ----------------------------------------------------------------------------
// 【RoPE 是什么？】见 CPU 版注释：把"位置信息"编码进向量。
// 对位置 pos 的每个分量对 (a, b) = (in[j], in[d/2+j]) 做二维旋转：
//   a' = a*cos(angle) - b*sin(angle)
//   b' = b*cos(angle) + a*sin(angle)
//   angle = pos / theta^(2j/d)（j = 0..d/2-1）
//
// 【并行设计】
//   每个线程负责一个 (token s, 分量对 j)：线程数 = seqlen * d/2。
//   内层再循环 nheads 个 head（它们共享同一个 cos/sin，算一次用多次，
//   和 CPU 版的结构一致）。
//   freq = theta^(2j/d) 每次现算（powf 开销小，d=128 时 j 最多 63）；
//   更快的做法是预计算查表，教学版从简。
// ============================================================================

#include "rope_nvidia.cuh"  // 本算子函数声明。

#include <cuda_runtime.h>  // CUDA 运行时。
#include <cuda_fp16.h>     // __half。
#include <cuda_bf16.h>     // __nv_bfloat16。
#include <cstdint>         // int64_t。

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

// 每个线程负责一个 (s, j) 对；内存排布 [seqlen][nheads][d]。
template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                            float theta, size_t seqlen, size_t nheads, size_t d) {
    const size_t half_d = d / 2;
    // 全局线程 id → (s, j)：
    // 线程 id 从 0 到 seqlen*half_d-1，按行优先拆成 s = id / half_d, j = id % half_d。
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= seqlen * half_d) return;

    size_t s = tid / half_d;        // 第几个 token
    size_t j = tid % half_d;        // 第几对分量

    float pos = static_cast<float>(pos_ids[s]);          // 本 token 的位置编号
    // freq[j] = theta^(2j/d)；angle = pos / freq = pos * theta^(-2j/d)
    // 用 powf 直接算：powf(theta, 2.0f*j/d) 就是 theta^(2j/d)。
    float angle = pos / powf(theta, 2.0f * static_cast<float>(j) / static_cast<float>(d));
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // 同一个 (s, j) 下所有 head 共享角度 → 循环 head 复用 cos/sin。
    for (size_t h = 0; h < nheads; h++) {
        size_t base = s * nheads * d + h * d;   // 当前 (token, head) 向量起点
        float a = to_float<T>(in[base + j]);            // 前半分量
        float b = to_float<T>(in[base + half_d + j]);   // 后半分量
        float ra = a * cos_val - b * sin_val;           // 旋转公式
        float rb = b * cos_val + a * sin_val;
        out[base + j] = from_float<T>(ra);
        out[base + half_d + j] = from_float<T>(rb);
    }
}

constexpr int kBlockSize = 256;

template <typename T>
void launch_rope(void *out, const void *in, const void *pos_ids, float theta,
                 size_t seqlen, size_t nheads, size_t d) {
    size_t total = seqlen * (d / 2);
    int grid = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    rope_kernel<T><<<grid, kBlockSize>>>(
        static_cast<T *>(out), static_cast<const T *>(in),
        static_cast<const int64_t *>(pos_ids), theta, seqlen, nheads, d);
    cudaDeviceSynchronize();
}

} // anonymous namespace

// 对外入口：与 CPU 版 rope 参数一致。
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, chaosuanDataType_t dtype, size_t seqlen, size_t nheads, size_t d) {
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        launch_rope<float>(out, in, pos_ids, theta, seqlen, nheads, d);
        break;
    case CHAOSUAN_DTYPE_F16:
        launch_rope<__half>(out, in, pos_ids, theta, seqlen, nheads, d);
        break;
    case CHAOSUAN_DTYPE_BF16:
        launch_rope<__nv_bfloat16>(out, in, pos_ids, theta, seqlen, nheads, d);
        break;
    default:
        throw std::runtime_error("Rope nvidia: unsupported dtype");
    }
}

} // namespace chaosuan::ops::nvidia
