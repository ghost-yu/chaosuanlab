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
    // 【坑 15：cos/sin 用 double 精度计算，与 CPU 版位级一致】
    //   CPU 版用 std::pow / std::cos / std::sin（内部 double 精度，正确舍入）；
    //   GPU 的 powf / cosf / sinf 是 float 版近似，结果可能差 1 ulp。
    //   1 ulp 单独看无所谓，但 rope 输出直接决定 attention 的 q/k 向量，
    //   差异会被后续 linear/softmax 放大，28 层累积后长序列 top-1 翻转。
    //   这里数据量小（seqlen × d/2 个角度），用 double 算完再转 float，
    //   与 CPU 的 std::cos(float)（glibc 正确舍入）位级一致。
    // 【坑 15 续：表达式必须与 CPU 版逐运算一致】
    //   CPU 版：freq[j] = pow(theta, 2.0f * (float)j / (float)d) —— 指数是
    //   **float 运算**；angle = pos / freq[j] —— **float 除法**。
    //   这里必须照抄：
    //   1) 指数 2.0f * j / d 先按 float 算（提升 double 前的值要和 CPU 相同）；
    //   2) pow 用 double 精度（CPU 的 std::pow(float,float) → glibc powf，
    //      是"正确舍入"的 float 结果；double 精度 pow 再转 float 与之位级一致）；
    //   3) angle 用 float 除法（CPU 是 float 除 float）；
    //   4) cos/sin 用 double 精度（CPU 的 std::cos(float) → 正确舍入的 float，
    //      double 精度 cos 再转 float 与之位级一致）。
    float freq = (float)pow((double)theta, (double)(2.0f * (float)j / (float)d));
    float angle = pos / freq;
    float cos_val = (float)cos((double)angle);
    float sin_val = (float)sin((double)angle);

    // 同一个 (s, j) 下所有 head 共享角度 → 循环 head 复用 cos/sin。
    for (size_t h = 0; h < nheads; h++) {
        size_t base = s * nheads * d + h * d;   // 当前 (token, head) 向量起点
        float a = to_float<T>(in[base + j]);            // 前半分量
        float b = to_float<T>(in[base + half_d + j]);   // 后半分量
        // 【坑 15 续：乘加/乘减显式分开舍入，禁止 FMA 融合】
        //   nvcc -fmad=true 会把 a*cos - b*sin 融合成 FMA 指令（一次舍入），
        //   CPU 版是"两次独立乘法 + 一次减法"（三次舍入）。舍入次数不同 →
        //   结果差 1 ulp → 长序列累积翻转。用 __fmul_rn/__fsub_rn/__fadd_rn
        //   强制各自按 IEEE 舍入，与 CPU 行为位级一致。
        float t1 = __fmul_rn(a, cos_val);   // a * cos
        float t2 = __fmul_rn(b, sin_val);   // b * sin
        float t3 = __fmul_rn(b, cos_val);   // b * cos
        float t4 = __fmul_rn(a, sin_val);   // a * sin
        float ra = __fsub_rn(t1, t2);       // a*cos - b*sin
        float rb = __fadd_rn(t3, t4);       // b*cos + a*sin
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
