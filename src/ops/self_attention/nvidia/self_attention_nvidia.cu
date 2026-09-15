// ============================================================================
// self_attention_nvidia.cu —— self_attention 算子的 CUDA 实现（作业 4，Task 4.2）
// ----------------------------------------------------------------------------
// 【self_attention 是什么？】见 CPU 版注释：Transformer 的核心。
// 三步（对每个查询头 h、每个位置 i）：
//   1. scores[j] = (q[i]·k[j]) * scale，j > i+offset 置 -INF（因果掩码）；
//   2. 数值稳定 softmax：exp(scores[j] - max) / Σexp(...)；
//   3. attn_val[i] = Σ_j softmax_j * v[j]。
// GQA：第 h 个查询头用第 h/nrep 个键值头。
//
// 【并行设计】每个线程负责一个 (i, h) 对（= 一个输出向量）：
//   - 线程数 = seqlen × nhead（Qwen2: 128×12，轻松塞满 GPU）；
//   - 一个线程内完整做三步，不需要和其他线程通信；
//   - scores 数组放哪？每个线程需要 total_len 个 float 存分数。
//     放"私有局部内存"（local memory）正确但慢；教学版直接用动态共享内存：
//     每个线程一行 scores，因为每行只有自己读写，__syncthreads 都不需要。
//
// 【共享内存大小】blockDim.x × total_len × 4 字节。
//   total_len 是动态的（KV 缓存增长），用 extern __shared__ + 动态大小，
//   launch 时第三个参数传字节数。
// ============================================================================

#include "self_attention_nvidia.cuh"  // 本算子函数声明。

#include <cuda_runtime.h>  // CUDA 运行时。
#include <cuda_fp16.h>     // __half。
#include <cuda_bf16.h>     // __nv_bfloat16。

#include <stdexcept>       // std::runtime_error。
#include <cfloat>          // INFINITY（float 正无穷）。

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

// 每个线程负责一个 (i, h)；共享内存里给每个线程一行 scores。
// 动态共享内存声明：extern __shared__ 数组，大小在 launch 时指定。
template <typename T>
__global__ void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                      float scale, size_t seqlen, size_t total_len,
                                      size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    extern __shared__ float s_scores[];   // [blockDim.x][total_len]

    // 本线程负责的 (i, h)
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= seqlen * nhead) return;
    size_t i = tid / nhead;               // 查询位置
    size_t h = tid % nhead;               // 查询头

    size_t nrep = nhead / nkvhead;        // GQA：几个查询头共享一个键值头
    size_t kh = h / nrep;                 // 本头对应的键值头

    float *my_scores = s_scores + threadIdx.x * total_len;  // 本线程的分数行

    // -----------------------------------------------------------------------
    // 第 1 步：算注意力分数（含因果掩码）
    // -----------------------------------------------------------------------
    // 因果掩码：第 i 个查询位置只能看到 j <= i + offset 的键。
    // offset = total_len - seqlen（KV 缓存里历史 token 数）。
    size_t offset = total_len - seqlen;
    float max_score = -INFINITY;
    for (size_t j = 0; j < total_len; j++) {
        if (j > i + offset) {              // 未来位置：掩码
            my_scores[j] = -INFINITY;
            continue;
        }
        // 点积：q 的第 (i,h) 行 × k 的第 (j,kh) 行
        // 【坑 15 续：点积累加显式分开舍入，禁止 FMA 融合】
        //   与 linear 同理：nvcc 会把 dot += q*k 融合成 FMA（一次舍入），
        //   CPU 版是"乘一次、加一次"（两次舍入）。__fmul_rn/__fadd_rn
        //   强制与 CPU 位级一致。
        float dot = 0.0f;
        for (size_t dim = 0; dim < d; dim++) {
            dot = __fadd_rn(dot, __fmul_rn(to_float<T>(q[i * nhead * d + h * d + dim]),
                                           to_float<T>(k[j * nkvhead * d + kh * d + dim])));
        }
        dot = __fmul_rn(dot, scale);   // 缩放也按"单独一次乘法"舍入
        my_scores[j] = dot;
        if (dot > max_score) max_score = dot;
    }

    // -----------------------------------------------------------------------
    // 第 2 步：数值稳定 softmax（原地把 scores 换成 exp 值，最后归一到 1）
    // -----------------------------------------------------------------------
    float sum_exp = 0.0f;
    for (size_t j = 0; j < total_len; j++) {
        // 【坑 15：exp 用 double 精度，与 CPU 版位级一致】
        //   CPU 版 std::exp(float) 内部 double 精度正确舍入；GPU expf 是
        //   float 近似，可能差 1 ulp。softmax 权重差 1 ulp 会在长序列累积，
        //   这里每线程 total_len 次 exp，数据量小，用 double 算再转 float。
        float e = (my_scores[j] == -INFINITY) ? 0.0f : (float)exp((double)(my_scores[j] - max_score));
        my_scores[j] = e;
        sum_exp += e;
    }

    // -----------------------------------------------------------------------
    // 第 3 步：softmax 权重对 v 加权求和 → 写输出
    // -----------------------------------------------------------------------
    for (size_t dim = 0; dim < dv; dim++) {
        float val = 0.0f;
        for (size_t j = 0; j < total_len; j++) {
            if (my_scores[j] > 0.0f) {     // 掩码位置权重为 0，跳过
                // 与点积同理：显式分开舍入，禁止 FMA 融合。
                val = __fadd_rn(val, __fmul_rn(my_scores[j], to_float<T>(v[j * nkvhead * dv + kh * dv + dim])));
            }
        }
        attn_val[i * nhead * dv + h * dv + dim] = from_float<T>(val / sum_exp);
    }
}

// 【坑 16：kBlockSize 必须足够小，防止动态共享内存超限】
//   动态共享内存 = kBlockSize × total_len × 4 字节。
//   默认每 block 共享内存上限 48KB（49152 B）。
//   旧值 256：total_len 达到 49 时 256×49×4 = 50176 B > 49152 B，
//   CUDA 核**启动失败**（launch failure）。而 launch 后只调了
//   cudaDeviceSynchronize() 没查错误 → 失败被**静默吞掉**，
//   输出张量是垃圾值 → 模型从第 50 个 token 开始乱码。
//   修复：kBlockSize=64 → 64×137×4 = 35KB < 48KB（128 步生成，
//   total_len ≤ 137），且 launch 后必须查 cudaGetLastError。
//   若未来 total_len 更大，可进一步缩小 kBlockSize 或调用
//   cudaFuncSetAttribute 提高动态共享内存上限（需要显式 opt-in）。
constexpr int kBlockSize = 64;

template <typename T>
void launch_self_attention(void *attn_val, const void *q, const void *k, const void *v,
                           float scale, size_t seqlen, size_t total_len, size_t nhead,
                           size_t nkvhead, size_t d, size_t dv) {
    size_t total_threads = seqlen * nhead;
    int grid = static_cast<int>((total_threads + kBlockSize - 1) / kBlockSize);
    // 动态共享内存：每线程一行 scores（total_len 个 float）。
    size_t smem_bytes = kBlockSize * total_len * sizeof(float);
    self_attention_kernel<T><<<grid, kBlockSize, smem_bytes>>>(
        static_cast<T *>(attn_val), static_cast<const T *>(q), static_cast<const T *>(k),
        static_cast<const T *>(v), scale, seqlen, total_len, nhead, nkvhead, d, dv);
    // 【坑 16 续：必须检查核启动错误！】
    //   cudaGetLastError 返回最近一次异步启动的错误（如共享内存超限），
    //   不检查的话错误会被后续调用覆盖，留下"看起来正常但结果全错"的坑。
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("self_attention kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();
}

} // anonymous namespace

// 对外入口：与 CPU 版 self_attention 参数一致。
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, chaosuanDataType_t dtype,
                    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        launch_self_attention<float>(attn_val, q, k, v, scale, seqlen, total_len, nhead, nkvhead, d, dv);
        break;
    case CHAOSUAN_DTYPE_F16:
        launch_self_attention<__half>(attn_val, q, k, v, scale, seqlen, total_len, nhead, nkvhead, d, dv);
        break;
    case CHAOSUAN_DTYPE_BF16:
        launch_self_attention<__nv_bfloat16>(attn_val, q, k, v, scale, seqlen, total_len, nhead, nkvhead, d, dv);
        break;
    default:
        throw std::runtime_error("SelfAttention nvidia: unsupported dtype");
    }
}

} // namespace chaosuan::ops::nvidia
