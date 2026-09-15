// ============================================================================
// linear_nvidia.cu —— linear 算子的 CUDA 实现（作业 4，Task 4.2）
// ----------------------------------------------------------------------------
// 【linear 是什么？】y = x·W^T + b：
//   in[m,k] @ weight[n,k]^T + bias[n] = out[m,n]。
// 大模型里每个线性层都在做这件事，是算力消耗最大的算子之一，
// 所以必须用 NVIDIA 的 cuBLAS 库（GPU 上最快的矩阵乘实现）。
//
// 【cuBLAS 的坑：列主序 vs 行主序】
// cuBLAS 假设矩阵按**列主序**（column-major）存储：内存里先排第 0 列。
// 而我们（和 PyTorch、C++ 数组）都是**行主序**。
// 标准解法（PyTorch linear 底层也是这么做的）：
//   行主序 Y[m,n] = X[m,k] @ W[n,k]^T
//   → cublasSgemm(N, N, n, m, k, α, W, n, X, k, β, Y, n)
//   即把 Y 看成列主序 [n,m]，W 看成列主序 [n,k]，X 看成列主序 [k,m]，
//   cuBLAS 算的 C = W @ X^T 恰好就是 Y^T。内存字节一模一样。
//
// 【类型策略】
//   F32  → cublasSgemm（教学重点：怎么调 cuBLAS）
//   F16/BF16 → 朴素 GEMM kernel（正确性优先，不做低精度优化，
//   留作后续优化题：换成 cublasGemmEx + Tensor Core）。
// ============================================================================

#include "linear_nvidia.cuh"  // 本算子函数声明。

#include <cuda_runtime.h>  // CUDA 运行时。
#include <cuda_fp16.h>     // __half。
#include <cuda_bf16.h>     // __nv_bfloat16。
#include <cublas_v2.h>     // cuBLAS（矩阵乘）。

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

// ---------------------------------------------------------------------------
// 朴素 GEMM kernel（F16/BF16 用）：每个线程算输出矩阵的一个元素。
//   out[i][j] = Σ_kk in[i][kk] * weight[j][kk] + bias[j]
// 性能远不如 cuBLAS，但正确性有保证，作为低精度类型的兜底。
// ---------------------------------------------------------------------------
template <typename T>
__global__ void linear_naive_kernel(T *out, const T *in, const T *weight,
                                    const T *bias, size_t m, size_t n, size_t k) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    size_t i = idx / n;   // 输出行
    size_t j = idx % n;   // 输出列

    float sum = 0.0f;
    for (size_t kk = 0; kk < k; kk++) {
        sum += to_float<T>(in[i * k + kk]) * to_float<T>(weight[j * k + kk]);
    }
    if (bias) sum += to_float<T>(bias[j]);
    out[idx] = from_float<T>(sum);
}

// 给 cuBLAS 结果加偏置（每个输出元素加对应的 bias[j]）。
template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t m, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    size_t j = idx % n;   // 第 j 列（输出神经元）
    out[idx] = from_float<T>(to_float<T>(out[idx]) + to_float<T>(bias[j]));
}

// cuBLAS handle：全局单例，第一次用时创建。
// 进程退出时由系统回收；重复创建销毁开销大，所以只建一次。
cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }
        return h;
    }();
    return handle;
}

constexpr int kBlockSize = 256;

// F32：走 cuBLAS（业界标准做法）。
void linear_f32_cublas(float *out, const float *in, const float *weight,
                       const float *bias, size_t m, size_t n, size_t k) {
    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f, beta = 0.0f;
    // 见文件头注释：行主序 Y[m,n] = X[m,k] @ W[n,k]^T。
    // 关键认知：行主序 [R,C] 的字节流 ≡ 列主序 [C,R] 的字节流（同一个内存换个读法）。
    //   W 行主序 [n,k] → 列主序 [k,n]（行数 k，lda=k），要 W^T 的效果 → OP_T；
    //   X 行主序 [m,k] → 列主序 [k,m]（行数 k，ldb=k），不转置 → OP_N；
    //   Y 行主序 [m,n] → 列主序 [n,m]（行数 n，ldc=n），作为 C 输出。
    // 数学验证：C[i'][j'] = Σ W[i'][kk]*X[j'][kk] = Y[j'][i']，内存字节完全一致。
    cublasStatus_t st = cublasSgemm(handle,
                                    CUBLAS_OP_T,        // op(W) = W^T（关键！）
                                    CUBLAS_OP_N,        // op(X) = X
                                    static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                                    &alpha,
                                    weight, static_cast<int>(k),   // W 列主序 [k,n]，lda=k
                                    in, static_cast<int>(k),       // X 列主序 [k,m]，ldb=k
                                    &beta,
                                    out, static_cast<int>(n));     // Y 列主序 [n,m]，ldc=n
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSgemm failed");
    }
    if (bias) {
        add_bias_kernel<float><<<static_cast<int>((m * n + kBlockSize - 1) / kBlockSize), kBlockSize>>>(
            out, bias, m, n);
    }
    cudaDeviceSynchronize();
}

template <typename T>
void launch_linear(void *out, const void *in, const void *weight, const void *bias,
                   size_t m, size_t n, size_t k) {
    size_t total = m * n;
    int grid = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    linear_naive_kernel<T><<<grid, kBlockSize>>>(static_cast<T *>(out),
                                                 static_cast<const T *>(in),
                                                 static_cast<const T *>(weight),
                                                 static_cast<const T *>(bias), m, n, k);
    cudaDeviceSynchronize();
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// 对外入口：与 CPU 版 linear 参数一致。
// ---------------------------------------------------------------------------
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            chaosuanDataType_t dtype, size_t m, size_t n, size_t k) {
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        linear_f32_cublas(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                          reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias),
                          m, n, k);
        break;
    case CHAOSUAN_DTYPE_F16:
        launch_linear<__half>(out, in, weight, bias, m, n, k);
        break;
    case CHAOSUAN_DTYPE_BF16:
        launch_linear<__nv_bfloat16>(out, in, weight, bias, m, n, k);
        break;
    default:
        throw std::runtime_error("Linear nvidia: unsupported dtype");
    }
}

} // namespace chaosuan::ops::nvidia
