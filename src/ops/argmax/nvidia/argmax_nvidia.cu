// ============================================================================
// argmax_nvidia.cu —— argmax 算子的 CUDA 实现（作业 4，Task 4.2）
// ----------------------------------------------------------------------------
// 【argmax 是什么？】见 CPU 版注释：找一串数里最大值的下标。
//   例：vals = [3, 7, 2, 9, 1] → max_idx = 3, max_val = 9。
//
// 【CPU vs GPU 的做法】
//   CPU：一个线程从头扫到尾，一趟出结果（串行）。
//   GPU：成千上万个线程**并行做归约**（reduction）：
//     第一步：把数据切成若干段，每段由一个 block 并行扫，
//            段内用"树形归约"快速找出本段最大（值, 下标）；
//     第二步：把所有段的结果（很少，就 block 个数）再归约一次，得全局最大。
//
// 【树形归约是什么？】
//   假设 256 个线程，每人先看 8 个元素，得到 256 个局部最大；
//   然后 256 → 128 → 64 → ... → 1，每一轮相邻两个合并取较大，
//   共 log2(256) = 8 轮。共享内存里做，比串行快一个数量级。
//
// 【和 CPU 版保持一致的语义】严格大于（>）才更新，相等不更新
//   → 多个最大值并列时取**下标最小**的那个。
// ============================================================================

#include "argmax_nvidia.cuh"  // 本算子函数声明。

#include <cuda_runtime.h>  // CUDA 运行时。
#include <cuda_fp16.h>     // __half。
#include <cuda_bf16.h>     // __nv_bfloat16。
#include <cstdint>         // int64_t。

#include <stdexcept>       // std::runtime_error。

namespace chaosuan::ops::nvidia {

namespace {

// 类型 → float（GPU 端转换辅助，见 swiglu_nvidia.cu 的说明）。
template <typename T>
__device__ float to_float(T x);

template <>
__device__ float to_float<float>(float x) { return x; }

template <>
__device__ float to_float<__half>(__half x) { return __half2float(x); }

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

// 一个 block 处理一段数据，输出本段的最大（值, 下标）到 partial 数组。
// block 内：先每人看若干元素得到局部最大，再用共享内存树形归约。
template <typename T, int kThreads>
__global__ void argmax_block_kernel(const T *vals, size_t numel,
                                    float *partial_val, int64_t *partial_idx) {
    // 共享内存：每个线程放一个候选（值, 下标）。kThreads 编译期常量 → 数组大小编译期已知。
    __shared__ float s_val[kThreads];
    __shared__ int64_t s_idx[kThreads];

    const size_t bid = blockIdx.x;              // 第几个 block
    const size_t seg = (numel + gridDim.x - 1) / gridDim.x;  // 每段长度（向上取整）

    // ---- 第 1 步：本线程负责段内 stride 步长的那些元素，找出局部最大 ----
    float best_f = -INFINITY;                   // 当前看到的最大值（float 比较）
    int64_t best_i = -1;                        // 对应的下标（-1 = 还没看到合法元素）

    for (size_t i = bid * seg + threadIdx.x; i < numel && i < (bid + 1) * seg;
         i += kThreads) {
        float v = to_float<T>(vals[i]);
        // 严格大于才更新 → 并列取最小下标（与 CPU 版一致）。
        if (v > best_f) {
            best_f = v;
            best_i = static_cast<int64_t>(i);
        }
    }

    // 每个线程把自己段的局部最大写进共享内存。
    s_val[threadIdx.x] = best_f;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();  // 同步：等所有线程都写完共享内存再开始归约

    // ---- 第 2 步：树形归约（每轮合并相邻两个，线程数减半） ----
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            // 只让前一半线程干活：比较自己和"另一半"的候选，取较大者。
            // 【关键】共享内存槽位 ≠ 真实元素下标！两个候选值相等时，
            // "保留 low 侧"并不能保证下标最小（low 侧可能是后遇到的下标）。
            // 所以比较规则必须是：值更大，或值相等但下标更小，才替换。
            // 这样才能与 CPU 版"严格大于、并列取最小下标"的语义一致。
            if (s_val[threadIdx.x + stride] > s_val[threadIdx.x] ||
                (s_val[threadIdx.x + stride] == s_val[threadIdx.x] &&
                 s_idx[threadIdx.x + stride] < s_idx[threadIdx.x])) {
                s_val[threadIdx.x] = s_val[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // 线程 0 把本 block 的最终候选写进 partial 数组。
    if (threadIdx.x == 0) {
        partial_val[bid] = s_val[0];
        partial_idx[bid] = s_idx[0];
    }
}

// 第二个 kernel：在 partial 数组（gridDim.x 个元素）上做一次简单规约。
// 一个 block 就够（partial 很小，比如 128 个 block 就 128 个候选）。
template <typename T, int kThreads>
__global__ void argmax_final_kernel(const float *partial_val, const int64_t *partial_idx,
                                    int64_t *max_idx, T *max_val, size_t n_partial) {
    __shared__ float s_val[kThreads];
    __shared__ int64_t s_idx[kThreads];

    // 每个线程先扫 partial 的 stride 步长部分。
    float best_f = -INFINITY;
    int64_t best_i = -1;
    for (size_t i = threadIdx.x; i < n_partial; i += kThreads) {
        float v = partial_val[i];
        if (v > best_f) {
            best_f = v;
            best_i = partial_idx[i];
        }
    }
    s_val[threadIdx.x] = best_f;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            // 与 argmax_block_kernel 相同的规则：值更大，或值相等但下标更小才替换。
            if (s_val[threadIdx.x + stride] > s_val[threadIdx.x] ||
                (s_val[threadIdx.x + stride] == s_val[threadIdx.x] &&
                 s_idx[threadIdx.x + stride] < s_idx[threadIdx.x])) {
                s_val[threadIdx.x] = s_val[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *max_idx = s_idx[0];
        *max_val = static_cast<T>(s_val[0]);   // float 转回原类型写回
    }
}

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 128;   // 归约段数上限（partial 数组大小）

template <typename T>
void launch_argmax(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    // 段数 = min(元素数, 128)。元素太少就只开需要的 block。
    size_t nblocks = (numel + kBlockSize - 1) / kBlockSize;
    if (nblocks > kMaxBlocks) nblocks = kMaxBlocks;

    // 临时显存：partial 结果（在调用侧分配，见 host 入口）。
    // 注意：kernel 不能随便分配显存（cudaMalloc 很慢），所以 host 先 malloc。
    // 这里用"每个 block 都处理"的 seg 方式，见 kernel 内部计算。
    static float *partial_val = nullptr;
    static int64_t *partial_idx = nullptr;
    // 简单起见用静态指针 + 首次分配（教学版；正式版应在 host 分配传入）。
    if (partial_val == nullptr) {
        cudaMalloc(&partial_val, kMaxBlocks * sizeof(float));
        cudaMalloc(&partial_idx, kMaxBlocks * sizeof(int64_t));
    }

    argmax_block_kernel<T, kBlockSize><<<nblocks, kBlockSize>>>(
        vals, numel, partial_val, partial_idx);
    argmax_final_kernel<T, kBlockSize><<<1, kBlockSize>>>(
        partial_val, partial_idx, max_idx, max_val, nblocks);
    cudaDeviceSynchronize();
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// 对外入口：与 CPU 版 argmax 参数一致。
// 注意 max_idx/max_val 是框架给张量分配的**显存指针**，kernel 直接写显存，
// 调用方（如模型代码）需要 D2H 拷回才能读（这是作业 4 模型侧的要点之一）。
// ---------------------------------------------------------------------------
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            chaosuanDataType_t dtype, size_t numel) {
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        launch_argmax<float>(reinterpret_cast<int64_t *>(max_idx),
                             reinterpret_cast<float *>(max_val),
                             reinterpret_cast<const float *>(vals), numel);
        break;
    case CHAOSUAN_DTYPE_F16:
        launch_argmax<__half>(reinterpret_cast<int64_t *>(max_idx),
                              reinterpret_cast<__half *>(max_val),
                              reinterpret_cast<const __half *>(vals), numel);
        break;
    case CHAOSUAN_DTYPE_BF16:
        launch_argmax<__nv_bfloat16>(reinterpret_cast<int64_t *>(max_idx),
                                     reinterpret_cast<__nv_bfloat16 *>(max_val),
                                     reinterpret_cast<const __nv_bfloat16 *>(vals), numel);
        break;
    default:
        throw std::runtime_error("Argmax nvidia: unsupported dtype");
    }
}

} // namespace chaosuan::ops::nvidia
