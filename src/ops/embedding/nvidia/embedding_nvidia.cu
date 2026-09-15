// ============================================================================
// embedding_nvidia.cu —— embedding 算子的 CUDA 实现（作业 4，Task 4.2）
// ----------------------------------------------------------------------------
// 【embedding 是什么？】词表查表。
//   输入：index —— 一串词编号（int64），比如 [7, 3, 42]；
//   权重：weight —— 词表矩阵 [vocab, dim]，第 i 行就是"第 i 个词的向量"；
//   输出：out —— [numel, dim]，第 i 行 = weight[index[i]]（把编号换成向量）。
//   大模型第一层就是 embedding：token 编号 → 可学习的向量。
//
// 【并行设计】每个线程负责"一行"（一个词向量）。
//   线程 i 做：查 weight 第 index[i] 行，把这 dim 个元素复制到 out 第 i 行。
//   dim 通常不大（128），一行一个线程 + 串行循环 dim 次即可；
//   如果 dim 很大，可以用 2D 网格（一个线程算一个元素），教学版从简。
//
// 【显存访问】weight 按行连续存放，同一线程连续读一行 → 合并访问（coalesced），
// 这是 GPU 内存访问最快的模式（相邻线程访问相邻地址）。
// ============================================================================

#include "embedding_nvidia.cuh"  // 本算子函数声明。

#include <cuda_runtime.h>  // CUDA 运行时。
#include <cuda_fp16.h>     // __half。
#include <cuda_bf16.h>     // __nv_bfloat16。
#include <cstdint>         // int64_t。

#include <stdexcept>       // std::runtime_error。

namespace chaosuan::ops::nvidia {

namespace {

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                 size_t numel, size_t embedding_dim) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;  // 输出第 i 行
    if (i < numel) {
        int64_t idx = index[i];                     // 词编号（已由 op.cpp 校验不越界）
        const T *src = weight + idx * embedding_dim; // 词表第 idx 行的起点
        T *dst = out + i * embedding_dim;            // 输出第 i 行的起点
        for (size_t j = 0; j < embedding_dim; j++) {
            dst[j] = src[j];                         // 逐元素复制（查表）
        }
    }
}

constexpr int kBlockSize = 256;

void launch_embedding(void *out, const void *index, const void *weight,
                      size_t numel, size_t embedding_dim, chaosuanDataType_t dtype) {
    int grid = static_cast<int>((numel + kBlockSize - 1) / kBlockSize);
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        embedding_kernel<float><<<grid, kBlockSize>>>(static_cast<float *>(out),
                                                      static_cast<const int64_t *>(index),
                                                      static_cast<const float *>(weight), numel, embedding_dim);
        break;
    case CHAOSUAN_DTYPE_F16:
        embedding_kernel<__half><<<grid, kBlockSize>>>(static_cast<__half *>(out),
                                                       static_cast<const int64_t *>(index),
                                                       static_cast<const __half *>(weight), numel, embedding_dim);
        break;
    case CHAOSUAN_DTYPE_BF16:
        embedding_kernel<__nv_bfloat16><<<grid, kBlockSize>>>(static_cast<__nv_bfloat16 *>(out),
                                                              static_cast<const int64_t *>(index),
                                                              static_cast<const __nv_bfloat16 *>(weight),
                                                              numel, embedding_dim);
        break;
    default:
        throw std::runtime_error("Embedding nvidia: unsupported dtype");
    }
    cudaDeviceSynchronize();
}

} // anonymous namespace

// 对外入口：与 CPU 版 embedding 参数一致。
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               chaosuanDataType_t dtype, size_t numel, size_t embedding_dim) {
    launch_embedding(out, index, weight, numel, embedding_dim, dtype);
}

} // namespace chaosuan::ops::nvidia
