// ============================================================================
// add_nvidia.cu —— add 算子的 CUDA 实现（作业 4，Task 4.2）
// ----------------------------------------------------------------------------
// 【CUDA 程序的基本结构】（后面每个算子都一样，这里讲透一次）
//   1. __global__ 函数 = kernel：跑在 GPU 上的函数，由很多"线程"同时执行。
//   2. 启动语法：kernel<<<grid, block>>>(参数...)
//        grid  = 网格里有多少个线程块（block）
//        block = 每个块里有多少个线程
//      例如 <<<10, 256>>> 就是 10 个块 × 每块 256 线程 = 2560 个线程并行。
//   3. 线程身份：blockIdx.x（第几个块）、threadDim.x（每块线程数）、
//      threadIdx.x（块内第几个线程）。
//      全局下标 i = blockIdx.x * blockDim.x + threadIdx.x。
//   4. 每个线程只处理自己负责的那一部分数据（本例：每个线程算一个元素）。
//
// 【为什么 kernel 里要判 if (i < numel)？】
// 数据个数 numel 不一定能被 (grid*block) 整除，多出来的线程就"空转"跳过，
// 防止越界访问内存。
//
// 【GPU 加法】c[i] = a[i] + b[i]：所有元素互不依赖，天然适合并行，
// 每个线程算一个，理论上是"最理想的并行"。
// ============================================================================

#include "add_nvidia.cuh"  // 本算子函数声明（nvidia 命名空间）。

#include <cuda_runtime.h>  // CUDA 运行时（kernel 启动、同步）。
#include <cuda_fp16.h>     // __half：fp16 类型及算术。
#include <cuda_bf16.h>     // __nv_bfloat16：bf16 类型及算术。

#include <stdexcept>       // std::runtime_error（不支持的类型报错）。

namespace chaosuan::ops::nvidia {

namespace {

// ---------------------------------------------------------------------------
// 模板 kernel：T 可以是 float / __half / __nv_bfloat16。
//   T 用运算符重载：__half 的 + 会调用 GPU 半精度加法单元（比转 float 快），
//   __nv_bfloat16 的 + 也一样。所以一个模板覆盖三种类型，不用写三份。
// ---------------------------------------------------------------------------
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    // 计算本线程负责的全局下标（见文件头注释第 4 点）。
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numel) {        // 越界保护
        c[i] = a[i] + b[i]; // 并行加法：每个线程独立做一次
    }
}

// 启动参数常量：每块 256 个线程。
// 为什么选 256？经验值——256~512 之间通常吞吐最好（GPU 按 32 个线程
// 一组（warp）调度，256 = 8 个 warp，容易占满计算单元）。
constexpr int kBlockSize = 256;

// 启动 kernel 并同步。
// kernel 是"异步"的：<<<>>> 只是把任务提交给 GPU 就返回了。
// 不 cudaDeviceSynchronize() 的话，调用方立刻读结果可能读到旧数据。
// 这里统一同步（简单正确；性能优化可后续用 stream 重叠）。
void launch_add(void *c, const void *a, const void *b, size_t numel, chaosuanDataType_t dtype) {
    int grid = static_cast<int>((numel + kBlockSize - 1) / kBlockSize);
    // 向上取整的块数：(numel + 255) / 256。多出的线程由 if (i < numel) 兜底。
    switch (dtype) {
    case CHAOSUAN_DTYPE_F32:
        add_kernel<float><<<grid, kBlockSize>>>(static_cast<float *>(c), static_cast<const float *>(a),
                                                static_cast<const float *>(b), numel);
        break;
    case CHAOSUAN_DTYPE_F16:
        // 注意：我们的 fp16_t 内部就是一个 uint16_t，和 CUDA 的 __half
        // 内存布局完全相同（都是 IEEE 半精度 16 位），所以直接 reinterpret。
        add_kernel<__half><<<grid, kBlockSize>>>(static_cast<__half *>(c), static_cast<const __half *>(a),
                                                 static_cast<const __half *>(b), numel);
        break;
    case CHAOSUAN_DTYPE_BF16:
        add_kernel<__nv_bfloat16><<<grid, kBlockSize>>>(static_cast<__nv_bfloat16 *>(c),
                                                        static_cast<const __nv_bfloat16 *>(a),
                                                        static_cast<const __nv_bfloat16 *>(b), numel);
        break;
    default:
        throw std::runtime_error("Add nvidia: unsupported dtype");
    }
    cudaDeviceSynchronize();  // 等 GPU 算完再返回
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// 对外入口：参数与 CPU 版 add 完全一致（op.cpp 里对 NVIDIA 设备就调它）。
// ---------------------------------------------------------------------------
void add(std::byte *c, const std::byte *a, const std::byte *b, chaosuanDataType_t dtype, size_t numel) {
    launch_add(c, a, b, numel, dtype);
}

} // namespace chaosuan::ops::nvidia
