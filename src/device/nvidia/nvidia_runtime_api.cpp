// ============================================================================
// nvidia_runtime_api.cpp —— CUDA 版 Runtime API（作业 4，Task 4.1）
// ----------------------------------------------------------------------------
// 【这一层是干什么的？】
// CHAOSUAN 的框架用一张"函数表"（ChaosuanRuntimeAPI，里面 12 个函数指针）
// 统一抽象所有设备。CPU 有一个实现（cpu_runtime_api.cpp），
// GPU（NVIDIA）也需要一个实现。本文件就是 CUDA 版：
//   每个函数把"框架的抽象操作"翻译成"CUDA 库的真实调用"。
//
// 对照表（框架函数 → CUDA 函数）：
//   getDeviceCount()        → cudaGetDeviceCount()   查询有几张显卡
//   setDevice(id)           → cudaSetDevice(id)      切换到第 id 张显卡
//   deviceSynchronize()     → cudaDeviceSynchronize() 等待显卡所有任务完成
//   createStream()          → cudaStreamCreate()     创建"流"（任务队列）
//   destroyStream(s)        → cudaStreamDestroy(s)
//   streamSynchronize(s)    → cudaStreamSynchronize(s) 等某个流完成
//   mallocDevice(size)      → cudaMalloc(&p, size)   在显存上分配内存
//   freeDevice(p)           → cudaFree(p)
//   mallocHost(size)        → cudaMallocHost(&p,size) 分配"页锁定"主机内存
//   freeHost(p)             → cudaFreeHost(p)
//   memcpySync(dst,src,sz,k)→ cudaMemcpy(dst,src,sz,k) 同步内存拷贝
//   memcpyAsync(...)        → cudaMemcpyAsync(...)     异步拷贝（放到流上）
//
// 【为什么 mallocHost 用 cudaMallocHost 而不是普通 malloc？】
// cudaMallocHost 分配的是"页锁定内存"（pinned memory）：
// 普通内存（pageable）在 GPU 拷贝时，驱动要先偷偷搬一次到固定区域，
// 慢；页锁定内存省掉这步，Host→Device 拷贝快很多。代价是分配慢一点。
// 张量里的"CPU 张量"就是用这个分配的。
//
// 【CUDA 错误检查】
// 几乎所有 CUDA 函数都返回错误码 cudaError_t，成功才是 cudaSuccess(0)。
// 我们写一个 CUDA_CHECK 宏：出错就打印错误信息并抛 C++ 异常，
// 这样出错时能立刻看到"哪个调用、错在哪"，而不是悄悄失败。
// ============================================================================

#include "../runtime_api.hpp"  // 框架的 Runtime API 声明（函数表结构 + 各设备入口）。

#include <cuda_runtime.h>      // CUDA 运行时头文件（cudaMalloc / cudaMemcpy 等）。

#include <iostream>            // std::cerr（打印错误信息）。
#include <stdexcept>           // std::runtime_error（抛异常）。

namespace {

// ---------------------------------------------------------------------------
// CUDA_CHECK：检查 CUDA 调用是否成功。
//   用法：CUDA_CHECK(cudaMalloc(&p, size));
//   宏展开成一个 do { ... } while(0) 块：
//     - 执行 CUDA 调用并拿到错误码 err；
//     - 不是 cudaSuccess 就打印错误字符串 + 代码位置，然后抛异常；
//     - 是成功就什么都不做。
// 把 return 语句包进宏（比如 return cudaMalloc(...)）不行，
// 所以统一写成"先调用、后检查"两行。
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err = (call);                                                     \
        if (err != cudaSuccess) {                                                     \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err)                   \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;          \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        }                                                                             \
    } while (0)

// 把框架的 memcpy 方向枚举翻译成 CUDA 的 cudaMemcpyKind。
cudaMemcpyKind to_cuda_kind(chaosuanMemcpyKind_t kind) {
    switch (kind) {
    case CHAOSUAN_MEMCPY_H2H: return cudaMemcpyHostToHost;     // 主机 → 主机
    case CHAOSUAN_MEMCPY_H2D: return cudaMemcpyHostToDevice;   // 主机 → 设备（显卡）
    case CHAOSUAN_MEMCPY_D2H: return cudaMemcpyDeviceToHost;   // 设备 → 主机
    case CHAOSUAN_MEMCPY_D2D: return cudaMemcpyDeviceToDevice; // 设备 → 设备
    default:
        throw std::invalid_argument("Unknown memcpy kind");
    }
}

} // anonymous namespace

namespace chaosuan::device::nvidia {

// 用命名空间把实现包一层，和 CPU 版的结构保持一致（cpu::runtime_api）。
namespace runtime_api {

// 查询有几张可用的 NVIDIA 显卡。cudaGetDeviceCount 把数量写到 &count。
// 注意：这个函数在框架启动时就被调用（Context 构造函数），
// 返回 0 就代表"没有 GPU"（比如编译了但机器上没装驱动）。
int getDeviceCount() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

// 切换到第 id 张显卡。之后所有 CUDA 调用（分配、kernel、拷贝）
// 默认都作用在这张卡上。
void setDevice(int id) {
    CUDA_CHECK(cudaSetDevice(id));
}

// 同步：等显卡上所有已经提交的任务都执行完。
// CPU 上什么都不用做（CPU 是同步的），GPU 上必须调，否则可能读到
// 还没算完的数据。
void deviceSynchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 创建"流"（stream）：一个任务队列。可以多个流并行跑不同任务，
// 本项目推理暂时只用默认流（nullptr），但框架 API 要求支持创建。
chaosuanStream_t createStream() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return reinterpret_cast<chaosuanStream_t>(stream);
}

// 销毁一个流。
void destroyStream(chaosuanStream_t stream) {
    CUDA_CHECK(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}

// 等某个流上的任务执行完。
void streamSynchronize(chaosuanStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

// 在显存（device memory）上分配 size 字节，返回显存指针。
void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

// 释放显存。
void freeDevice(void *ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

// 在"页锁定主机内存"上分配 size 字节（见文件头注释）。
void *mallocHost(size_t size) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

// 释放页锁定主机内存。
void freeHost(void *ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

// 同步内存拷贝：dst（目标）、src（源）、size（字节数）、kind（方向）。
// cudaMemcpy 默认是同步的：函数返回时数据一定已经拷完。
void memcpySync(void *dst, const void *src, size_t size, chaosuanMemcpyKind_t kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, to_cuda_kind(kind)));
}

// 异步内存拷贝：把拷贝任务放进 stream 队列就返回（不等它完成）。
// 之后必须 device_synchronize / stream_synchronize 才能安全读数据。
void memcpyAsync(void *dst, const void *src, size_t size, chaosuanMemcpyKind_t kind,
                 chaosuanStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, to_cuda_kind(kind),
                               reinterpret_cast<cudaStream_t>(stream)));
}

// ---------------------------------------------------------------------------
// 函数表：把上面 12 个函数的地址按顺序填进结构体。
// 注意顺序必须和 runtime.h 里 ChaosuanRuntimeAPI 的字段声明完全一致！
// （CPU 版也是这样填的，见 cpu_runtime_api.cpp。）
// ---------------------------------------------------------------------------
static const ChaosuanRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

// 对外入口：返回 CUDA 版函数表（被 device.cpp 的 getRuntimeAPI 调用，
// 只有定义了 ENABLE_NVIDIA_API 宏才会链接到这个文件）。
const ChaosuanRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace chaosuan::device::nvidia
