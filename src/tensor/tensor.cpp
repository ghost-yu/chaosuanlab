// ============================================================================
// tensor.cpp —— Tensor（张量）类的实现（作业 #1）
//
// 【给只有 C 语言基础的新生导读】
// 本项目用 C++17 编写，先掌握下面几个 C++ 概念再读代码：
//
//   1. 命名空间 namespace：C 的全局名字容易重名，C++ 用 namespace 把名字包起来，
//      本项目的类都放在 namespace chaosuan { ... } 里，外面访问要写 chaosuan::Tensor。
//
//   2. 类 class：C 的 struct 只能存数据；C++ 的 class 把"数据 + 操作数据的函数"放一起。
//      写在类里的函数叫"成员函数"，类外实现时用 类名::函数名 开头（如 Tensor::load）。
//
//   3. 引用 &：C 传参用指针 int *p；C++ 多了"引用" int &r，本质是变量的别名，
//      写起来像值、实际是同一个变量，避免拷贝。
//      const std::vector<size_t> &shape 意思是："shape 是外面那个 vector 的别名，
//      函数内不能改它（const），也不会复制一份（&）"。
//
//   4. std::vector<size_t>：标准库动态数组（自动管理内存、可 push_back 追加）。
//      size_t 是无符号 64 位整数，表示大小/下标；ptrdiff_t 是有符号的，
//      因为步长（strides）允许是负数（翻转等操作会产生）。
//
//   5. std::shared_ptr<T>（智能指针）：带引用计数的指针。多个 shared_ptr 指向同一块
//      内存时计数 +1，最后一个被销毁时自动释放内存。这里 tensor_t = std::shared_ptr<Tensor>。
//
//   6. std::move()：把对象的内部资源"偷走"而不是复制（避免深拷贝，省时间）。
//
//   7. 模板 template<typename T>：写一个函数，编译器按 T 自动生成多份代码。
//      例如 template <typename T> void print_data(...) 里 T 会被替换成 float/int 等。
//
//   8. if constexpr：模板函数里的"编译期 if"，条件编译期就能确定，
//      编译器只保留 true 那一支代码，false 的直接丢弃。
//
//   9. static_cast / reinterpret_cast：C 里强转写 (int)x；C++ 推荐 static_cast<int>(x)
//      （安全转换）；reinterpret_cast<const float *>(p) 相当于 C 的 (const float*)p，
//      不改变内存，只改变"看待这段内存的方式"。
//
//  10. std::byte：C++17 的"字节"类型，强调这是原始内存，不能直接当数算，
//      必须转成具体类型才能读写。数据区首地址就是 std::byte*。
//
//  11. 初始化列表 : _meta(...), _storage(...)：构造函数在函数体 { } 之前，
//      用 : 后面的列表初始化成员变量。
//
// 【张量是什么？】
// 张量 = 多维数组。除了数据本身，还需要"元数据"来描述它：
//   - shape（形状）：每一维有几个元素，如 {2,3,5} 表示 2×3×5 三维张量；
//   - strides（步长）：第 i 维下标 +1 时，内存地址往后跳几个元素。
//     连续（行主序）时：最后一维步长 = 1，倒数第二维 = 最后一维长度，依此类推。
//     例：shape={2,3,5} → 元素总数 30，步长 = {15, 5, 1}。
//     任意元素 (i,j,k) 的地址 = 起点 + (i*15 + j*5 + k*1) * 元素大小。
//   - offset（字节偏移）：本张量数据在 storage（内存块）里的起点，单位是字节。
//   - storage：真正的内存块（被多个张量共享，如切片/转置视图）。
//
// 作业 #1 要做 5 个小问（Task-1.1 ~ Task-1.5），对应下面 5 个函数。
// 每个函数先看注释理解原理，再看实现；注释里的"为什么"比代码本身更重要。
// ============================================================================

#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace chaosuan {

// ---------------------------------------------------------------------------
// 私有构造函数：创建 Tensor 对象时自动调用。
//   TensorMeta meta          —— 元数据（数据类型、形状、步长），按值传入；
//   core::storage_t storage  —— 存储对象（实际内存的"管家"），按值传入；
//   size_t offset            —— 数据相对存储起点的字节偏移（默认 0）。
//
// : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset)
// 是"成员初始化列表"：在进入函数体之前把三个成员变量初始化好。
// std::move 把 meta / storage 内部资源直接移交，避免深拷贝。
// 注意：这个构造函数是私有的（private），外面不能直接 new，
// 只能通过 create() 等工厂函数创建 —— 这是为了确保 Tensor 永远合法。
// ---------------------------------------------------------------------------
Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

// ---------------------------------------------------------------------------
// 工厂函数 create()：按形状/类型/设备创建一个新张量。
//
// 参数解释（典型的 C++ 引用传参，见文件头导读第 3 条）：
//   const std::vector<size_t> &shape  —— 形状，如 {2,3,5} 表示 2×3×5 三维张量；
//   chaosuanDataType_t dtype          —— 数据类型（见 include/chaosuan.h 的枚举）；
//   chaosuanDeviceType_t device_type  —— 设备类型（CPU=0 / NVIDIA=1）；
//   int device                        —— 设备编号（多张显卡时用，CPU 时是 0）。
//
// 做四件事：
//   1. 根据 shape 算出"默认连续步长" strides；
//   2. 算出总元素数 total_elems 和总字节数；
//   3. 向 Runtime 申请一块内存（主机内存或显存），包装成 Storage；
//   4. 用 new Tensor(...) 构造对象，包进 shared_ptr 返回。
//
// 【什么是步长 strides？—— 整个项目最重要概念，务必看懂】
// 张量数据在内存里是一长串。要把它当成多维数组看，就得知道：
// 第 i 维下标 +1 时，内存地址要往后跳几个元素 —— 这就是 strides[i]。
// 连续排列（row-major，行主序）时从最后一维往前累乘：
//   最后一维步长 = 1；倒数第二维步长 = 最后一维的长度；再往前依此类推。
// 例：shape={2,3,5} → 元素总数 30，步长 = {15, 5, 1}。
// ---------------------------------------------------------------------------
tensor_t Tensor::create(const std::vector<size_t> &shape,
                        chaosuanDataType_t dtype,
                        chaosuanDeviceType_t device_type,
                        int device) {
    // shape.size() 返回 vector 里元素个数，即张量维数（2×3×5 就是 3 维）。
    size_t ndim_ = shape.size();

    // 申请一个和 shape 等长的 vector 用来放步长（初始全是 0）。
    std::vector<ptrdiff_t> strides(ndim_);

    // ptrdiff_t 是"指针差值"类型（有符号整数），因为步长允许是负的。
    // stride 变量从 1 开始，从最后一维往前"累乘"出每一步长。
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        // 第一次循环 i=1：strides[ndim_-1]（最后一维）= 1，然后 stride *= shape[最后一维]
        // 第二次循环 i=2：strides[ndim_-2]（倒数第二维）= 上一轮累乘结果，再乘……
        // 这样从后往前，每一维的步长 = 它后面所有维度的乘积。
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    // 例如 shape={2,3,5}：循环结束后 strides = {15, 5, 1}，stride = 30（总元素数）。

    // TensorMeta 是 struct，用 {dtype, shape, strides} 聚合初始化（按声明顺序填字段）。
    TensorMeta meta{dtype, shape, strides};

    // 上面循环结束后，stride 恰好等于所有维度的乘积 = 总元素数（2*3*5=30）。
    size_t total_elems = stride;

    // utils::dsize(dtype)：查表返回该类型每个元素占几个字节（F32 → 4 字节）。
    size_t dtype_size = utils::dsize(dtype);

    // 分配内存：
    //   - 如果用户要 CPU 张量，但当前线程的"当前设备"不是 CPU
    //     （比如之前激活过 GPU），走 allocateHostStorage 分配"主机内存"。
    //   - 否则先 setDevice 把当前线程切到目标设备，再分配设备内存（显存）。
    //
    // core::context() 是全局单例（整个程序只有一份的 Context 对象），
    // 保存当前设备状态；.runtime() 返回当前运行时（封装了 CPU/GPU 操作）。
    if (device_type == CHAOSUAN_DEVICE_CPU && core::context().runtime().deviceType() != CHAOSUAN_DEVICE_CPU) {
        // 申请 total_elems * dtype_size 字节的主机内存，包装成 Storage 对象。
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);

        // new Tensor(meta, storage) 在堆上构造对象；shared_ptr 接管它。
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        // 先切设备（确保后续分配/计算作用在正确的 GPU 上），再分配显存。
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

// ---------------------------------------------------------------------------
// data()：返回本张量数据的起始地址。
//   _storage->memory() —— Storage 里保存的内存块首地址（std::byte*）；
//   _offset           —— 字节偏移（本张量数据在这个内存块里的起点）。
// 两者相加 = 本张量真正数据的地址。
// 注意：_offset 的单位是**字节**！不是元素个数。
// ---------------------------------------------------------------------------
std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

// const 版本：当调用者是 const 对象（只读）时，编译器选这个重载，
// 返回 const 指针，保证外面不能通过它修改数据。
const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

// ndim()：返回维度数（张量是几维的），就是 shape 数组的长度。
size_t Tensor::ndim() const {
    return _meta.shape.size();
}

// shape()：返回形状。返回"const 引用"（const std::vector<size_t>&），
// 意思是：直接把内部 _meta.shape 交出去，不复制；调用者只能看不能改。
const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

// strides()：返回步长，同样是 const 引用（不复制、只读）。
const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

// dtype()：返回数据类型（枚举值，如 CHAOSUAN_DTYPE_F32）。
chaosuanDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

// deviceType()：返回设备类型（CPU 或 GPU）。
// 实现是直接问 Storage（内存是谁分配的），因为内存类型才是最可靠的。
chaosuanDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

// deviceId()：返回设备编号（第几张显卡）。
int Tensor::deviceId() const {
    return _storage->deviceId();
}

// ---------------------------------------------------------------------------
// numel()：返回张量包含的元素总数 = shape 各维相乘。
// 用标准库函数 std::accumulate（在 <numeric> 里）：
//   accumulate(begin, end, 初始值, 二元操作) 把 [begin,end) 范围内的元素
//   逐个用"二元操作"累积起来。这里就是 1 * shape[0] * shape[1] * ... = 总元素数。
// ---------------------------------------------------------------------------
size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

// elementSize()：单个元素占多少字节（比如 float 是 4）。
size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

// ---------------------------------------------------------------------------
// info()：把张量的元信息拼成一行可读字符串，方便打印调试。
// std::stringstream 是"字符串拼接工具"：用法和 std::cout 一样（都用 <<），
// 但输出目标是内存里的字符串，最后 .str() 取出。
// ---------------------------------------------------------------------------
std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {   // 范围 for：逐个取出 shape 的每个元素
        ss << s << " ";              // 拼成 "shape[ 2 3 5 ] strides[ 15 5 1 ] ..."
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

// ---------------------------------------------------------------------------
// print_data<T>：递归打印张量数据（模板函数，见导读第 7 条）。
//
// 参数：
//   data    —— 数据区指针（已转成 T*，T 是元素类型）；
//   shape   —— 形状；strides —— 步长；dim —— 当前递归到第几维（从 0 开始）。
//
// 为什么要递归？
//   因为张量可能是"非连续"的（比如 permute 之后），不能简单从头打印到尾，
//   必须按"形状"一层层展开：第 0 维有 shape[0] 个"行"，每行内部又是一个子张量。
//   递归到最后一维时，这一行内部才真正连续，可以逐个打印。
// 关键点：在第 dim 维，下标 i 的元素，在内存中的位置是 data + i * strides[dim]。
// ---------------------------------------------------------------------------
template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {   // 递归到最后一维：这一行可以逐个打印了
        for (size_t i = 0; i < shape[dim]; i++) {
            // if constexpr（编译期 if，见导读第 8 条）：
            // 如果 T 是 bf16_t 或 fp16_t（两种 16 位低精度浮点），直接 cout 会打印乱码，
            // 所以先用 utils::cast<float>(...) 把它转成 float 再打印。
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                // 其他类型（int、float、double...）直接打印。
                // data[i * strides[dim]]：第 i 个元素的地址按步长定位。
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;   // 每打印完一行换行
    } else if (dim < shape.size() - 1) {   // 不是最后一维：对每个"行头"递归
        for (size_t i = 0; i < shape[dim]; i++) {
            // 第 i 行的起点 = data + i * strides[dim]，接着递归打印下一维。
            // data + i * strides[dim] 是指针算术：指针加 n 会往后跳 n 个 T 元素。
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

// ---------------------------------------------------------------------------
// debug_print：按数据类型把"字节指针"转成"具体类型指针"，再交给 print_data。
// 因为 data() 返回 std::byte*（原始字节），必须根据 dtype 枚举（switch 分支）
// 决定"把这串字节当成 float 数组 / int 数组 / ..."来看。
// reinterpret_cast<const float *>(data) 就是 C 的 (const float*)data：
// 不改变内存内容，只是告诉编译器"按 float 的方式解释这段内存"。
// ---------------------------------------------------------------------------
void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, chaosuanDataType_t dtype) {
    switch (dtype) {
    case CHAOSUAN_DTYPE_BYTE:   // 每个 case 对应一种数据类型，调用对应 T 的模板
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case CHAOSUAN_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case CHAOSUAN_DTYPE_I8:     // int8_t 就是 C 里的 signed char（有符号 8 位整数）
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case CHAOSUAN_DTYPE_I16:    // int16_t：16 位有符号整数（short）
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case CHAOSUAN_DTYPE_I32:    // int32_t：32 位有符号整数（int）
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case CHAOSUAN_DTYPE_I64:    // int64_t：64 位有符号整数（long long）
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case CHAOSUAN_DTYPE_U8:     // uint8_t：无符号 8 位整数（unsigned char）
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case CHAOSUAN_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case CHAOSUAN_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case CHAOSUAN_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case CHAOSUAN_DTYPE_F16:    // fp16_t：16 位半精度浮点（项目自定义类型）
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case CHAOSUAN_DTYPE_F32:    // float：32 位单精度浮点（最常用）
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case CHAOSUAN_DTYPE_F64:    // double：64 位双精度浮点
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case CHAOSUAN_DTYPE_BF16:   // bf16_t：Brain Float 16（项目自定义，深度学习常用）
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        // 走到这里说明 dtype 枚举值不在上面列表里（不可能发生），抛异常。
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

// ---------------------------------------------------------------------------
// debug()：把张量完整打印出来（元信息 + 所有数据），调试用。
//
// 关键难点：如果张量在 GPU 显存里，CPU 是**不能直接读显存**的！
// 所以流程是：
//   1. 同步设备：等所有异步操作（比如 GPU kernel、异步拷贝）都完成；
//   2. 打印元信息（info）；
//   3. 如果是 CPU 张量：直接打印；
//   4. 如果是 GPU 张量：先创建一块临时 CPU 张量，调用 memcpy_sync
//      把显存数据拷回主机内存（D2H = Device To Host），再打印拷回来的副本。
// ---------------------------------------------------------------------------
void Tensor::debug() const {
    // 1. 同步设备：setDevice 切到本张量所在设备；device_synchronize 等待 GPU 操作完成。
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();

    std::cout << this->info() << std::endl;   // 先打印元信息

    if (this->deviceType() == CHAOSUAN_DEVICE_CPU) {
        // CPU 张量：data() 就是主机地址，直接交给 debug_print。
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        // GPU 张量：先 create 一块大小 = 本张量总字节数的 CPU 张量当"临时中转站"。
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());

        // memcpy_sync：同步内存拷贝（拷完才返回），函数指针来自 API 表。
        //   参数1 tmp_tensor->data() —— 目标地址（主机内存）；
        //   参数2 this->data()       —— 源地址（显存）；
        //   参数3 numel()*elementSize() —— 字节数；
        //   参数4 CHAOSUAN_MEMCPY_D2H —— 方向：Device(显存) To Host(主机)。
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            CHAOSUAN_MEMCPY_D2H);

        // 拷回来后，用 this 的 shape/strides 去解释 tmp_tensor 的数据
        // （tmp_tensor 的形状只是"字节数"一维数组，必须用原张量的形状打印）。
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

// ---------------------------------------------------------------------------
// 【Task-1.2】isContiguous()：判断张量在内存里是否"连续"。
//
// 【背景：什么叫连续？】
// 假设一个 2×3 的张量（shape={2,3}）。
//   - 连续（contiguous）：6 个元素在内存里紧挨着排成一行，
//     先放第 0 行的 3 个，再放第 1 行的 3 个（row-major 行主序）。
//     此时 strides = {3, 1}。
//   - 不连续：比如做了 transpose 转置后，shape 还是 {2,3} 但 strides
//     变成 {1,2}，意味着元素在内存里是"跳着"取的，不挨着。
//
// 【判断方法】
// 连续的必要且充分条件是：按行主序，每个维度的步长"恰好等于
// 它后面所有维度大小的乘积"（最后一维步长 = 1）。
// 所以算法：从最后一维往前遍历，维护一个 expected_stride（期望步长），
//   对每一维检查：如果该维大小不为 1，则实际步长必须 == expected_stride；
//   然后把 expected_stride 乘上这一维的大小，继续往前。
// 为什么大小是 1 的维度可以跳过？
//   因为该维只有一个元素，无论步长是多少，取元素时都不会"跳"，
//   不影响"内存是否紧密排列"。
// ---------------------------------------------------------------------------
bool Tensor::isContiguous() const {
    if (this->numel() == 0) {   // 空张量（0 个元素）：没有元素可言，算连续
        return true;
    }

    // expected_stride：期望步长，从最后一维的 1 开始，往前往前累乘。
    // 用 ptrdiff_t（有符号）是为了后面乘 shape 时不溢出无符号类型。
    ptrdiff_t expected_stride = 1;

    for (size_t i = this->ndim(); i > 0; --i) {
        const size_t dim = i - 1;

        // 维度大小不为 1 时，步长必须恰好等于期望值，否则不连续；
        // 维度大小为 1 时跳过（只有一个元素，步长不影响连续性）。
        if (_meta.shape[dim] != 1 && _meta.strides[dim] != expected_stride) {
            return false;
        }
        // 累乘更新期望值：往前的维度，期望步长要乘上这一维的大小。
        // static_cast<ptrdiff_t>：把 size_t（无符号）转成 ptrdiff_t（有符号）。
        expected_stride *= static_cast<ptrdiff_t>(_meta.shape[dim]);
    }

    return true;   // 所有维度都检查通过，说明连续
}

// ---------------------------------------------------------------------------
// 【Task-1.4】permute(order)：维度重排（转置的推广）。
// 见后续小问实现。
// ---------------------------------------------------------------------------
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

// ---------------------------------------------------------------------------
// 【Task-1.3】view(shape)：重塑视图（reshape，但不复制数据）。
//
// 【目标】
// 把张量"换一种形状解释"，同时**不复制数据**。比如把 2×3×5 看成 2×15。
// 成功的关键约束：新形状的元素总数必须等于旧形状（2*3*5 == 2*15），
// 且新视图必须与旧张量的内存布局"兼容"——也就是说，在不搬数据的前提下
// 能表达出新形状的连续排列（或至少是合法的非连续排列）。
//
// 【为什么不是随便改形状就行？】
// 例：一个连续的张量 (2,3,5)（步长 {15,5,1}），内存是紧密的一排 30 个元素，
// 可以 view 成任何"元素总数相同"的形状（2×15、3×10、30…），因为内存怎么分段都行。
// 但如果先 permute 变成不连续（步长 {1,15,5} 这种），内存就不是紧密排列了，
// 此时想 view 成 (2,15) 就**做不到**：新形状要求元素按 15 个一组连续，
// 但内存里元素是跳着放的。本函数要检测并拒绝这种操作。
//
// 【算法：块（chunk）对齐】
// 把旧张量从最后一维往前进，切成若干"连续块"（chunk）：
//   一个块 = 内存里连续的一段元素（在块内，步长是连续排列的）。
// 块在哪里断开？当某个维度的步长不等于"连续应走的步长"时断开。
// 然后从新形状的最后一维往前，逐个"消耗"这些块，算出每个新维度的步长。
// 如果某个块的元素数和新维度对不上，说明不兼容，报错。
//
// 更直观的理解：这就像把一串珠子（旧张量的内存）重新分成若干段
// （新形状的各维）。只要"断点"能对齐，就能成功；对不齐就失败。
// ---------------------------------------------------------------------------
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    const size_t old_numel = this->numel();   // 旧元素总数（旧 shape 各维乘积）
    const size_t new_numel = std::accumulate(   // 新元素总数（新 shape 各维乘积）
        shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    // 元素总数必须一致（view 不改变元素个数）。
    CHECK_ARGUMENT(old_numel == new_numel, "View shape must preserve the number of elements");

    // 申请一个和 shape 等长的 vector，用来放新步长。
    std::vector<ptrdiff_t> new_strides(shape.size());

    // ---------------- 情况 1：空张量（0 个元素） ----------------
    // 没有元素，任何形状都合法（数据是空的，无所谓连续不连续）。
    if (old_numel == 0) {
        if (shape == _meta.shape) {   // 形状没变：直接沿用旧步长
            new_strides = _meta.strides;
        } else {                      // 形状变了：按新形状生成"连续步长"
            // 从最后一维往前累乘，生成连续步长（和 create 里一样）。
            ptrdiff_t stride = 1;
            for (size_t i = shape.size(); i > 0; --i) {
                new_strides[i - 1] = stride;
                stride *= static_cast<ptrdiff_t>(shape[i - 1]);
            }
        }

        // 构造新张量：共享同一 storage，offset 不变（数据本来就是空的）。
        TensorMeta meta{_meta.dtype, shape, std::move(new_strides)};
        return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
    }

    // ---------------- 情况 2：旧形状为空（标量张量） ----------------
    // 标量 = 0 维张量（只有一个元素，shape 是空 vector）。
    // 元素就一个，怎么 view 都行：按新形状生成连续步长即可。
    if (_meta.shape.empty()) {
        ptrdiff_t stride = 1;
        for (size_t i = shape.size(); i > 0; --i) {
            new_strides[i - 1] = stride;
            stride *= static_cast<ptrdiff_t>(shape[i - 1]);
        }

        TensorMeta meta{_meta.dtype, shape, std::move(new_strides)};
        return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
    }

    // -----------------------------------------------------------------------
    // ---------------- 情况 3：常规情况 —— "块对齐"算法 ----------------
    // 思路回顾：
    //   从旧张量的最后一维往前扫描，把旧张量分解成若干"连续块"；
    //   同时从新形状的最后一维往前"消耗"这些块，为每个新维度算出步长。
    // -----------------------------------------------------------------------

    // view_dim：当前正在处理的新形状维度的下标（从最后一维开始）。
    // 用 ptrdiff_t 是因为它最后要递减到 -1（表示全部处理完），
    // size_t 是 unsigned，减到 0 再减会变成巨大的数，判不了 < 0。
    ptrdiff_t view_dim = static_cast<ptrdiff_t>(shape.size()) - 1;

    // tensor_numel：当前旧"块"已累计的元素数（从 1 开始累乘）。
    // view_numel：当前新维度已累计的元素数（从 1 开始累乘）。
    size_t tensor_numel = 1;
    size_t view_numel = 1;

    // chunk_base_stride：当前块内的"基步长"——块内最细粒度那一步的长度。
    // 初始取旧张量最后一维的步长（连续块的起点通常从最后一维开始）。
    ptrdiff_t chunk_base_stride = _meta.strides.back();
    // _meta.strides.back()：vector 的 back() 返回最后一个元素。

    // 从旧最后一维往前扫描（i 从 shape.size() 递减到 1，dim = i-1）。
    for (size_t i = _meta.shape.size(); i > 0; --i) {
        const size_t tensor_dim = i - 1;
        tensor_numel *= _meta.shape[tensor_dim];         // 累计当前旧块的大小

        // 判断"块是否在此处结束"：
        //   结束条件 1：已经到第 0 维（最前面，前面没有东西了）；
        //   结束条件 2：当前维的"前一个维度"步长不等于"连续应走的步长"。
        //     连续应走的步长 = 当前块累计元素数 × 基步长。
        //     如果不相等，说明前一个维度的元素和本块不连续，块到此结束。
        //   （shape[tensor_dim - 1] != 1 的检查：大小为 1 的维度不参与
        //     连续性判断，理由同 isContiguous。）
        const bool chunk_ends =
            tensor_dim == 0
            || (_meta.shape[tensor_dim - 1] != 1
                && _meta.strides[tensor_dim - 1]
                       != static_cast<ptrdiff_t>(tensor_numel) * chunk_base_stride);

        if (!chunk_ends) {
            continue;   // 块还没结束，继续向前累加（进入下一个循环）
        }

        // ---- 块结束：用这个块去"喂"新形状的维度 ----
        // 从新形状的最后一维开始往前，只要满足下面两个条件之一，
        // 就给当前新维度分配步长：
        //   a) 当前新维度累计大小 < 块大小（还能继续填）；
        //   b) 新维度大小为 1（单个元素的维度，怎么填都行）。
        while (view_dim >= 0
               && (view_numel < tensor_numel || shape[static_cast<size_t>(view_dim)] == 1)) {
            // 新维度步长 = 当前累计元素数 × 基步长。
            // 例：块大小 15，基步长 1，第一个新维度（最后一维）步长 = 1*1=1，
            //     累加后 view_numel=shape[最后]，第二个新维度步长 = shape[最后]*1。
            new_strides[static_cast<size_t>(view_dim)] =
                static_cast<ptrdiff_t>(view_numel) * chunk_base_stride;
            view_numel *= shape[static_cast<size_t>(view_dim)];          // 累加新维度大小
            --view_dim;                                                  // 向前推进一个新维度
        }

        // 关键校验：块的大小必须恰好被新维度消耗完。
        // 如果 view_numel != tensor_numel，说明这个块没能被新形状完整"吃掉"，
        // 即新形状与旧内存布局不兼容（比如把非连续内存硬 reshape 成
        // 一个不存在的连续形状），直接报错。
        CHECK_ARGUMENT(view_numel == tensor_numel,
                       "View shape is incompatible with the tensor's shape and strides");

        // 如果后面还有更前面的旧维度（tensor_dim > 0）：
        // 重置块统计（tensor_numel / view_numel 归 1），
        // 并把基步长更新为"前一个维度的步长"（新的块从那里开始）。
        if (tensor_dim > 0) {
            chunk_base_stride = _meta.strides[tensor_dim - 1];
            tensor_numel = 1;
            view_numel = 1;
        }
    }

    // 循环结束后，所有新维度都必须被分配完成（view_dim 应该推进到 -1）。
    // 如果还有没分配的新维度，说明新形状太大/块不够用，不兼容。
    CHECK_ARGUMENT(view_dim == -1,
                   "View shape is incompatible with the tensor's shape and strides");

    // 构造新张量：共享同一 storage（数据没动），offset 不变。
    TensorMeta meta{_meta.dtype, shape, std::move(new_strides)};
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

// ---------------------------------------------------------------------------
// 【Task-1.5】slice(dim, start, end)：沿第 dim 维切片，取 [start, end) 区间。
// 见后续小问实现。
// ---------------------------------------------------------------------------
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

// ---------------------------------------------------------------------------
// 【Task-1.1】load(src)：把主机（CPU）内存里的数据拷进本张量。
//
// 【这是"CPU 数据进入 GPU"的入口】
// 场景：你在 CPU 上算好了一堆数据（比如读入的图片像素），想放进张量里
// （张量可能在显存里），就调用 load。
//
// 步骤：
//   1. 先切换到本张量所在的设备（确保后面的拷贝操作作用于正确的 GPU）；
//   2. 调运行时函数表的 memcpy_sync 执行拷贝：
//        目标 = this->data()（本张量的数据地址，可能是显存指针）；
//        源   = src_（你传进来的主机内存指针）；
//        长度 = 元素数 × 每元素字节数；
//        方向 = CHAOSUAN_MEMCPY_H2D（Host 主机 → Device 设备）。
//   如果是 CPU 张量，这套代码就是普通的内存拷贝（主机到主机），
//   同一套代码通吃，所以不用区分设备类型。
//
// memcpy_sync 里的 sync 表示"同步拷贝"：函数返回时数据一定已经拷完
// （CUDA 里对应 cudaMemcpy，默认同步）。
// ---------------------------------------------------------------------------
void Tensor::load(const void *src_) {
    // setDevice：把当前线程切到目标设备（多 GPU 时必须先切对）。
    core::context().setDevice(this->deviceType(), this->deviceId());

    // api() 返回一个函数表（结构体，里面全是函数指针），
    // 根据当前运行时（CPU 或 CUDA）指向不同的实现。
    // memcpy_sync 参数依次是：目标地址、源地址、字节数、方向。
    core::context().runtime().api()->memcpy_sync(
        this->data(),       // 目标：张量数据地址（可能是显存）
        src_,               // 源：主机内存（用户传入的 const void*）
        this->numel() * this->elementSize(),  // 字节数 = 元素数 × 每元素字节
        CHAOSUAN_MEMCPY_H2D);   // 方向枚举：Host → Device
}

// ---------------------------------------------------------------------------
// 以下三个函数（contiguous / reshape / to）在作业 #1 中**不需要实现**，
// 是后续作业的任务。它们目前都调用 TO_BE_IMPLEMENTED()（抛"未实现"异常），
// 返回一个与原张量共享存储的假张量占位，保证程序能编译链接通过。
// ---------------------------------------------------------------------------

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(chaosuanDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace chaosuan
