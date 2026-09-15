-- ============================================================================
-- xmake/nvidia.lua —— CUDA（NVIDIA 平台）编译配置（作业 4）
-- ----------------------------------------------------------------------------
-- 什么时候被加载？xmake.lua 里开了 `xmake f --nv-gpu=y` 时，
-- xmake.lua 会 `includes("xmake/nvidia.lua")`，本文件才生效。
-- 关掉 nv-gpu 时 CUDA 代码完全不参与编译（README 的硬性要求）。
--
-- 本文件定义两个静态库 target：
--   1. chaosuan-device-nvidia：CUDA 版 Runtime API。
--      文件是普通 .cpp（只调用 CUDA 的 host API，如 cudaMalloc），
--      所以用 g++ 编译，链接 libcudart 即可。
--   2. chaosuan-ops-nvidia：8 个算子的 CUDA kernel 实现。
--      文件是 .cu（里面有 __global__ kernel），必须用 nvcc 编译，
--      xmake 的 add_rules("cuda") 会自动把 .cu 交给 nvcc。
--
-- 【RTX 4090 架构】Ada Lovelace，计算能力 8.9，所以 -arch=compute_89。
-- 如果要换显卡（比如 A100 是 8.0），改成对应的 compute capability。
-- ============================================================================

-- ---------------------------------------------------------------------------
-- target 1：CUDA Runtime API（.cpp，g++ 编译）
-- ---------------------------------------------------------------------------
target("chaosuan-device-nvidia")
    set_kind("static")                    -- 静态库：libchaosuan-device-nvidia.a
    set_languages("cxx17")
    set_warnings("all")                   -- 开启警告（不用 error，CUDA 头文件警告多）

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")   -- 共享库要求 PIC
    end

    -- CUDA 头文件目录 + 库目录 + 链接 CUDA 运行时
    add_includedirs("/usr/local/cuda/include")
    add_linkdirs("/usr/local/cuda/lib64")
    add_links("cudart")

    add_files("../src/device/nvidia/*.cpp")

    on_install(function (target) end)     -- 静态库不需要安装
target_end()

-- ---------------------------------------------------------------------------
-- target 2：CUDA 算子 kernel（.cu，nvcc 编译）
-- ---------------------------------------------------------------------------
target("chaosuan-ops-nvidia")
    set_kind("static")
    add_deps("chaosuan-tensor")           -- 算子依赖张量层

    set_languages("cxx17")
    set_warnings("all")

    -- 关键：告诉 xmake 用 CUDA 规则编译 .cu 文件（自动调 nvcc）
    add_rules("cuda")
    -- 关闭可重定位设备代码（-rdc=true）：xmake 默认开启，但 rdc 会生成
    -- __cudaRegisterLinkedBinary 符号（libcudart.so 不导出，g++ 链接报错）。
    -- 本项目 kernel 不跨文件调用，不需要 rdc，走常规 __cudaRegisterFatBinary 路径。
    set_values("cuda.rdc", false)

    add_includedirs("/usr/local/cuda/include")
    add_linkdirs("/usr/local/cuda/lib64")
    add_links("cudart", "cublas")         -- cuBLAS 给 linear 用（矩阵乘）

    -- RTX 4090 = Ada Lovelace = 计算能力 8.9。
    -- 注意必须写 -arch=sm_89（生成常规 cubin + __cudaRegisterFatBinary），
    -- 不能写 -arch=compute_89,code=sm_89：后者生成"可重定位设备代码"，
    -- 需要 __cudaRegisterLinkedBinary 符号（libcudart.so 不导出它，
    -- g++ 链接会报 undefined symbol）。详见教学笔记作业4 排错记录。
    -- -Xcompiler 把参数透传给宿主编译器（g++），-fPIC 是共享库必需。
    set_policy("check.auto_ignore_flags", false)
    -- -rdc=false 必须写：xmake 的 cuda 规则默认加 -rdc=true（可重定位设备代码），
    -- 它会生成 __cudaRegisterLinkedBinary 符号（libcudart.so 不导出，
    -- g++ 链接报 undefined symbol）。本项目 kernel 不跨文件调用，
    -- 不需要 rdc，显式关掉后走常规 __cudaRegisterFatBinary 路径。
    add_cuflags("-arch=sm_89", "-rdc=false", "-Xcompiler=-fPIC", "-Xcompiler=-Wno-unknown-pragmas")

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()

-- ---------------------------------------------------------------------------
-- 主共享库 libchaosuan.so 也要链接 CUDA 运行时：
-- nvcc 编译 .cu 时会生成 "__cudaRegisterLinkedBinary" 等 fatbin 注册符号，
-- 这些符号的实现都在 libcudart 里。如果主库不链接 cudart，
-- 加载 .so 时会报 undefined symbol（我们踩过的坑，见教学笔记）。
-- xmake 允许"重新打开"已存在的 target 追加配置，这里就是追加链接选项。
-- ---------------------------------------------------------------------------
target("chaosuan")
    add_linkdirs("/usr/local/cuda/lib64")
    add_links("cudart")
target_end()
