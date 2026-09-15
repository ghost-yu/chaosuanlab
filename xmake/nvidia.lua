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

    add_includedirs("/usr/local/cuda/include")
    add_linkdirs("/usr/local/cuda/lib64")
    add_links("cudart", "cublas")         -- cuBLAS 给 linear 用（矩阵乘）

    -- RTX 4090 = Ada Lovelace = 计算能力 8.9。
    -- -Xcompiler 把参数透传给宿主编译器（g++），-fPIC 是共享库必需。
    add_cuflags("-arch=compute_89,code=sm_89", "-Xcompiler=-fPIC", "-Xcompiler=-Wno-unknown-pragmas")

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
