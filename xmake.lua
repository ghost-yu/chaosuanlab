add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

target("chaosuan-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("chaosuan-device")
    set_kind("static")
    add_deps("chaosuan-utils")
    add_deps("chaosuan-device-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("chaosuan-core")
    set_kind("static")
    add_deps("chaosuan-utils")
    add_deps("chaosuan-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("chaosuan-tensor")
    set_kind("static")
    add_deps("chaosuan-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("chaosuan-ops")
    set_kind("static")
    add_deps("chaosuan-ops-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("chaosuan")
    set_kind("shared")
    add_deps("chaosuan-utils")
    add_deps("chaosuan-device")
    add_deps("chaosuan-core")
    add_deps("chaosuan-tensor")
    add_deps("chaosuan-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/chaosuan/*.cc")
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying chaosuan to python/chaosuan/libchaosuan/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/chaosuan/libchaosuan/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/chaosuan/libchaosuan/")
        end
    end)
target_end()