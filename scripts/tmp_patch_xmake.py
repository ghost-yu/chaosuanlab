import io

p = r"G:\shi\chaosuan\xmake.lua"
s = io.open(p, encoding="utf-8").read()

old = """target("chaosuan-device")
    set_kind("static")
    add_deps("chaosuan-utils")
    add_deps("chaosuan-device-cpu")
"""
new = """target("chaosuan-device")
    set_kind("static")
    add_deps("chaosuan-utils")
    add_deps("chaosuan-device-cpu")
    if has_config("nv-gpu") then
        -- 作业 4：开了 --nv-gpu=y 才编译并链接 CUDA Runtime API
        add_deps("chaosuan-device-nvidia")
    end
"""
assert old in s, "device block not found"
s = s.replace(old, new)

old2 = """target("chaosuan-ops")
    set_kind("static")
    add_deps("chaosuan-ops-cpu")
"""
new2 = """target("chaosuan-ops")
    set_kind("static")
    add_deps("chaosuan-ops-cpu")
    if has_config("nv-gpu") then
        -- 作业 4：开了 --nv-gpu=y 才编译并链接 CUDA 算子
        add_deps("chaosuan-ops-nvidia")
    end
"""
assert old2 in s, "ops block not found"
s = s.replace(old2, new2)

io.open(p, "w", encoding="utf-8").write(s)
print("xmake.lua updated")
