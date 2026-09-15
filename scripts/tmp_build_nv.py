import sys

sys.path.insert(0, r"G:\shi\chaosuan\scripts")
from ssh_run import run

cmd = (
    "cd /root && tar -xzf chaosuan.tar.gz -C chaosuan && cd chaosuan && "
    "XMAKE_ROOT=y xmake f --nv-gpu=y -cv 2>&1 | tail -3 && "
    "XMAKE_ROOT=y xmake 2>&1 | tail -3"
)
print(run(cmd, timeout=1500))
