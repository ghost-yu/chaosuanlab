import sys

sys.path.insert(0, r"G:\shi\chaosuan\scripts")
from ssh_run import run

print(run(
    "cd /root/chaosuan && cp build/linux/x86_64/release/libchaosuan.so python/chaosuan/libchaosuan/ && "
    "PYTHONPATH=python python3 test/test_runtime.py --device nvidia 2>&1 | tail -12"
))
