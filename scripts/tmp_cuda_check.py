import sys

sys.path.insert(0, r"G:\shi\chaosuan\scripts")
from ssh_run import run

print(run("ls /usr/local/cuda-12.8/bin/ | grep nvcc"))
print(run("/usr/local/cuda-12.8/bin/nvcc --version | tail -2"))
print(run("ls /usr/local/cuda/lib64/ | grep -E 'libcudart|libcublas' | head -8"))
