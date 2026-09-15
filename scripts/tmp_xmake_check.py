import sys

sys.path.insert(0, r"G:\shi\chaosuan\scripts")
from ssh_run import run

print(run("xmake --version | head -2"))
print(run("ls /usr/local/cuda/include/ | grep -E 'cuda_runtime|cublas' | head -5"))
print(run("ls /usr/local/cuda/bin/ | head -20"))
