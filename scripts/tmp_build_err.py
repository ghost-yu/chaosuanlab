import sys

sys.path.insert(0, r"G:\shi\chaosuan\scripts")
from ssh_run import run

print(run("cd /root/chaosuan && XMAKE_ROOT=y xmake 2>&1 | grep -B2 -A8 -iE 'error' | head -50"))
