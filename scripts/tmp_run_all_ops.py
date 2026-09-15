import sys

sys.path.insert(0, r"G:\shi\chaosuan\scripts")
from ssh_run import run

cmd = (
    "cd /root/chaosuan && "
    "for t in add argmax embedding linear rms_norm rope self_attention swiglu; do "
    'r=$(PYTHONPATH=python python3 test/ops/$t.py 2>&1 | grep -E "Test passed|Error|error" | tail -1); '
    'echo "$t => $r"; done'
)
print(run(cmd))
