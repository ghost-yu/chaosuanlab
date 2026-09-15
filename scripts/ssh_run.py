import base64
import paramiko
import sys

HOST = "hn01-ssh.gpuhome.cc"
PORT = 31192
USER = "root"
PWD = "bja5e7vr"


def run(cmd, timeout=600, show_err=True):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD)
    full = (
        "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.local/bin:$PATH && "
        + cmd
    )
    b64 = base64.b64encode(full.encode()).decode()
    stdin, stdout, stderr = c.exec_command(f"echo {b64} | base64 -d | bash", timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    c.close()
    if show_err and err:
        print("[STDERR]", err[-1500:])
    return out


if __name__ == "__main__":
    cmd = sys.stdin.read()
    print(run(cmd), end="")
