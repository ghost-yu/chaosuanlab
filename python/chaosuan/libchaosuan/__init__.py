import os
import sys
import ctypes
from pathlib import Path

from .runtime import load_runtime
from .runtime import ChaosuanRuntimeAPI
from .chaosuan_types import chaosuanDeviceType_t, DeviceType
from .chaosuan_types import chaosuanDataType_t, DataType
from .chaosuan_types import chaosuanMemcpyKind_t, MemcpyKind
from .chaosuan_types import chaosuanStream_t
from .tensor import chaosuanTensor_t
from .tensor import load_tensor
from .ops import load_ops


def load_shared_library():
    lib_dir = Path(__file__).parent

    if sys.platform.startswith("linux"):
        libname = "libchaosuan.so"
    elif sys.platform == "win32":
        libname = "chaosuan.dll"
    elif sys.platform == "darwin":
        libname = "chaosuan.dylib"
    else:
        raise RuntimeError("Unsupported platform")

    lib_path = os.path.join(lib_dir, libname)

    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Shared library not found: {lib_path}")

    return ctypes.CDLL(str(lib_path))


LIB_CHAOSUAN = load_shared_library()
load_runtime(LIB_CHAOSUAN)
load_tensor(LIB_CHAOSUAN)
load_ops(LIB_CHAOSUAN)


__all__ = [
    "LIB_CHAOSUAN",
    "ChaosuanRuntimeAPI",
    "chaosuanStream_t",
    "chaosuanTensor_t",
    "chaosuanDataType_t",
    "DataType",
    "chaosuanDeviceType_t",
    "DeviceType",
    "chaosuanMemcpyKind_t",
    "MemcpyKind",
    "chaosuanStream_t",
]
