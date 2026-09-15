import ctypes
from ctypes import c_void_p, c_size_t, c_int, Structure, CFUNCTYPE
from .chaosuan_types import *

# Define function pointer types
get_device_count_api = CFUNCTYPE(c_int)
set_device_api = CFUNCTYPE(None, c_int)
device_synchronize_api = CFUNCTYPE(None)

create_stream_api = CFUNCTYPE(chaosuanStream_t)
destroy_stream_api = CFUNCTYPE(None, chaosuanStream_t)
stream_synchronize_api = CFUNCTYPE(None, chaosuanStream_t)

malloc_device_api = CFUNCTYPE(c_void_p, c_size_t)
free_device_api = CFUNCTYPE(None, c_void_p)
malloc_host_api = CFUNCTYPE(c_void_p, c_size_t)
free_host_api = CFUNCTYPE(None, c_void_p)

memcpy_sync_api = CFUNCTYPE(None, c_void_p, c_void_p, c_size_t, chaosuanMemcpyKind_t)
memcpy_async_api = CFUNCTYPE(None, c_void_p, c_void_p, c_size_t, chaosuanMemcpyKind_t, chaosuanStream_t)


# Define the struct matching ChaosuanRuntimeAPI
class ChaosuanRuntimeAPI(Structure):
    _fields_ = [
        ("get_device_count", get_device_count_api),
        ("set_device", set_device_api),
        ("device_synchronize", device_synchronize_api),
        ("create_stream", create_stream_api),
        ("destroy_stream", destroy_stream_api),
        ("stream_synchronize", stream_synchronize_api),
        ("malloc_device", malloc_device_api),
        ("free_device", free_device_api),
        ("malloc_host", malloc_host_api),
        ("free_host", free_host_api),
        ("memcpy_sync", memcpy_sync_api),
        ("memcpy_async", memcpy_async_api),
    ]


# Load shared library
def load_runtime(lib):
    # Declare API function prototypes
    lib.chaosuanGetRuntimeAPI.argtypes = [chaosuanDeviceType_t]
    lib.chaosuanGetRuntimeAPI.restype = ctypes.POINTER(ChaosuanRuntimeAPI)

    lib.chaosuanSetContextRuntime.argtypes = [chaosuanDeviceType_t, c_int]
    lib.chaosuanSetContextRuntime.restype = None
