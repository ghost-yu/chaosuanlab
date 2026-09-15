from ctypes import POINTER, c_uint8, c_void_p, c_size_t, c_ssize_t, c_int
from .chaosuan_types import chaosuanDataType_t, chaosuanDeviceType_t

# Handle type
chaosuanTensor_t = c_void_p


def load_tensor(lib):
    lib.tensorCreate.argtypes = [
        POINTER(c_size_t),  # shape
        c_size_t,  # ndim
        chaosuanDataType_t,  # dtype
        chaosuanDeviceType_t,  # device_type
        c_int,  # device_id
    ]
    lib.tensorCreate.restype = chaosuanTensor_t

    # Function: tensorDestroy
    lib.tensorDestroy.argtypes = [chaosuanTensor_t]
    lib.tensorDestroy.restype = None

    # Function: tensorGetData
    lib.tensorGetData.argtypes = [chaosuanTensor_t]
    lib.tensorGetData.restype = c_void_p

    # Function: tensorGetNdim
    lib.tensorGetNdim.argtypes = [chaosuanTensor_t]
    lib.tensorGetNdim.restype = c_size_t

    # Function: tensorGetShape
    lib.tensorGetShape.argtypes = [chaosuanTensor_t, POINTER(c_size_t)]
    lib.tensorGetShape.restype = None

    # Function: tensorGetStrides
    lib.tensorGetStrides.argtypes = [chaosuanTensor_t, POINTER(c_ssize_t)]
    lib.tensorGetStrides.restype = None

    # Function: tensorGetDataType
    lib.tensorGetDataType.argtypes = [chaosuanTensor_t]
    lib.tensorGetDataType.restype = chaosuanDataType_t

    # Function: tensorGetDeviceType
    lib.tensorGetDeviceType.argtypes = [chaosuanTensor_t]
    lib.tensorGetDeviceType.restype = chaosuanDeviceType_t

    # Function: tensorGetDeviceId
    lib.tensorGetDeviceId.argtypes = [chaosuanTensor_t]
    lib.tensorGetDeviceId.restype = c_int

    # Function: tensorDebug
    lib.tensorDebug.argtypes = [chaosuanTensor_t]
    lib.tensorDebug.restype = None

    # Function: tensorIsContiguous
    lib.tensorIsContiguous.argtypes = [chaosuanTensor_t]
    lib.tensorIsContiguous.restype = c_uint8

    # Function: tensorLoad
    lib.tensorLoad.argtypes = [chaosuanTensor_t, c_void_p]
    lib.tensorLoad.restype = None

    # Function: tensorView(chaosuanTensor_t tensor, size_t *shape);
    lib.tensorView.argtypes = [chaosuanTensor_t, POINTER(c_size_t), c_size_t]
    lib.tensorView.restype = chaosuanTensor_t

    # Function: tensorPermute(chaosuanTensor_t tensor, size_t *order);
    lib.tensorPermute.argtypes = [chaosuanTensor_t, POINTER(c_size_t)]
    lib.tensorPermute.restype = chaosuanTensor_t

    # Function: tensorSlice(chaosuanTensor_t tensor,
    #                     size_t dim, size_t start, size_t end);
    lib.tensorSlice.argtypes = [
        chaosuanTensor_t,  # tensor handle
        c_size_t,  # dim  : which axis to slice
        c_size_t,  # start: inclusive
        c_size_t,  # end  : exclusive
    ]
    lib.tensorSlice.restype = chaosuanTensor_t
