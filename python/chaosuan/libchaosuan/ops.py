from .tensor import chaosuanTensor_t
from ctypes import c_float

def load_ops(lib):
    lib.chaosuanAdd.argtypes = [chaosuanTensor_t, chaosuanTensor_t, chaosuanTensor_t]
    lib.chaosuanAdd.restype = None

    lib.chaosuanArgmax.argtypes = [chaosuanTensor_t, chaosuanTensor_t, chaosuanTensor_t]
    lib.chaosuanArgmax.restype = None

    lib.chaosuanEmbedding.argtypes = [chaosuanTensor_t, chaosuanTensor_t, chaosuanTensor_t]
    lib.chaosuanEmbedding.restype = None

    lib.chaosuanLinear.argtypes = [chaosuanTensor_t, chaosuanTensor_t, chaosuanTensor_t, chaosuanTensor_t]
    lib.chaosuanLinear.restype = None

    lib.chaosuanRearrange.argtypes = [chaosuanTensor_t, chaosuanTensor_t]
    lib.chaosuanRearrange.restype = None

    lib.chaosuanRmsNorm.argtypes = [chaosuanTensor_t, chaosuanTensor_t, chaosuanTensor_t, c_float]
    lib.chaosuanRmsNorm.restype = None

    lib.chaosuanROPE.argtypes = [chaosuanTensor_t, chaosuanTensor_t, chaosuanTensor_t, c_float]
    lib.chaosuanROPE.restype = None

    lib.chaosuanSelfAttention.argtypes = [
        chaosuanTensor_t,  # attn_val
        chaosuanTensor_t,  # q
        chaosuanTensor_t,  # k
        chaosuanTensor_t,  # v
        c_float    # scale
    ]
    lib.chaosuanSelfAttention.restype = None

    lib.chaosuanSwiGLU.argtypes = [chaosuanTensor_t, chaosuanTensor_t, chaosuanTensor_t]
    lib.chaosuanSwiGLU.restype = None
