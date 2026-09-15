# ============================================================================
# qwen2.py —— Qwen2 模型 C API 的 ctypes 包装（作业 3）
# ----------------------------------------------------------------------------
# 作用：把 C++ 侧（src/chaosuan/qwen2.cc）导出的 4 个 C 函数，包装成
# Python 能调用的函数。同时定义两个 C 结构体在 Python 里的对应物：
#   - ChaosuanQwen2Meta：模型超参数（层数、维度、词表等）；
#   - ChaosuanQwen2Weights：所有权重张量的指针集合。
# ctypes 里"结构体"用 class ...(Structure) 定义，_fields_ 的顺序和类型
# 必须与 C 头文件（include/chaosuan/models/qwen2.h）**完全一致**，
# 否则内存布局对不上，读出来就是乱码/段错误。
# ============================================================================

from ctypes import (
    POINTER,
    Structure,
    c_float,
    c_int,
    c_int64,
    c_size_t,
    c_void_p,
)

from .chaosuan_types import chaosuanDataType_t, chaosuanDeviceType_t
from .tensor import chaosuanTensor_t


# ---------------------------------------------------------------------------
# C 结构体：ChaosuanQwen2Meta（对应 include/chaosuan/models/qwen2.h）
# ---------------------------------------------------------------------------
class ChaosuanQwen2Meta(Structure):
    _fields_ = [
        ("dtype", chaosuanDataType_t),   # 权重数据类型（BF16 = 19）
        ("nlayer", c_size_t),            # 层数（28）
        ("hs", c_size_t),                # hidden_size（1536）
        ("nh", c_size_t),                # 注意力头数（12）
        ("nkvh", c_size_t),              # KV 头数（GQA，2）
        ("dh", c_size_t),                # 每个头的维度（128）
        ("di", c_size_t),                # MLP 中间维度（8960）
        ("maxseq", c_size_t),            # 最大序列长度（KV-Cache 容量）
        ("voc", c_size_t),               # 词表大小（151936）
        ("epsilon", c_float),            # RMSNorm 的 eps
        ("theta", c_float),              # RoPE 频率基
        ("end_token", c_int64),          # 结束符 token id
    ]


# ---------------------------------------------------------------------------
# C 结构体：ChaosuanQwen2Weights（对应 include/chaosuan/models/qwen2.h）
# 注意：数组字段在 C 里是"指针数组"，ctypes 用 POINTER(chaosuanTensor_t)
# 表示；Python 侧需要自己建一个 (chaosuanTensor_t * nlayer)() 数组填进去。
# ---------------------------------------------------------------------------
class ChaosuanQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", chaosuanTensor_t),      # 词向量表 [voc, hs]
        ("out_embed", chaosuanTensor_t),     # lm_head（权重共享时 = in_embed）
        ("out_norm_w", chaosuanTensor_t),    # 最后一层归一化 [hs]
        ("attn_norm_w", POINTER(chaosuanTensor_t)),  # 每层：注意力前归一化
        ("attn_q_w", POINTER(chaosuanTensor_t)),     # 每层：q 投影权重
        ("attn_q_b", POINTER(chaosuanTensor_t)),     # 每层：q 偏置
        ("attn_k_w", POINTER(chaosuanTensor_t)),     # 每层：k 投影权重
        ("attn_k_b", POINTER(chaosuanTensor_t)),     # 每层：k 偏置
        ("attn_v_w", POINTER(chaosuanTensor_t)),     # 每层：v 投影权重
        ("attn_v_b", POINTER(chaosuanTensor_t)),     # 每层：v 偏置
        ("attn_o_w", POINTER(chaosuanTensor_t)),     # 每层：o 投影权重（无 bias）
        ("mlp_norm_w", POINTER(chaosuanTensor_t)),   # 每层：MLP 前归一化
        ("mlp_gate_w", POINTER(chaosuanTensor_t)),   # 每层：SwiGLU gate
        ("mlp_up_w", POINTER(chaosuanTensor_t)),     # 每层：SwiGLU up
        ("mlp_down_w", POINTER(chaosuanTensor_t)),   # 每层：SwiGLU down
    ]


# 模型句柄：不透明指针（Python 侧只存不用，所有操作交给 C++）。
chaosuanQwen2Model_t = c_void_p


def load_qwen2(lib):
    """给共享库设置 4 个 Qwen2 函数的参数类型和返回类型（ctypes 必须的）。"""
    # 创建模型：meta 指针、设备类型、设备 id 数组指针、设备个数
    lib.chaosuanQwen2ModelCreate.argtypes = [
        POINTER(ChaosuanQwen2Meta),
        chaosuanDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.chaosuanQwen2ModelCreate.restype = chaosuanQwen2Model_t

    # 销毁模型
    lib.chaosuanQwen2ModelDestroy.argtypes = [chaosuanQwen2Model_t]
    lib.chaosuanQwen2ModelDestroy.restype = None

    # 取权重结构指针（Python 往里面填张量）
    lib.chaosuanQwen2ModelWeights.argtypes = [chaosuanQwen2Model_t]
    lib.chaosuanQwen2ModelWeights.restype = POINTER(ChaosuanQwen2Weights)

    # 推理：token 数组指针、token 个数 → 下一个 token
    lib.chaosuanQwen2ModelInfer.argtypes = [
        chaosuanQwen2Model_t,
        POINTER(c_int64),
        c_size_t,
    ]
    lib.chaosuanQwen2ModelInfer.restype = c_int64
