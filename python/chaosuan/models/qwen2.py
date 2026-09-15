# ============================================================================
# qwen2.py —— Qwen2 模型的高层 Python 封装（作业 3）
# ----------------------------------------------------------------------------
# 这个文件是"胶水层"，做三件事：
#   1. 读模型的 config.json，得到超参数（层数、维度、词表等）；
#   2. 读 safetensors 权重文件，把每个权重转成 CHAOSUAN 的张量，
#      通过 C API（tensorCreate + tensorLoad）送进 C++ 后端；
#   3. 提供 generate()：逐 token 调用 C++ 的 infer，拼出完整输出。
#
# 注意：这里**不允许**用 PyTorch 做模型推理（README 的硬性要求）！
# 用 torch 只做一件事：把 bf16 权重的原始字节取出来（.view(torch.uint16)），
# 真正的计算全部在 C++ 后端用我们自己的算子完成。
#
# 【DeepSeek-R1-Distill-Qwen-1.5B 的架构参数】
#   hidden_size=1536, num_hidden_layers=28, num_attention_heads=12,
#   num_key_value_heads=2 (GQA), head_dim=128, intermediate_size=8960,
#   vocab_size=151936, rms_norm_eps=1e-6, rope_theta=10000.0,
#   tie_word_embeddings=true（lm_head 与 embedding 共享权重）
# ============================================================================

import ctypes
import json
from pathlib import Path
from typing import Sequence

from ..libchaosuan import LIB_CHAOSUAN
from ..libchaosuan import DataType
from ..libchaosuan import DeviceType
from ..libchaosuan.qwen2 import (
    ChaosuanQwen2Meta,
    ChaosuanQwen2Weights,
    chaosuanQwen2Model_t,
)

try:
    import torch
    from safetensors import safe_open
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "qwen2 model loading needs torch & safetensors "
        "(pip3 install torch safetensors)"
    ) from e


class Qwen2:
    """用 CHAOSUAN 后端加载并运行 DeepSeek-R1-Distill-Qwen-1.5B。"""

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU, device_id: int = 0):
        # 作业 3 只支持 CPU；作业 4 起支持 NVIDIA（RTX 4090）。
        # device_id：多卡时的 GPU 编号，本项目只适配单卡（4090），恒为 0。
        self._device = device
        self._device_id = device_id

        model_path = Path(model_path)

        # ------------------------------------------------------------------
        # 1. 读取 config.json 并组装超参数结构 ChaosuanQwen2Meta
        # ------------------------------------------------------------------
        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self._meta = ChaosuanQwen2Meta(
            # 关键精度决策：权重按 F32 加载！
            # HF 的 bf16 模型里，RoPE 之后 q/k 与 f32 的 cos/sin 相乘会
            # 类型提升成 f32，之后整个前向都在 f32 上做（只有权重是 bf16）。
            # 我们统一用 f32 计算 + f32 权重（bf16→f32 无损）与它几乎一致，
            # 比"全程 bf16"更贴近 HF 的数值结果（保证 --test 逐 token 一致）。
            dtype=DataType.F32,
            nlayer=cfg["num_hidden_layers"],
            hs=cfg["hidden_size"],
            nh=cfg["num_attention_heads"],
            nkvh=cfg["num_key_value_heads"],
            dh=cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
            di=cfg["intermediate_size"],
            maxseq=cfg.get("max_position_embeddings", 32768),
            voc=cfg["vocab_size"],
            epsilon=cfg.get("rms_norm_eps", 1e-6),
            theta=cfg.get("rope_theta", 10000.0),
            end_token=cfg.get("eos_token_id", 151645),
        )
        self._nlayer = self._meta.nlayer

        # ------------------------------------------------------------------
        # 2. 读 safetensors，把每个权重做成 CHAOSUAN 张量（bf16 字节拷贝）
        # ------------------------------------------------------------------
        # tensors[name] = torch 张量（bf16，CPU 上）
        tensors = {}
        for file in sorted(model_path.glob("*.safetensors")):
            with safe_open(str(file), framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensors[name] = f.get_tensor(name)  # torch.bf16 tensor

        # 小工具：把 torch bf16 张量转成 CHAOSUAN 张量。
        # 所有权：创建出来的 chaosuanTensor_t 之后由 C++ 模型统一释放。
        def make_tensor(arr):
            # arr: [d0, d1, ...] bf16 → 转成 (c_size_t*ndim) 的 shape 数组
            ndim = arr.dim()
            shape_arr = (ctypes.c_size_t * ndim)(*arr.shape)
            t = LIB_CHAOSUAN.tensorCreate(
                shape_arr,
                ndim,
                DataType.F32,    # 权重统一存成 f32（bf16→f32 无损，计算精度对齐 HF）
                self._device,    # 权重张量直接建在目标设备（CPU 或 NVIDIA）上
                self._device_id,
            )
            # 取 f32 字节：bf16 → f32 是精确转换（bf16 就是 f32 截断的尾数），
            # 用 torch 的 .float() 完成，再按 4 字节读出。
            raw = arr.contiguous().float().numpy().tobytes()
            buf = ctypes.create_string_buffer(raw)
            # tensorLoad 内部是 memcpy H2D：host 权重数据 → 设备张量（GPU 时自动走 cudaMemcpy）。
            LIB_CHAOSUAN.tensorLoad(t, buf)
            return t

        # 所有权重张量列表（Python 侧不再释放，交给 C++）
        self._created = []

        def named(name):
            t = make_tensor(tensors[name])
            self._created.append(t)
            return t

        # 每层一个指针数组（ctypes 数组，长度 = nlayer）
        def layer_array(fn):
            arr = (ctypes.c_void_p * self._nlayer)()
            for i in range(self._nlayer):
                arr[i] = fn(i)
            return arr

        # ---- 组装 ChaosuanQwen2Weights 结构 ----
        self._weights = ChaosuanQwen2Weights()

        in_embed = named("model.embed_tokens.weight")   # [voc, hs]（查表用）
        # 【重要坑 8】config 里 tie_word_embeddings=true，但这个模型的
        # safetensors 里其实**独立保存了 lm_head.weight**，且它的值与
        # embed_tokens.weight 并不完全相同（逐元素相同的只有 ~0.7%）！
        # HF 加载时 lm_head.weight 在文件里存在 → 直接加载、不执行 tie 覆盖，
        # 所以 logits 必须用 lm_head.weight 计算。如果像"教科书"那样简单
        # tied（out_embed = in_embed），logits 会整体偏移，argmax 翻转。
        self._weights.in_embed = in_embed
        self._weights.out_embed = named("lm_head.weight")  # [voc, hs]（算 logits 用）
        self._weights.out_norm_w = named("model.norm.weight")  # [hs]

        self._weights.attn_norm_w = layer_array(
            lambda i: named(f"model.layers.{i}.input_layernorm.weight"))
        self._weights.attn_q_w = layer_array(
            lambda i: named(f"model.layers.{i}.self_attn.q_proj.weight"))
        self._weights.attn_q_b = layer_array(
            lambda i: named(f"model.layers.{i}.self_attn.q_proj.bias"))
        self._weights.attn_k_w = layer_array(
            lambda i: named(f"model.layers.{i}.self_attn.k_proj.weight"))
        self._weights.attn_k_b = layer_array(
            lambda i: named(f"model.layers.{i}.self_attn.k_proj.bias"))
        self._weights.attn_v_w = layer_array(
            lambda i: named(f"model.layers.{i}.self_attn.v_proj.weight"))
        self._weights.attn_v_b = layer_array(
            lambda i: named(f"model.layers.{i}.self_attn.v_proj.bias"))
        self._weights.attn_o_w = layer_array(
            lambda i: named(f"model.layers.{i}.self_attn.o_proj.weight"))
        self._weights.mlp_norm_w = layer_array(
            lambda i: named(f"model.layers.{i}.post_attention_layernorm.weight"))
        self._weights.mlp_gate_w = layer_array(
            lambda i: named(f"model.layers.{i}.mlp.gate_proj.weight"))
        self._weights.mlp_up_w = layer_array(
            lambda i: named(f"model.layers.{i}.mlp.up_proj.weight"))
        self._weights.mlp_down_w = layer_array(
            lambda i: named(f"model.layers.{i}.mlp.down_proj.weight"))

        # ------------------------------------------------------------------
        # 3. 创建 C++ 模型对象并填入权重
        # ------------------------------------------------------------------
        # 先 Create（拿到模型句柄 + 内部 Weights 结构），再往结构里填张量，
        # C++ 会在第一次 infer 时统一"收编"这些权重。
        self._model = LIB_CHAOSUAN.chaosuanQwen2ModelCreate(
            ctypes.byref(self._meta),
            self._device,       # CPU 或 NVIDIA
            None,               # 设备 id 数组（单卡作业不涉及多设备分发）
            0,
        )
        w_ptr = LIB_CHAOSUAN.chaosuanQwen2ModelWeights(self._model)
        ctypes.memmove(w_ptr, ctypes.byref(self._weights), ctypes.sizeof(self._weights))

    # ========================================================================
    # generate —— 文本生成（自回归）
    # ------------------------------------------------------------------------
    # 与 test_infer.py 的约定一致：
    #   - 返回"完整 token 序列"（prompt + 新生成的 token）；
    #   - 生成到 end_token 或达到 max_new_tokens 停止。
    # 测试模式（--test）下 top_k=1 → 纯贪心（argmax），必须与 HF 完全一致。
    # ========================================================================
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128

        # ---- 第一次调用：prefill 整个 prompt（C++ 内部把 k/v 写入缓存）----
        next_id = self._infer(list(inputs))

        generated = []
        while True:
            # 【与 transformers 行为一致】先把 token 收进结果，再判断结束。
            # HF 的 generate() 返回的序列**包含** eos token（结束符），
            # 如果先判 end 再 append，就会比 HF 少最后一个 token，
            # 导致 test_infer.py 的逐 token 断言失败。
            generated.append(next_id)
            if next_id == self._meta.end_token:   # 遇到结束符，停止（结束符已计入）
                break
            if len(generated) >= max_new_tokens:  # 达到长度上限，停止
                break
            # 后续每次只喂"上一个新生成的 token"（decode，复用 KV-Cache）
            next_id = self._infer([next_id])

        return list(inputs) + generated

    # ------------------------------------------------------------------------
    # _infer —— 调用 C++ 后端做一次前向，返回 argmax 的下一个 token
    # ------------------------------------------------------------------------
    def _infer(self, token_ids):
        n = len(token_ids)
        arr = (ctypes.c_int64 * n)(*token_ids)
        return LIB_CHAOSUAN.chaosuanQwen2ModelInfer(self._model, arr, n)

    # ------------------------------------------------------------------------
    # __del__ —— 对象销毁时释放 C++ 模型（所有权重 + KV-Cache）
    # ------------------------------------------------------------------------
    def __del__(self):
        if getattr(self, "_model", None):
            LIB_CHAOSUAN.chaosuanQwen2ModelDestroy(self._model)
            self._model = None
