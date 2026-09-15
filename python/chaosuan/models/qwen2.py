from typing import Sequence
from ..libchaosuan import LIB_CHAOSUAN
from ..libchaosuan import DeviceType

from pathlib import Path
import safetensors


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # TODO: Implement model constructor

        model_path = Path(model_path)

        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name_ in data_.keys():
                ## TODO: load the model weights
                pass

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        # TODO: Implement generate function

        return []
