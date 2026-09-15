from typing import Sequence, Tuple

from .libchaosuan import (
    LIB_CHAOSUAN,
    chaosuanTensor_t,
    chaosuanDeviceType_t,
    DeviceType,
    chaosuanDataType_t,
    DataType,
)
from ctypes import c_size_t, c_int, c_ssize_t, c_void_p


class Tensor:
    def __init__(
        self,
        shape: Sequence[int] = None,
        dtype: DataType = DataType.F32,
        device: DeviceType = DeviceType.CPU,
        device_id: int = 0,
        tensor: chaosuanTensor_t = None,
    ):
        if tensor:
            self._tensor = tensor
        else:
            _ndim = 0 if shape is None else len(shape)
            _shape = None if shape is None else (c_size_t * len(shape))(*shape)
            self._tensor: chaosuanTensor_t = LIB_CHAOSUAN.tensorCreate(
                _shape,
                c_size_t(_ndim),
                chaosuanDataType_t(dtype),
                chaosuanDeviceType_t(device),
                c_int(device_id),
            )

    def __del__(self):
        if hasattr(self, "_tensor") and self._tensor is not None:
            LIB_CHAOSUAN.tensorDestroy(self._tensor)
            self._tensor = None

    def shape(self) -> Tuple[int]:
        buf = (c_size_t * self.ndim())()
        LIB_CHAOSUAN.tensorGetShape(self._tensor, buf)
        return tuple(buf[i] for i in range(self.ndim()))

    def strides(self) -> Tuple[int]:
        buf = (c_ssize_t * self.ndim())()
        LIB_CHAOSUAN.tensorGetStrides(self._tensor, buf)
        return tuple(buf[i] for i in range(self.ndim()))

    def ndim(self) -> int:
        return int(LIB_CHAOSUAN.tensorGetNdim(self._tensor))

    def dtype(self) -> DataType:
        return DataType(LIB_CHAOSUAN.tensorGetDataType(self._tensor))

    def device_type(self) -> DeviceType:
        return DeviceType(LIB_CHAOSUAN.tensorGetDeviceType(self._tensor))

    def device_id(self) -> int:
        return int(LIB_CHAOSUAN.tensorGetDeviceId(self._tensor))

    def data_ptr(self) -> c_void_p:
        return LIB_CHAOSUAN.tensorGetData(self._tensor)

    def lib_tensor(self) -> chaosuanTensor_t:
        return self._tensor

    def debug(self):
        LIB_CHAOSUAN.tensorDebug(self._tensor)

    def __repr__(self):
        return f"<Tensor shape={self.shape}, dtype={self.dtype}, device={self.device_type}:{self.device_id}>"

    def load(self, data: c_void_p):
        LIB_CHAOSUAN.tensorLoad(self._tensor, data)

    def is_contiguous(self) -> bool:
        return bool(LIB_CHAOSUAN.tensorIsContiguous(self._tensor))

    def view(self, *shape: int) -> chaosuanTensor_t:
        _shape = (c_size_t * len(shape))(*shape)
        return Tensor(
            tensor=LIB_CHAOSUAN.tensorView(self._tensor, _shape, c_size_t(len(shape)))
        )

    def permute(self, *perm: int) -> chaosuanTensor_t:
        assert len(perm) == self.ndim()
        _perm = (c_size_t * len(perm))(*perm)
        return Tensor(tensor=LIB_CHAOSUAN.tensorPermute(self._tensor, _perm))

    def slice(self, dim: int, start: int, end: int):
        return Tensor(
            tensor=LIB_CHAOSUAN.tensorSlice(
                self._tensor, c_size_t(dim), c_size_t(start), c_size_t(end)
            )
        )
