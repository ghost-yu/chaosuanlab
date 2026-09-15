from . import libchaosuan
from .libchaosuan import LIB_CHAOSUAN
from ctypes import c_void_p


class RuntimeAPI:
    def __init__(self, device_type: libchaosuan.DeviceType):
        self._api = LIB_CHAOSUAN.chaosuanGetRuntimeAPI(
            libchaosuan.chaosuanDeviceType_t(device_type)
        )

    def get_device_count(self) -> int:
        result = self._api.contents.get_device_count()
        return result

    def set_device(self, device_id: int) -> None:
        self._api.contents.set_device(device_id)

    def device_synchronize(self) -> None:
        self._api.contents.device_synchronize()

    def create_stream(self) -> libchaosuan.chaosuanStream_t:
        stream = self._api.contents.create_stream()
        return stream

    def destroy_stream(self, stream: libchaosuan.chaosuanStream_t) -> None:
        self._api.contents.destroy_stream(stream)

    def stream_synchronize(self, stream: libchaosuan.chaosuanStream_t) -> None:
        self._api.contents.stream_synchronize(stream)

    def malloc_device(self, size: int) -> c_void_p:
        ptr = self._api.contents.malloc_device(size)
        return ptr

    def free_device(self, ptr: c_void_p) -> None:
        print(f"[chaosuan] free_device({ptr})")
        self._api.contents.free_device(ptr)

    def malloc_host(self, size: int) -> c_void_p:
        ptr = self._api.contents.malloc_host(size)
        return ptr

    def free_host(self, ptr: c_void_p) -> None:
        self._api.contents.free_host(ptr)

    def memcpy_sync(
        self,
        dst: c_void_p,
        src: c_void_p,
        size: int,
        kind: libchaosuan.MemcpyKind,
    ) -> None:
        self._api.contents.memcpy_sync(
            dst, src, size, libchaosuan.chaosuanMemcpyKind_t(kind)
        )

    def memcpy_async(
        self,
        dst: c_void_p,
        src: c_void_p,
        size: int,
        kind: libchaosuan.MemcpyKind,
        stream: libchaosuan.chaosuanStream_t,
    ) -> None:
        self._api.contents.memcpy_async(
            dst, src, size, libchaosuan.chaosuanMemcpyKind_t(kind), stream
        )
