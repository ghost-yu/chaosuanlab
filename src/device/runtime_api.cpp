#include "runtime_api.hpp"

namespace chaosuan::device {

int getDeviceCount() {
    return 0;
}

void setDevice(int) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void deviceSynchronize() {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

chaosuanStream_t createStream() {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return nullptr;
}

void destroyStream(chaosuanStream_t stream) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}
void streamSynchronize(chaosuanStream_t stream) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void *mallocDevice(size_t size) {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return nullptr;
}

void freeDevice(void *ptr) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void *mallocHost(size_t size) {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return nullptr;
}

void freeHost(void *ptr) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void memcpySync(void *dst, const void *src, size_t size, chaosuanMemcpyKind_t kind) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void memcpyAsync(void *dst, const void *src, size_t size, chaosuanMemcpyKind_t kind, chaosuanStream_t stream) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

static const ChaosuanRuntimeAPI NOOP_RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

const ChaosuanRuntimeAPI *getUnsupportedRuntimeAPI() {
    return &NOOP_RUNTIME_API;
}

const ChaosuanRuntimeAPI *getRuntimeAPI(chaosuanDeviceType_t device_type) {
    // Implement for all device types
    switch (device_type) {
    case CHAOSUAN_DEVICE_CPU:
        return chaosuan::device::cpu::getRuntimeAPI();
    case CHAOSUAN_DEVICE_NVIDIA:
#ifdef ENABLE_NVIDIA_API
        return chaosuan::device::nvidia::getRuntimeAPI();
#else
        return getUnsupportedRuntimeAPI();
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
        return nullptr;
    }
}
} // namespace chaosuan::device
