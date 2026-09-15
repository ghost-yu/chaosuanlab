#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>

namespace chaosuan::device::cpu {

namespace runtime_api {
int getDeviceCount() {
    return 1;
}

void setDevice(int) {
    // do nothing
}

void deviceSynchronize() {
    // do nothing
}

chaosuanStream_t createStream() {
    return (chaosuanStream_t)0; // null stream
}

void destroyStream(chaosuanStream_t stream) {
    // do nothing
}
void streamSynchronize(chaosuanStream_t stream) {
    // do nothing
}

void *mallocDevice(size_t size) {
    return std::malloc(size);
}

void freeDevice(void *ptr) {
    std::free(ptr);
}

void *mallocHost(size_t size) {
    return mallocDevice(size);
}

void freeHost(void *ptr) {
    freeDevice(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, chaosuanMemcpyKind_t kind) {
    std::memcpy(dst, src, size);
}

void memcpyAsync(void *dst, const void *src, size_t size, chaosuanMemcpyKind_t kind, chaosuanStream_t stream) {
    memcpySync(dst, src, size, kind);
}

static const ChaosuanRuntimeAPI RUNTIME_API = {
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

} // namespace runtime_api

const ChaosuanRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace chaosuan::device::cpu
