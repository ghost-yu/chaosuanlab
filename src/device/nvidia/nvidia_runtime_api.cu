#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>

namespace chaosuan::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    TO_BE_IMPLEMENTED();
}

void setDevice(int) {
    TO_BE_IMPLEMENTED();
}

void deviceSynchronize() {
    TO_BE_IMPLEMENTED();
}

chaosuanStream_t createStream() {
    TO_BE_IMPLEMENTED();
}

void destroyStream(chaosuanStream_t stream) {
    TO_BE_IMPLEMENTED();
}
void streamSynchronize(chaosuanStream_t stream) {
    TO_BE_IMPLEMENTED();
}

void *mallocDevice(size_t size) {
    TO_BE_IMPLEMENTED();
}

void freeDevice(void *ptr) {
    TO_BE_IMPLEMENTED();
}

void *mallocHost(size_t size) {
    TO_BE_IMPLEMENTED();
}

void freeHost(void *ptr) {
    TO_BE_IMPLEMENTED();
}

void memcpySync(void *dst, const void *src, size_t size, chaosuanMemcpyKind_t kind) {
    TO_BE_IMPLEMENTED();
}

void memcpyAsync(void *dst, const void *src, size_t size, chaosuanMemcpyKind_t kind) {
    TO_BE_IMPLEMENTED();
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
} // namespace chaosuan::device::nvidia
