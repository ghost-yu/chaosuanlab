#ifndef CHAOSUAN_RUNTIME_H
#define CHAOSUAN_RUNTIME_H

#include "../chaosuan.h"

__C {
    // Runtime API Functions
    // Device
    typedef int (*get_device_count_api)();
    typedef void (*set_device_api)(int);
    typedef void (*device_synchronize_api)();
    // Stream
    typedef chaosuanStream_t (*create_stream_api)();
    typedef void (*destroy_stream_api)(chaosuanStream_t);
    typedef void (*stream_synchronize_api)(chaosuanStream_t);
    // Memory
    typedef void *(*malloc_device_api)(size_t);
    typedef void (*free_device_api)(void *);
    typedef void *(*malloc_host_api)(size_t);
    typedef void (*free_host_api)(void *);
    // Memory copy
    typedef void (*memcpy_sync_api)(void *, const void *, size_t, chaosuanMemcpyKind_t);
    typedef void (*memcpy_async_api)(void *, const void *, size_t, chaosuanMemcpyKind_t, chaosuanStream_t);

    struct ChaosuanRuntimeAPI {
        get_device_count_api get_device_count;
        set_device_api set_device;
        device_synchronize_api device_synchronize;
        create_stream_api create_stream;
        destroy_stream_api destroy_stream;
        stream_synchronize_api stream_synchronize;
        malloc_device_api malloc_device;
        free_device_api free_device;
        malloc_host_api malloc_host;
        free_host_api free_host;
        memcpy_sync_api memcpy_sync;
        memcpy_async_api memcpy_async;
    };

    // Chaosuan API for getting the runtime APIs
    __export const ChaosuanRuntimeAPI *chaosuanGetRuntimeAPI(chaosuanDeviceType_t);

    // Chaosuan API for switching device context
    __export void chaosuanSetContextRuntime(chaosuanDeviceType_t, int);
}

#endif // CHAOSUAN_RUNTIME_H
