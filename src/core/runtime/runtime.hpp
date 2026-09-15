#pragma once
#include "../core.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/allocator.hpp"

namespace chaosuan::core {
class Runtime {
private:
    chaosuanDeviceType_t _device_type;
    int _device_id;
    const ChaosuanRuntimeAPI *_api;
    MemoryAllocator *_allocator;
    bool _is_active;
    void _activate();
    void _deactivate();
    chaosuanStream_t _stream;
    Runtime(chaosuanDeviceType_t device_type, int device_id);

public:
    friend class Context;

    ~Runtime();

    // Prevent copying
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    // Prevent moving
    Runtime(Runtime &&) = delete;
    Runtime &operator=(Runtime &&) = delete;

    chaosuanDeviceType_t deviceType() const;
    int deviceId() const;
    bool isActive() const;

    const ChaosuanRuntimeAPI *api() const;

    storage_t allocateDeviceStorage(size_t size);
    ;
    storage_t allocateHostStorage(size_t size);
    void freeStorage(Storage *storage);

    chaosuanStream_t stream() const;
    void synchronize() const;
};
} // namespace chaosuan::core
