#pragma once

#include "chaosuan/runtime.h"

#include "../storage/storage.hpp"

namespace chaosuan::core {
class MemoryAllocator {
protected:
    const ChaosuanRuntimeAPI *_api;
    MemoryAllocator(const ChaosuanRuntimeAPI *runtime_api) : _api(runtime_api){};

public:
    virtual ~MemoryAllocator() = default;
    virtual std::byte *allocate(size_t size) = 0;
    virtual void release(std::byte *memory) = 0;
};

} // namespace chaosuan::core
