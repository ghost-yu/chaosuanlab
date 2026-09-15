#pragma once
#include "chaosuan.h"

#include "../core.hpp"

#include <memory>

namespace chaosuan::core {
class Storage {
private:
    std::byte *_memory;
    size_t _size;
    Runtime &_runtime;
    bool _is_host;
    Storage(std::byte *memory, size_t size, Runtime &runtime, bool is_host);

public:
    friend class Runtime;
    ~Storage();

    std::byte *memory() const;
    size_t size() const;
    chaosuanDeviceType_t deviceType() const;
    int deviceId() const;
    bool isHost() const;
};

}; // namespace chaosuan::core
