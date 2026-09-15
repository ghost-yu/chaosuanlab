#pragma once
#include "chaosuan.h"

#include "../utils.hpp"

namespace chaosuan::device {
class DeviceResource {
private:
    chaosuanDeviceType_t _device_type;
    int _device_id;

public:
    DeviceResource(chaosuanDeviceType_t device_type, int device_id)
        : _device_type(device_type),
          _device_id(device_id) {
    }
    ~DeviceResource() = default;

    chaosuanDeviceType_t getDeviceType() const { return _device_type; }
    int getDeviceId() const { return _device_id; };
};
} // namespace chaosuan::device
