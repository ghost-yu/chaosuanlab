#pragma once

#include "../device_resource.hpp"

namespace chaosuan::device::nvidia {
class Resource : public chaosuan::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();
};
} // namespace chaosuan::device::nvidia
