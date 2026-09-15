#pragma once

#include "../device_resource.hpp"

namespace chaosuan::device::cpu {
class Resource : public chaosuan::device::DeviceResource {
public:
    Resource();
    ~Resource() = default;
};
} // namespace chaosuan::device::cpu