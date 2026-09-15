#include "nvidia_resource.cuh"

namespace chaosuan::device::nvidia {

Resource::Resource(int device_id) : chaosuan::device::DeviceResource(CHAOSUAN_DEVICE_NVIDIA, device_id) {}

} // namespace chaosuan::device::nvidia
