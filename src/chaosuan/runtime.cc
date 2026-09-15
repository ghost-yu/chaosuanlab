#include "chaosuan/runtime.h"
#include "../core/context/context.hpp"
#include "../device/runtime_api.hpp"

// Chaosuan API for setting context runtime.
__C void chaosuanSetContextRuntime(chaosuanDeviceType_t device_type, int device_id) {
    chaosuan::core::context().setDevice(device_type, device_id);
}

// Chaosuan API for getting the runtime APIs
__C const ChaosuanRuntimeAPI *chaosuanGetRuntimeAPI(chaosuanDeviceType_t device_type) {
    return chaosuan::device::getRuntimeAPI(device_type);
}