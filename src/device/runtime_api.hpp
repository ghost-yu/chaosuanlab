#pragma once
#include "chaosuan/runtime.h"

#include "../utils.hpp"

namespace chaosuan::device {
const ChaosuanRuntimeAPI *getRuntimeAPI(chaosuanDeviceType_t device_type);

const ChaosuanRuntimeAPI *getUnsupportedRuntimeAPI();

namespace cpu {
const ChaosuanRuntimeAPI *getRuntimeAPI();
}

#ifdef ENABLE_NVIDIA_API
namespace nvidia {
const ChaosuanRuntimeAPI *getRuntimeAPI();
}
#endif
} // namespace chaosuan::device
