#pragma once

#include "../../tensor/tensor.hpp"

namespace chaosuan::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up);
}
