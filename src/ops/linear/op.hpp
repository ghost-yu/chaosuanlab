#pragma once

#include "../../tensor/tensor.hpp"

namespace chaosuan::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
}
