#pragma once

#include "../../tensor/tensor.hpp"

namespace chaosuan::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
}
