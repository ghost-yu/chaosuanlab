#pragma once

#include "../../tensor/tensor.hpp"

namespace chaosuan::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight);
}
