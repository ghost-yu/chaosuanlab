#pragma once
#include "chaosuan/tensor.h"

#include "../tensor/tensor.hpp"

__C {
    typedef struct ChaosuanTensor {
        chaosuan::tensor_t tensor;
    } ChaosuanTensor;
}
