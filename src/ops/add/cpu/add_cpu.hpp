#pragma once
#include "chaosuan.h"

#include <cstddef>

namespace chaosuan::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, chaosuanDataType_t type, size_t size);
}