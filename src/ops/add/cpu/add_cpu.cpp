#include "add_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void add_(T *c, const T *a, const T *b, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, chaosuan::bf16_t> || std::is_same_v<T, chaosuan::fp16_t>) {
            c[i] = chaosuan::utils::cast<T>(chaosuan::utils::cast<float>(a[i]) + chaosuan::utils::cast<float>(b[i]));
        } else {
            c[i] = a[i] + b[i];
        }
    }
}

namespace chaosuan::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, chaosuanDataType_t type, size_t numel) {
    switch (type) {
    case CHAOSUAN_DTYPE_F32:
        return add_(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), numel);
    case CHAOSUAN_DTYPE_BF16:
        return add_(reinterpret_cast<chaosuan::bf16_t *>(c), reinterpret_cast<const chaosuan::bf16_t *>(a),
                    reinterpret_cast<const chaosuan::bf16_t *>(b), numel);
    case CHAOSUAN_DTYPE_F16:
        return add_(reinterpret_cast<chaosuan::fp16_t *>(c), reinterpret_cast<const chaosuan::fp16_t *>(a),
                    reinterpret_cast<const chaosuan::fp16_t *>(b), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace chaosuan::ops::cpu
