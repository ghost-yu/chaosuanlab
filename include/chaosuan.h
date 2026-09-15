#ifndef __CHAOSUAN_H__
#define __CHAOSUAN_H__

#if defined(_WIN32)
#define __export __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define __export __attribute__((visibility("default")))
#else
#define __export
#endif

#ifdef __cplusplus
#define __C extern "C"
#include <cstddef>
#include <cstdint>
#else
#define __C
#include <stddef.h>
#include <stdint.h>
#endif

// Device Types
typedef enum {
    CHAOSUAN_DEVICE_CPU = 0,
    //// TODO: Add more device types here. Numbers need to be consecutive.
    CHAOSUAN_DEVICE_NVIDIA = 1,
    CHAOSUAN_DEVICE_TYPE_COUNT
} chaosuanDeviceType_t;

// Data Types
typedef enum {
    CHAOSUAN_DTYPE_INVALID = 0,
    CHAOSUAN_DTYPE_BYTE = 1,
    CHAOSUAN_DTYPE_BOOL = 2,
    CHAOSUAN_DTYPE_I8 = 3,
    CHAOSUAN_DTYPE_I16 = 4,
    CHAOSUAN_DTYPE_I32 = 5,
    CHAOSUAN_DTYPE_I64 = 6,
    CHAOSUAN_DTYPE_U8 = 7,
    CHAOSUAN_DTYPE_U16 = 8,
    CHAOSUAN_DTYPE_U32 = 9,
    CHAOSUAN_DTYPE_U64 = 10,
    CHAOSUAN_DTYPE_F8 = 11,
    CHAOSUAN_DTYPE_F16 = 12,
    CHAOSUAN_DTYPE_F32 = 13,
    CHAOSUAN_DTYPE_F64 = 14,
    CHAOSUAN_DTYPE_C16 = 15,
    CHAOSUAN_DTYPE_C32 = 16,
    CHAOSUAN_DTYPE_C64 = 17,
    CHAOSUAN_DTYPE_C128 = 18,
    CHAOSUAN_DTYPE_BF16 = 19,
} chaosuanDataType_t;

// Runtime Types
// Stream
typedef void *chaosuanStream_t;

// Memory Copy Directions
typedef enum {
    CHAOSUAN_MEMCPY_H2H = 0,
    CHAOSUAN_MEMCPY_H2D = 1,
    CHAOSUAN_MEMCPY_D2H = 2,
    CHAOSUAN_MEMCPY_D2D = 3,
} chaosuanMemcpyKind_t;

#endif // __CHAOSUAN_H__
