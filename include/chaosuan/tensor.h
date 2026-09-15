#ifndef CHAOSUAN_TENSOR_H
#define CHAOSUAN_TENSOR_H

#include "../chaosuan.h"

__C {
    typedef struct ChaosuanTensor *chaosuanTensor_t;

    __export chaosuanTensor_t tensorCreate(
        size_t * shape,
        size_t ndim,
        chaosuanDataType_t dtype,
        chaosuanDeviceType_t device_type,
        int device_id);

    __export void tensorDestroy(
        chaosuanTensor_t tensor);

    __export void *tensorGetData(
        chaosuanTensor_t tensor);

    __export size_t tensorGetNdim(
        chaosuanTensor_t tensor);

    __export void tensorGetShape(
        chaosuanTensor_t tensor,
        size_t * shape);

    __export void tensorGetStrides(
        chaosuanTensor_t tensor,
        ptrdiff_t * strides);

    __export chaosuanDataType_t tensorGetDataType(
        chaosuanTensor_t tensor);

    __export chaosuanDeviceType_t tensorGetDeviceType(
        chaosuanTensor_t tensor);

    __export int tensorGetDeviceId(
        chaosuanTensor_t tensor);

    __export void tensorDebug(
        chaosuanTensor_t tensor);

    __export uint8_t tensorIsContiguous(
        chaosuanTensor_t tensor);

    __export void tensorLoad(
        chaosuanTensor_t tensor,
        const void *data);

    __export chaosuanTensor_t tensorView(
        chaosuanTensor_t tensor,
        size_t * shape,
        size_t ndim);

    __export chaosuanTensor_t tensorPermute(
        chaosuanTensor_t tensor,
        size_t * order);

    __export chaosuanTensor_t tensorSlice(
        chaosuanTensor_t tensor,
        size_t dim,
        size_t start,
        size_t end);
}

#endif // CHAOSUAN_TENSOR_H
