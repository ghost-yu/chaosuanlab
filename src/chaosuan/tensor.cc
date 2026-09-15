#include "chaosuan_tensor.hpp"

#include <vector>

__C {
    chaosuanTensor_t tensorCreate(
        size_t * shape,
        size_t ndim,
        chaosuanDataType_t dtype,
        chaosuanDeviceType_t device_type,
        int device_id) {
        std::vector<size_t> shape_vec(shape, shape + ndim);
        return new ChaosuanTensor{chaosuan::Tensor::create(shape_vec, dtype, device_type, device_id)};
    }

    void tensorDestroy(
        chaosuanTensor_t tensor) {
        delete tensor;
    }

    void *tensorGetData(
        chaosuanTensor_t tensor) {
        return tensor->tensor->data();
    }

    size_t tensorGetNdim(
        chaosuanTensor_t tensor) {
        return tensor->tensor->ndim();
    }

    void tensorGetShape(
        chaosuanTensor_t tensor,
        size_t * shape) {
        std::copy(tensor->tensor->shape().begin(), tensor->tensor->shape().end(), shape);
    }

    void tensorGetStrides(
        chaosuanTensor_t tensor,
        ptrdiff_t * strides) {
        std::copy(tensor->tensor->strides().begin(), tensor->tensor->strides().end(), strides);
    }

    chaosuanDataType_t tensorGetDataType(
        chaosuanTensor_t tensor) {
        return tensor->tensor->dtype();
    }

    chaosuanDeviceType_t tensorGetDeviceType(
        chaosuanTensor_t tensor) {
        return tensor->tensor->deviceType();
    }

    int tensorGetDeviceId(
        chaosuanTensor_t tensor) {
        return tensor->tensor->deviceId();
    }

    void tensorDebug(
        chaosuanTensor_t tensor) {
        tensor->tensor->debug();
    }

    uint8_t tensorIsContiguous(
        chaosuanTensor_t tensor) {
        return uint8_t(tensor->tensor->isContiguous());
    }

    void tensorLoad(
        chaosuanTensor_t tensor,
        const void *data) {
        tensor->tensor->load(data);
    }

    chaosuanTensor_t tensorView(
        chaosuanTensor_t tensor,
        size_t * shape,
        size_t ndim) {
        std::vector<size_t> shape_vec(shape, shape + ndim);
        return new ChaosuanTensor{tensor->tensor->view(shape_vec)};
    }

    chaosuanTensor_t tensorPermute(
        chaosuanTensor_t tensor,
        size_t * order) {
        std::vector<size_t> order_vec(order, order + tensor->tensor->ndim());
        return new ChaosuanTensor{tensor->tensor->permute(order_vec)};
    }

    chaosuanTensor_t tensorSlice(
        chaosuanTensor_t tensor,
        size_t dim,
        size_t start,
        size_t end) {
        return new ChaosuanTensor{tensor->tensor->slice(dim, start, end)};
    }
}
