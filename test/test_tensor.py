import chaosuan

import torch
from test_utils import *
import argparse


def test_tensor():
    torch_tensor = torch.arange(60, dtype=torch_dtype("i64")).reshape(3, 4, 5)
    chaosuan_tensor = chaosuan.Tensor(
        (3, 4, 5), dtype=chaosuan_dtype("i64"), device=chaosuan_device("cpu")
    )

    # Test load
    print("===Test load===")
    chaosuan_tensor.load(torch_tensor.data_ptr())
    chaosuan_tensor.debug()
    assert chaosuan_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(chaosuan_tensor, torch_tensor)

    # Test view
    print("===Test view===")
    torch_tensor_view = torch_tensor.view(6, 10)
    chaosuan_tensor_view = chaosuan_tensor.view(6, 10)
    chaosuan_tensor_view.debug()
    assert chaosuan_tensor_view.shape() == torch_tensor_view.shape
    assert chaosuan_tensor_view.strides() == torch_tensor_view.stride()
    assert chaosuan_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(chaosuan_tensor_view, torch_tensor_view)

    # Test permute
    print("===Test permute===")
    torch_tensor_perm = torch_tensor.permute(2, 0, 1)
    chaosuan_tensor_perm = chaosuan_tensor.permute(2, 0, 1)
    chaosuan_tensor_perm.debug()
    assert chaosuan_tensor_perm.shape() == torch_tensor_perm.shape
    assert chaosuan_tensor_perm.strides() == torch_tensor_perm.stride()
    assert chaosuan_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(chaosuan_tensor_perm, torch_tensor_perm)

    # Test slice
    print("===Test slice===")
    torch_tensor_slice = torch_tensor[:, :, 1:4]
    chaosuan_tensor_slice = chaosuan_tensor.slice(2, 1, 4)
    chaosuan_tensor_slice.debug()
    assert chaosuan_tensor_slice.shape() == torch_tensor_slice.shape
    assert chaosuan_tensor_slice.strides() == torch_tensor_slice.stride()
    assert chaosuan_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(chaosuan_tensor_slice, torch_tensor_slice)


if __name__ == "__main__":
    test_tensor()

    print("\n\033[92mTest passed!\033[0m\n")
