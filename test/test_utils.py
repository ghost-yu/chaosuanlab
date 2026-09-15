import chaosuan
import torch


def random_tensor(
    shape, dtype_name, device_name, device_id=0, scale=None, bias=None
) -> tuple[torch.Tensor, chaosuan.Tensor]:
    torch_tensor = torch.rand(
        shape,
        dtype=torch_dtype(dtype_name),
        device=torch_device(device_name, device_id),
    )
    if scale is not None:
        torch_tensor *= scale
    if bias is not None:
        torch_tensor += bias

    chaosuan_tensor = chaosuan.Tensor(
        shape,
        dtype=chaosuan_dtype(dtype_name),
        device=chaosuan_device(device_name),
        device_id=device_id,
    )

    api = chaosuan.RuntimeAPI(chaosuan_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        chaosuan_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        chaosuan.MemcpyKind.D2D,
    )

    return torch_tensor, chaosuan_tensor


def random_int_tensor(shape, device_name, dtype_name="i64", device_id=0, low=0, high=2):
    torch_tensor = torch.randint(
        low,
        high,
        shape,
        dtype=torch_dtype(dtype_name),
        device=torch_device(device_name, device_id),
    )

    chaosuan_tensor = chaosuan.Tensor(
        shape,
        dtype=chaosuan_dtype(dtype_name),
        device=chaosuan_device(device_name),
        device_id=device_id,
    )

    api = chaosuan.RuntimeAPI(chaosuan_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        chaosuan_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        chaosuan.MemcpyKind.D2D,
    )

    return torch_tensor, chaosuan_tensor


def zero_tensor(
    shape, dtype_name, device_name, device_id=0
) -> tuple[torch.Tensor, chaosuan.Tensor]:
    torch_tensor = torch.zeros(
        shape,
        dtype=torch_dtype(dtype_name),
        device=torch_device(device_name, device_id),
    )

    chaosuan_tensor = chaosuan.Tensor(
        shape,
        dtype=chaosuan_dtype(dtype_name),
        device=chaosuan_device(device_name),
        device_id=device_id,
    )

    api = chaosuan.RuntimeAPI(chaosuan_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        chaosuan_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        chaosuan.MemcpyKind.D2D,
    )

    return torch_tensor, chaosuan_tensor


def arrange_tensor(
    start, end, device_name, device_id=0
) -> tuple[torch.Tensor, chaosuan.Tensor]:
    torch_tensor = torch.arange(start, end, device=torch_device(device_name, device_id))
    chaosuan_tensor = chaosuan.Tensor(
        (end - start,),
        dtype=chaosuan_dtype("i64"),
        device=chaosuan_device(device_name),
        device_id=device_id,
    )

    api = chaosuan.RuntimeAPI(chaosuan_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        chaosuan_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        chaosuan.MemcpyKind.D2D,
    )

    return torch_tensor, chaosuan_tensor


def check_equal(
    chaosuan_result: chaosuan.Tensor,
    torch_answer: torch.Tensor,
    atol=1e-5,
    rtol=1e-5,
    strict=False,
):
    shape = chaosuan_result.shape()
    strides = chaosuan_result.strides()
    assert shape == torch_answer.shape
    assert torch_dtype(dtype_name(chaosuan_result.dtype())) == torch_answer.dtype

    right = 0
    for i in range(len(shape)):
        if strides[i] > 0:
            right += strides[i] * (shape[i] - 1)
        else:  # TODO: Support negative strides in the future
            raise ValueError("Negative strides are not supported yet")

    tmp = torch.zeros(
        (right + 1,),
        dtype=torch_answer.dtype,
        device=torch_device(
            device_name(chaosuan_result.device_type()), chaosuan_result.device_id()
        ),
    )
    result = torch.as_strided(tmp, shape, strides)
    api = chaosuan.RuntimeAPI(chaosuan_result.device_type())
    api.memcpy_sync(
        result.data_ptr(),
        chaosuan_result.data_ptr(),
        (right + 1) * tmp.element_size(),
        chaosuan.MemcpyKind.D2D,
    )

    if strict:
        if torch.equal(result, torch_answer):
            return True
    else:
        if torch.allclose(result, torch_answer, atol=atol, rtol=rtol):
            return True

    print(f"CHAOSUAN result: \n{result}")
    print(f"Torch answer: \n{torch_answer}")
    return False


def benchmark(torch_func, chaosuan_func, device_name, warmup=10, repeat=100):
    api = chaosuan.RuntimeAPI(chaosuan_device(device_name))

    def time_op(func):
        import time

        for _ in range(warmup):
            func()
        api.device_synchronize()
        start = time.time()
        for _ in range(repeat):
            func()
        api.device_synchronize()
        end = time.time()
        return (end - start) / repeat

    torch_time = time_op(torch_func)
    chaosuan_time = time_op(chaosuan_func)
    print(
        f"        Torch time: {torch_time*1000:.5f} ms \n        CHAOSUAN time: {chaosuan_time*1000:.5f} ms"
    )


def torch_device(device_name: str, device_id=0):
    if device_name == "cpu":
        return torch.device("cpu")
    elif device_name == "nvidia":
        return torch.device(f"cuda:{device_id}")
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def chaosuan_device(device_name: str):
    if device_name == "cpu":
        return chaosuan.DeviceType.CPU
    elif device_name == "nvidia":
        return chaosuan.DeviceType.NVIDIA
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def device_name(chaosuan_device: chaosuan.DeviceType):
    if chaosuan_device == chaosuan.DeviceType.CPU:
        return "cpu"
    elif chaosuan_device == chaosuan.DeviceType.NVIDIA:
        return "nvidia"
    else:
        raise ValueError(f"Unsupported chaosuan device: {chaosuan_device}")


def torch_dtype(dtype_name: str):
    if dtype_name == "f16":
        return torch.float16
    elif dtype_name == "f32":
        return torch.float32
    elif dtype_name == "f64":
        return torch.float64
    elif dtype_name == "bf16":
        return torch.bfloat16
    elif dtype_name == "i32":
        return torch.int32
    elif dtype_name == "i64":
        return torch.int64
    elif dtype_name == "u32":
        return torch.uint32
    elif dtype_name == "u64":
        return torch.uint64
    elif dtype_name == "bool":
        return torch.bool
    else:
        raise ValueError(f"Unsupported dtype name: {dtype_name}")


def chaosuan_dtype(dtype_name: str):
    if dtype_name == "f16":
        return chaosuan.DataType.F16
    elif dtype_name == "f32":
        return chaosuan.DataType.F32
    elif dtype_name == "f64":
        return chaosuan.DataType.F64
    elif dtype_name == "bf16":
        return chaosuan.DataType.BF16
    elif dtype_name == "i32":
        return chaosuan.DataType.I32
    elif dtype_name == "i64":
        return chaosuan.DataType.I64
    elif dtype_name == "u32":
        return chaosuan.DataType.U32
    elif dtype_name == "u64":
        return chaosuan.DataType.U64
    elif dtype_name == "bool":
        return chaosuan.DataType.BOOL
    else:
        raise ValueError(f"Unsupported dtype name: {dtype_name}")


def dtype_name(chaosuan_dtype: chaosuan.DataType):
    if chaosuan_dtype == chaosuan.DataType.F16:
        return "f16"
    elif chaosuan_dtype == chaosuan.DataType.F32:
        return "f32"
    elif chaosuan_dtype == chaosuan.DataType.F64:
        return "f64"
    elif chaosuan_dtype == chaosuan.DataType.BF16:
        return "bf16"
    elif chaosuan_dtype == chaosuan.DataType.I32:
        return "i32"
    elif chaosuan_dtype == chaosuan.DataType.I64:
        return "i64"
    elif chaosuan_dtype == chaosuan.DataType.U32:
        return "u32"
    elif chaosuan_dtype == chaosuan.DataType.U64:
        return "u64"
    elif chaosuan_dtype == chaosuan.DataType.BOOL:
        return "bool"
    else:
        raise ValueError(f"Unsupported chaosuan dtype: {chaosuan_dtype}")
