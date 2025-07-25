import torch
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np


class HostDeviceMem(object):
    def __init__(self, max_shape, dtype: np.dtype, stream: cuda.Stream = None):
        self.dtype = dtype  # np.dtype
        self.current_shape = max_shape
        self.max_shape = max_shape
        self.host = cuda.pagelocked_empty((trt.volume(max_shape),), self.dtype)
        self.device = cuda.mem_alloc(self.host.nbytes)

        if stream is None:
            self.stream = cuda.Stream()
        else:
            self.stream = stream

        self.dtype_mapping = {
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.uint8: torch.uint8,
            np.bool_: torch.bool,
        }  # type:dict

    def shape_element_size(self, shape) -> int:
        return trt.volume(shape)

    def shape_size(self, shape, dtype: np.dtype = None):
        if dtype is None:
            dtype = self.dtype
        return self.shape_element_size(shape) * dtype().itemsize

    def set_batchsize(self, batchsize=-1):
        shape = self.max_shape
        if batchsize > 0:
            shape[0] = batchsize
        self.set_shape(shape)

    def set_shape(self, shape=None):
        if shape is None:
            self.current_shape = self.max_shape
        else:
            self.current_shape = shape

    def ptr(self):
        return int(self.device)

    def set_numpy(self, array: np.ndarray, only_host=False):
        np.copyto(self.host[: self.element_size()], array.ravel())
        if not only_host:
            cuda.memcpy_htod_async(int(self.device), self.host, self.stream)
            self.stream.synchronize()

    def set_torch(self, array: torch.Tensor):
        if array.device == torch.device("cpu"):
            self.set_numpy(array.numpy())
        else:
            cuda.memcpy_dtod_async(
                int(self.device),
                array.contiguous().data_ptr(),
                self.byte_size(),
                self.stream,
            )
            self.stream.synchronize()

    def numpy2torch_type(self, type) -> torch.dtype:
        torch_type = self.dtype_mapping.get(type)
        assert torch_type is not None, "unsupported type"

        return torch_type

    def read_torch(self, dtype: torch.dtype = None) -> torch.Tensor:
        result = torch.empty(
            size=self.current_shape,
            memory_format=torch.contiguous_format,
            dtype=self.numpy2torch_type(self.dtype) if dtype is None else dtype,
            device=f"cuda:0",
        )

        cuda.memcpy_dtod_async(
            result.contiguous().data_ptr(),
            int(self.device),
            self.byte_size(),
            self.stream,
        )
        self.stream.synchronize()
        return result

    def read_numpy(self) -> np.ndarray:
        cuda.memcpy_dtoh_async(self.host, int(self.device), self.stream)
        self.stream.synchronize()

        return (
            np.array(self.host)[: self.element_size()]
            .reshape(self.current_shape)
            .copy()
        )

    def max_byte_size(self) -> int:
        return self.shape_size(self.max_shape)

    def max_element_size(self) -> int:
        return self.shape_element_size(self.max_shape)

    def byte_size(self) -> int:
        return self.shape_size(self.current_shape)

    def element_size(self) -> int:
        return self.shape_element_size(self.current_shape)

    def __len__(self):
        return self.element_size()

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
