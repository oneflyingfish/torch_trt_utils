import torch
import pycuda.autoinit
import pycuda.driver as cuda
from typing import List, Tuple, Dict
import numpy as np
from .memory import HostDeviceMem
import tensorrt as trt
import os
from typing import Optional, List
from ..executor import ModelExectuor, TensorDesc, TensorDataType


class TensorRTExecutor(ModelExectuor):
    def __init__(
        self,
        model_paths: List[str] | str = [],
        cuda_id=0,
    ):
        if cuda_id != 0:
            print(
                f"warning: device!=0 may meet error at this time, use export CUDA_VISIBLE_DEVICES={cuda_id} instead is suggested."
            )

        if isinstance(model_paths, str):
            model_paths = [model_paths]

        self.engine_path = model_paths[0]

        assert os.path.exists(
            self.engine_path
        ), f"model path {self.engine_path} does not exists!"

        # init torch context
        a = torch.randn(size=(1, 1), device=torch.device(f"cuda:{cuda_id}"))
        del a

        # cuda.init()
        self.gpu_id = cuda_id
        # self.gpu = cuda.Device(self.gpu_id)
        # self.cuda_ctx = self.gpu.make_context()

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(self.trt_logger)
        self.trt_engine = None
        with open(self.engine_path, "rb") as f:
            self.trt_engine = self.trt_runtime.deserialize_cuda_engine(f.read())

        self.trt_context = self.trt_engine.create_execution_context()

        self.torch_device = torch.device(f"cuda:{self.gpu_id}")

        # self.cuda_stream = cuda.Stream()
        # self.torch_stream = torch.cuda.ExternalStream(
        #     self.cuda_stream.handle, device=self.torch_device
        # )

        self.torch_stream = torch.cuda.current_stream()
        self.cuda_stream = cuda.Stream(self.torch_stream.cuda_stream)

        # init desc from engine file
        self.inputs_desc = []  # type:List[TensorDesc]
        self.outputs_desc = []  # type:List[TensorDesc]

        self.current_inputs_shape = {}
        self.current_outputs_shape = {}

        self.name_to_index = {}  # type:Dict[str,int]

        for idx in range(self.trt_engine.num_io_tensors):
            name = self.trt_engine.get_tensor_name(idx)
            shape = [int(dim) for dim in self.trt_engine.get_tensor_shape(name)]
            dtype = self.TrtDataTypeToTensorDataType(
                self.trt_engine.get_tensor_dtype(name)
            )

            io_mode = self.trt_engine.get_tensor_mode(name)
            if io_mode == trt.TensorIOMode.INPUT:
                desc = TensorDesc(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    dynamic_desc={
                        "min": [
                            int(dim)
                            for dim in self.trt_engine.get_tensor_profile_shape(
                                name, 0
                            )[0]
                        ],
                        "best": [
                            int(dim)
                            for dim in self.trt_engine.get_tensor_profile_shape(
                                name, 0
                            )[1]
                        ],
                        "max": [
                            int(dim)
                            for dim in self.trt_engine.get_tensor_profile_shape(
                                name, 0
                            )[2]
                        ],
                    },
                )
                self.name_to_index[name] = len(self.inputs_desc)
                self.inputs_desc.append(desc)
                self.current_inputs_shape = {f"{name}": None}

            elif io_mode == trt.TensorIOMode.OUTPUT:
                desc = TensorDesc(name=name, shape=shape, dtype=dtype)

                self.name_to_index[name] = len(self.outputs_desc)
                self.outputs_desc.append(desc)
                self.current_outputs_shape = {f"{name}": None}

            else:
                print(f"warning: unknown io_tensor, name={name}, io_mode={io_mode}")

        # read output shape
        for type_str in ["min", "best", "max"]:
            for desc in self.inputs_desc:
                self.trt_context.set_input_shape(
                    desc.name, tuple(desc.dynamic_desc[type_str])
                )

            for desc in self.outputs_desc:
                if desc.dynamic_desc is None:
                    desc.dynamic_desc = dict()
                desc.dynamic_desc[type_str] = [
                    int(dim) for dim in self.trt_context.get_tensor_shape(desc.name)
                ]
                print(
                    f"add {type_str} {desc.name}: {self.trt_context.get_tensor_shape(desc.name)}"
                )

                print(
                    f"{type_str} {desc.name}: {desc.dynamic_desc}, {id(desc)} {id(desc.dynamic_desc)}, {id(desc.dynamic_desc[type_str])}"
                )

        # self.cuda_ctx_pushed = False

        # init cuda memory with max shape
        self.inputs_mem = []  # type:List[HostDeviceMem]
        self.outputs_mem = []  # type:List[HostDeviceMem]

        for desc in self.inputs_desc:
            mem = HostDeviceMem(
                desc.dynamic_desc["max"],
                TensorDataType.to_numpy_type(desc.dtype),
                self.cuda_stream,
            )
            self.inputs_mem.append(mem)

        for desc in self.outputs_desc:
            mem = HostDeviceMem(
                desc.dynamic_desc["max"],
                TensorDataType.to_numpy_type(desc.dtype),
                self.cuda_stream,
            )
            self.outputs_mem.append(mem)

    def SetInputShapes(self, shapes: Dict[str, List[int]] | List[List[int]]):
        if isinstance(shapes, list):
            shapes = {self.inputs_desc[i].name: shapes[i] for i in range(len(shapes))}

        # set input shape
        for name, shape in shapes.items():
            shape = [int(dim) for dim in shape]
            if shape != self.current_inputs_shape[name]:
                self.current_inputs_shape[name] = shape
                self.trt_context.set_input_shape(name, tuple(shape))

                mem = self.inputs_mem[self.name_to_index[name]]
                mem.set_shape(shape)
                self.trt_context.set_tensor_address(name, int(mem.ptr()))

        # read output shape
        for desc in self.outputs_desc:
            name = desc.name
            shape = [int(dim) for dim in self.trt_context.get_tensor_shape(name)]
            self.current_outputs_shape[name] = shape

            mem = self.outputs_mem[self.name_to_index[name]]
            mem.set_shape(shape)
            self.trt_context.set_tensor_address(name, int(mem.ptr()))

    def Inference(
        self, inputs: List[np.ndarray] | List[torch.Tensor], output_type="numpy"
    ) -> List[np.ndarray] | List[torch.Tensor]:
        """
        input and output with torch.Tensor in CUDA can be faster than numpy. if output_type="torch", data will put in CUDA.
        """
        assert self.trt_context is not None, "executor not init"

        input_desc = self.GetModelInputDesc()

        assert len(inputs) == len(
            self.GetModelInputDesc()
        ), f"expected {len(input_desc)} inputs, get f{len(inputs)}"

        self.SetInputShapes([list(tensor.shape) for tensor in inputs])

        for i, tensor in enumerate(inputs):
            if isinstance(tensor, np.ndarray):
                self.inputs_mem[i].set_numpy(tensor)
            else:
                self.inputs_mem[i].set_torch(tensor)

        self.trt_context.execute_async_v3(
            stream_handle=self.cuda_stream.handle,
        )

        if output_type == "numpy":
            return [mem.read_numpy() for mem in self.outputs_mem]
        elif output_type == "torch":
            return [mem.read_torch() for mem in self.outputs_mem]
        else:
            raise Exception("unsupport output_type")

    def GetModelInputDesc(self) -> List[TensorDesc]:
        return self.inputs_desc

    def GetModelOutputDesc(self) -> List[TensorDesc]:
        return self.outputs_desc

    def TrtDataTypeToTensorDataType(self, type) -> TensorDataType:
        return TensorDataType.from_numpy_type(trt.nptype(type))

    # def push_ctx(self):
    #     self.cuda_ctx.push()

    # def pop_ctx(self):
    #     self.cuda_ctx.pop()

    def Release(self):
        if self.torch_stream is not None:
            self.torch_stream.synchronize()
            self.cuda_stream.synchronize()
            del self.torch_stream
            del self.trt_context
            del self.trt_engine
            del self.trt_runtime

            self.torch_stream = None
            self.trt_context = None
            self.trt_engine = None
            self.trt_runtime = None

        # if self.cuda_ctx is not None:
        #     self.cuda_ctx.detach()
        #     self.cuda_ctx = None

    def __del__(self):
        self.Release()
