import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import onnx
from tensorrt import CalibrationAlgoType
import time
from PIL import Image
from typing import Optional, List
from abc import ABC, abstractmethod
from .memory import HostDeviceMem
from copy import deepcopy

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class CalibratorDatasetObject(ABC):
    def __init__(self):
        self.datasets = []
        self.names = []

    def get_index_by_name(self, name: str):
        return self.names.index(name)

    def data(self) -> list:
        return self.datasets

    def __len__(self) -> int:
        if self.datasets is None:
            return 0
        return len(self.datasets)

    def shape(self, input_id=0) -> Optional[tuple]:
        if len(self) < 1:
            return None
        return self.datasets[0][input_id].shape

    def dtype(self, input_id=0) -> np.dtype:
        if len(self) < 1:
            return None
        return self.datasets[0][input_id].dtype

    def __getitem__(self, index) -> Optional[List[np.ndarray]]:
        if index < self.datasize:
            return self.datasets[index]
        else:
            return None

    def input_count(self):
        if self.datasets is None:
            return 0
        else:
            return len(self.datasets[0])

    @property
    def datasize(self):
        return len(self)

    @property
    def batch_count(self):
        return len(self)


class MyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader: CalibratorDatasetObject, cache_file="int8.cache"):
        super(MyEntropyCalibrator, self).__init__()
        self.data_loader = data_loader
        self.cache_file = cache_file
        self.batch_size = len(data_loader)
        assert self.batch_size > 0, "empty CalibratorDataset"

        self.current_index = 0

        self.stream = cuda.Stream()
        # cuda.mem_alloc(
        #     trt.volume(self.data_loader[0][i].shape)
        #     * self.data_loader[0][i].dtype.itemsize
        # )

        self.device_inputs = [
            HostDeviceMem(
                self.data_loader[0][i].shape,
                self.data_loader[0][i].dtype,
                stream=self.stream,
            )
            for i in range(self.data_loader.input_count())
        ]

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names=None):
        if self.current_index >= len(self.data_loader):
            return None

        batch = self.data_loader[self.current_index]

        for i in range(self.data_loader.input_count()):
            self.device_inputs[i].set_numpy(batch[i])
            # cuda.memcpy_htod(self.device_inputs[i], batch[i])
        self.current_index += 1

        index_order = []
        if names is None:
            index_order = list(range(self.data_loader.input_count()))
        else:
            index_order = [self.data_loader.get_index_by_name(name) for name in names]
        return [self.device_inputs[index].ptr() for index in index_order]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def save_engine(
    onnx_file_path,
    trt_model_path,
    max_workspace_size=1 << 30,
    fp16_mode=True,
    int8_mode=False,
    min_batch=1,
    optimize_batch=1,
    max_batch=1,
    calibrator=None,
) -> bool:
    onnx_model = onnx.load(onnx_file_path)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode and builder.platform_has_fast_int8:
        assert calibrator is not None, "ERROR: no calibration_set."
        config.int8_calibrator = calibrator
        config.set_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()
    onnx_input_name = onnx_model.graph.input[0].name
    onnx_input_shape = [
        dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim
    ]
    onnx_input_shape[0] = min_batch
    min_shape = tuple(onnx_input_shape)
    onnx_input_shape[0] = optimize_batch
    opt_shape = tuple(onnx_input_shape)
    onnx_input_shape[0] = max_batch
    max_shape = tuple(onnx_input_shape)
    print(
        f"quant for input min-shape:{min_shape}, opt-shape: {opt_shape}, max-shape:{max_shape}"
    )
    profile.set_shape(onnx_input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build serialized engine.")
        return False

    with open(trt_model_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved as {trt_model_path}")
    return True


def save_engine_mixed_inputs(
    onnx_file_path,
    trt_model_path,
    max_workspace_size=1 << 30,
    fp16_mode=True,
    int8_mode=False,
    dynamic_axes=None,
    calibrator=None,
):
    onnx_model = onnx.load(onnx_file_path)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode and builder.platform_has_fast_int8:
        assert calibrator is not None, "ERROR: no calibration_set."
        config.int8_calibrator = calibrator
        config.set_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()
    for input_node in onnx_model.graph.input:
        onnx_input_name = input_node.name
        onnx_input_shape = [
            dim.dim_value for dim in input_node.type.tensor_type.shape.dim
        ]

        # support_dynamic = False
        # for v in onnx_input_shape:
        #     if not isinstance(v, int) or v < 1:
        #         support_dynamic = True
        #         break
        # if not support_dynamic:
        #     continue

        min_shape = deepcopy(onnx_input_shape)
        opt_shape = deepcopy(onnx_input_shape)
        max_shape = deepcopy(onnx_input_shape)
        if dynamic_axes is not None:
            if onnx_input_name in dynamic_axes:
                if "min" in dynamic_axes[onnx_input_name]:
                    for axis in dynamic_axes[onnx_input_name]["min"]:
                        min_shape[axis] = dynamic_axes[onnx_input_name]["min"][axis]

                if "opt" in dynamic_axes[onnx_input_name]:
                    for axis in dynamic_axes[onnx_input_name]["opt"]:
                        opt_shape[axis] = dynamic_axes[onnx_input_name]["opt"][axis]

                if "max" in dynamic_axes[onnx_input_name]:
                    for axis in dynamic_axes[onnx_input_name]["max"]:
                        max_shape[axis] = dynamic_axes[onnx_input_name]["max"][axis]

        min_shape = tuple(min_shape)
        opt_shape = tuple(opt_shape)
        max_shape = tuple(max_shape)

        for check_shape in [min_shape, opt_shape, max_shape]:
            for v in check_shape:
                assert (
                    isinstance(v, int) and v > 0
                ), f"need to give range for {onnx_input_name}, get {[min_shape, opt_shape, max_shape]}"
        profile.set_shape(onnx_input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build serialized engine.")
        return False

    with open(trt_model_path, "wb") as f:
        f.write(serialized_engine)
    return True


# if __name__ == "__main__":
#     input_model_path = "model/yolov11m_dynamic.onnx"
#     output_model_path = "model/yolov11m_dynamic.engine"

#     calibration_dataset_path = "dataset/"  # some *.mp4 videos in fold that can run yolo , need no special name and ratio
#     data_set = CalibratorDataset(
#         calibration_dataset_path,
#         input_shape=(-1, 3, 640, 640),
#         batch_size=1,
#         skip_frame=20,
#         dataset_limit=1 * 1000,
#     )
#     calibrator = MyEntropyCalibrator(
#         data_loader=data_set, cache_file="model/yolov11m_dynamic.cache"
#     )

#     save_engine(
#         input_model_path,
#         output_model_path,
#         fp16_mode=True,
#         int8_mode=True,
#         min_batch=1,
#         optimize_batch=1,
#         max_batch=20,
#         calibrator=calibrator,
#     )
