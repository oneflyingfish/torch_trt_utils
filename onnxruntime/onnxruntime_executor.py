import onnxruntime as ort
from typing import Optional, List
from ..executor import ModelExectuor, TensorDesc, TensorDataType
import onnx
import numpy as np
import torch


class OnnxRuntimeExecutor(ModelExectuor):
    def __init__(
        self,
        model_paths: List[str] | str = [],
        providers=None,
    ):
        if providers is None:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            
        if isinstance(model_paths, str):
            model_paths = [model_paths]

        self.onnx_path = model_paths[0]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        self.ort_session = ort.InferenceSession(
            self.onnx_path,
            providers=providers,
            sess_options=sess_options,
        )

        self.inputs_desc = []
        self.outputs_desc = []

        for desc in self.ort_session.get_inputs():
            self.inputs_desc.append(
                TensorDesc(
                    name=desc.name,
                    shape=desc.shape,
                    dtype=self.OrtDataTypeToTensorDataType(desc.type),
                )
            )

        for desc in self.ort_session.get_outputs():
            self.outputs_desc.append(
                TensorDesc(
                    name=desc.name,
                    shape=desc.shape,
                    dtype=self.OrtDataTypeToTensorDataType(desc.type),
                )
            )

    def Inference(self, inputs: list, output_type="numpy") -> list:
        input_desc = self.GetModelInputDesc()
        output_desc = self.GetModelOutputDesc()

        ort_inputs = {}
        assert len(inputs) == len(
            self.GetModelInputDesc()
        ), f"expected {len(input_desc)} inputs, get f{len(inputs)}"

        for idx, value in enumerate(inputs):
            ort_inputs[input_desc[idx].name] = (
                value if isinstance(value, np.ndarray) else value.detach().cpu().numpy()
            )

        outputs = self.ort_session.run([o.name for o in output_desc], ort_inputs)

        if output_type == "numpy":
            return outputs
        elif output_type == "torch":
            return [torch.from_numpy(value) for value in outputs]
        else:
            raise Exception("unsupport output_type")

    def GetModelInputDesc(self) -> List[TensorDesc]:
        return self.inputs_desc

    def GetModelOutputDesc(self) -> List[TensorDesc]:
        return self.outputs_desc

    def OrtDataTypeToTensorDataType(self, type_str) -> TensorDataType:
        if type_str in self.type_map_to_tensor:
            return self.type_map_to_tensor[type_str]
        else:
            return TensorDataType.UNDEFINED

    @property
    def type_map_to_tensor(self):
        return {
            "tensor(int8)": TensorDataType.INT8,
            "tensor(int16)": TensorDataType.INT16,
            "tensor(int32)": TensorDataType.INT32,
            "tensor(int64)": TensorDataType.INT64,
            "tensor(uint8)": TensorDataType.UINT8,
            "tensor(uint16)": TensorDataType.UINT16,
            "tensor(uint32)": TensorDataType.UINT32,
            "tensor(uint64)": TensorDataType.UINT64,
            "tensor(string)": TensorDataType.STRING,
            "tensor(bool)": TensorDataType.BOOL,
            "tensor(bfloat16)": TensorDataType.BFLOAT16,
            "tensor(float16)": TensorDataType.FLOAT16,
            "tensor(float)": TensorDataType.FLOAT32,
            "tensor(double)": TensorDataType.FLOAT64,
            "tensor(complex64)": TensorDataType.COMPLEX64,
            "tensor(complex128)": TensorDataType.COMPLEX128,
        }
