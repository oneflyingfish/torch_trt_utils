from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from enum import Enum, auto
import numpy as np


class TensorDataType(Enum):
    UNDEFINED = 0
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()

    UINT8 = auto()
    UINT16 = auto()
    UINT32 = auto()
    UINT64 = auto()

    STRING = auto()
    BOOL = auto()

    FLOAT8 = auto()
    BFLOAT16 = auto()
    FLOAT16 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()

    COMPLEX64 = auto()
    COMPLEX128 = auto()

    @staticmethod
    def from_numpy_type(type):
        if type == np.int8:
            return TensorDataType.INT8
        elif type == np.int16:
            return TensorDataType.INT16
        elif type == np.int32:
            return TensorDataType.INT32
        elif type == np.int64:
            return TensorDataType.INT64

        elif type == np.uint8:
            return TensorDataType.UINT8
        elif type == np.uint16:
            return TensorDataType.UINT16
        elif type == np.uint32:
            return TensorDataType.UINT32
        elif type == np.uint64:
            return TensorDataType.UINT64

        elif type == np.float16:
            return TensorDataType.FLOAT16
        elif type == np.float32:
            return TensorDataType.FLOAT32
        elif type == np.float64:
            return TensorDataType.FLOAT64

        elif type == np.string_:
            return TensorDataType.STRING
        elif type == np.bool_:
            return TensorDataType.BOOL

        elif type == np.complex64:
            return TensorDataType.COMPLEX64
        elif type == np.complex128:
            return TensorDataType.COMPLEX128
        else:
            raise Exception(f"unsupport numpy type from {type} to TensorDataType")

    @staticmethod
    def to_numpy_type(type):
        if type == TensorDataType.INT8:
            return np.int8
        elif type == TensorDataType.INT16:
            return np.int16
        elif type == TensorDataType.INT32:
            return np.int32
        elif type == TensorDataType.INT64:
            return np.int64

        elif type == TensorDataType.UINT8:
            return np.uint8
        elif type == TensorDataType.UINT16:
            return np.uint16
        elif type == TensorDataType.UINT32:
            return np.uint32
        elif type == TensorDataType.UINT64:
            return np.uint64

        elif type == TensorDataType.FLOAT16:
            return np.float16
        elif type == TensorDataType.FLOAT32:
            return np.float32
        elif type == TensorDataType.FLOAT64:
            return np.float64

        elif type == TensorDataType.STRING:
            return np.string_
        elif type == TensorDataType.BOOL:
            return np.bool_

        elif type == TensorDataType.COMPLEX64:
            return np.complex64
        elif type == TensorDataType.COMPLEX128:
            return np.complex128
        else:
            raise Exception(f"unsupport numpy type for {type} to numpy")


class TensorDesc:
    def __init__(self, name, shape, dtype=TensorDataType.FLOAT32, dynamic_desc={}):
        self.name: str = name
        self.shape: Tuple[int] = shape
        self.dtype: TensorDataType = dtype

        self.dynamic_desc: dict = dynamic_desc

    def __str__(self):
        if len(self.dynamic_desc) > 0:
            return f'name="{self.name}", shape={self.shape}, dtype={self.dtype}, dynamic_desc={self.dynamic_desc}'
        else:
            return (
                f'name="{self.name}", shape={self.shape}, dtype={self.dtype}, dynamic'
            )


class ModelExectuor(ABC):
    def __init__(self):
        pass

    def warmup(self, inputs, times=3):
        for _ in range(times):
            self.Inference(inputs)

    def SetInputShapes(self, shapes: Dict[str, List[int]] | List[List[int]]):
        pass

    @abstractmethod
    def Inference(self, inputs: list, output_type="numpy") -> list:
        return []

    @abstractmethod
    def GetModelInputDesc(self) -> List[TensorDesc]:
        return []

    @abstractmethod
    def GetModelOutputDesc(self) -> List[TensorDesc]:
        return []

    def PrintIODesc(self):
        print("________________________")
        print("input desc:")
        for desc in self.GetModelInputDesc():
            print(desc)

        print("output desc:")
        for desc in self.GetModelOutputDesc():
            print(desc)
        print("________________________")

    @abstractmethod
    def Release(self):
        pass
