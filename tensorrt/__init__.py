import torch

torch.cuda.init()
dummy = torch.tensor([0]).cuda()
del dummy

# import pycuda.autoinit
import pycuda.driver as cuda

from .memory import HostDeviceMem
from .quant import (
    CalibratorDatasetObject,
    MyEntropyCalibrator,
    save_engine,
    save_engine_mixed_inputs,
)
from .tensorrt_executor import TensorRTExecutor
