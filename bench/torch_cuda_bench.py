from .time_bench import TIME_PERF
from .cuda_bench import test_time_cuda


def test_torch_cuda_time(enable=TIME_PERF):
    print(
        "warning: you have use deperated function, use bench.cuda_bench.test_time_cuda instead"
    )
    return test_time_cuda(enable=enable)
