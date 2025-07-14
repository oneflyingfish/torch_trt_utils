from .time_bench import TIME_PERF
import torch

def test_torch_cuda_time(enable=TIME_PERF):
    def wrapper0(func):
        def wrapper(*args, **kwargs):
            if enable:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                output = func(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                print(f"<@{func.__qualname__}>: {start.elapsed_time(end)*0.001:.3f}s")
                return output
            else:
                return func(*args, **kwargs)

        return wrapper

    return wrapper0
