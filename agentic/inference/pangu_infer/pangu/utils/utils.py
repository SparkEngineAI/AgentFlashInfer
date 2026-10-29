import inspect
import torch
import torch_npu

PANGU_DEBUG = False


def debug_data(data, prefix=None, rank=0):
    if not PANGU_DEBUG:
        return
    if isinstance(data, torch.Tensor):
        suffix = f"{data} {data.dtype} {data.shape} {data.float().sum()}"
    elif isinstance(data, list):
        suffix = f"{data}"
    elif isinstance(data, int):
        suffix = f"{data}"
    elif isinstance(data, str):
        suffix = f"{data}"
    else:
        suffix = f"{data}" 
    local_rank = torch.distributed.get_rank()
    if rank == -1 or rank == local_rank:
        frame = inspect.stack()[1]
        data_name = frame.code_context[0].split('(')[1].split(',')[0]
        print(f"rank:{local_rank} file:{frame.filename} line:{frame.lineno} prefix:{prefix} {data_name} {suffix}",
              flush=True)


def print_rank_0(message):
    if torch.distributed.get_rank() == 0:
        print(message, flush=True)


def check(judge, prefix=None):
    if not judge:
        frame = inspect.stack()[1]
        raise RuntimeError(f"file:{frame.filename} line:{frame.lineno} prefix:{prefix}")
