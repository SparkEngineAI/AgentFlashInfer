import argparse
import json
import os

import torch.distributed as dist

from vllm_ascend.soc_info import NPUSocInfo
from pangu_infer.pangu.utils.ranktable import get_cmd_stdout, get_local_device_list, \
    postprocess_allgather_global_device_list, gen_ranktable, save_ranktable_to

parser = argparse.ArgumentParser(
    description="Arguments of rank table generator", )
parser.add_argument("--local-host", type=str, required=True, help="local ip")
parser.add_argument("--prefill-device-cnt",
                    type=int,
                    required=True,
                    help="number of prefill devices")
parser.add_argument("--decode-device-cnt",
                    type=int,
                    required=True,
                    help="number of decode devices")
args = parser.parse_args()
local_host = args.local_host
prefill_device_cnt = args.prefill_device_cnt
decode_device_cnt = args.decode_device_cnt

print("enter py")

hccn_tool_path = os.environ.get("HCCN_TOOL_PATH",
                                "/usr/local/Ascend/driver/tools/hccn_tool")
master_addr = os.environ.get("MASTER_ADDR")
master_port = os.environ.get("MASTER_PORT")
rank = os.environ.get("RANK")
local_rank = os.environ.get("LOCAL_RANK")
# This variable is set by torchrun,
# and is different from WORLD_SIZE in gen_rank_table.sh.
world_size = os.environ.get("WORLD_SIZE")
soc_info = NPUSocInfo()

print(f"local_host: {local_host}")
print("gen ranktable.json")

# generate local device list for local rank 0, and gather it to all ranks
local_device_list = get_local_device_list(local_host, local_rank)

# allgather to get global device list
dist.init_process_group(backend=dist.Backend.GLOO)
global_device_list = [None] * dist.get_world_size()
dist.all_gather_object(global_device_list, local_device_list)
global_device_list = postprocess_allgather_global_device_list(global_device_list)

assert (prefill_device_cnt + decode_device_cnt) <= len(global_device_list), \
"prefill_device_cnt + decode_device_cnt must be less than or equal to number of all devices in cluster"

# gen ranktable and save it
ranktable = gen_ranktable(world_size, global_device_list, prefill_device_cnt, decode_device_cnt)
default_ranktable_path = os.getenv("RANKTABLE_PATH", "ranktable.json")
if local_rank == '0':
    save_ranktable_to(ranktable, default_ranktable_path)
