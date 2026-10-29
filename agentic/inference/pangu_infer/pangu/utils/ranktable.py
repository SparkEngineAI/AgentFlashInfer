import argparse
import json
import os

import torch.distributed as dist

from vllm_ascend.soc_info import NPUSocInfo

hccn_tool_path = os.environ.get("HCCN_TOOL_PATH",
                                "/usr/local/Ascend/driver/tools/hccn_tool")
soc_info = NPUSocInfo()


def get_cmd_stdout(cmd):
    import subprocess
    return subprocess.run(cmd, capture_output=True,
                          shell=True).stdout.decode("utf-8").strip()


# generate local device list for local rank 0, and gather it to all ranks
def get_local_device_list(local_host, local_rank):
    num_cards = get_cmd_stdout("npu-smi info -l | grep \"Total Count\"").split(
        ":")[1].strip()
    num_cards = int(num_cards)
    chips_per_card = get_cmd_stdout("npu-smi info -l | grep \"Chip Count\"").split(
        "\n")[0].split(":")[1].strip()
    chips_per_card = int(chips_per_card)
    local_device_list: list[dict[str, str]] = list()
    if local_rank == "0":
        super_pod_id = "0"
        for card_id in range(num_cards):
            for chip_id in range(chips_per_card):
                device_id = card_id * chips_per_card + chip_id
                if soc_info.is_a3:
                    if os.getenv("HCCL_INTRA_ROCE_ENABLE", "0") != "0":
                        print(f"---------enable HCCL_INTRA_ROCE_ENABLE")
                        device_ip = get_cmd_stdout(
                            f"{hccn_tool_path} -i {device_id} -ip -g | grep ipaddr"
                        ).split(":")[1].strip()
                    else:
                        device_ip = get_cmd_stdout(
                            f"{hccn_tool_path} -i {device_id} -vnic -g | grep ipaddr"
                        ).split(":")[1].strip()
                    super_device_id = get_cmd_stdout(
                        f"npu-smi info -t spod-info -i {card_id} -c {chip_id} | grep SDID"
                    ).split(":")[1].strip()
                    super_pod_id = get_cmd_stdout(
                        f"npu-smi info -t spod-info -i {card_id} -c {chip_id} | grep \"Super Pod ID\""
                    ).split(":")[1].strip()
                else:
                    device_ip = get_cmd_stdout(
                        f"{hccn_tool_path} -i {device_id} -ip -g | grep ipaddr"
                    ).split(":")[1].strip()

                device_info = {
                    "server_id": local_host,
                    "device_id": str(device_id),
                    "device_ip": str(device_ip),
                }
                if soc_info.is_a3:
                    device_info.update({
                        "super_pod_id": str(super_pod_id),
                        "super_device_id": str(super_device_id)
                    })
                local_device_list.append(device_info)
    return local_device_list


def postprocess_allgather_global_device_list(global_device_list):
    global_device_list = [
        device_info for device_list in global_device_list
        for device_info in device_list  # type: ignore[attr-defined]
    ]
    cnt = 1
    for device_info in global_device_list:  # type: ignore[assignment]
        device_info["cluster_id"] = str(cnt)
        cnt += 1
    return global_device_list


def gen_ranktable(world_size, global_device_list, prefill_device_cnt, decode_device_cnt):
    ranktable = {
        "version":
        "1.2",
        "server_count":
        str(world_size),
        "prefill_device_list":
        global_device_list[:prefill_device_cnt],
        "decode_device_list":
        global_device_list[prefill_device_cnt:prefill_device_cnt +
                        decode_device_cnt],
        "status":
        "completed"
    }
    return ranktable


def save_ranktable_to(ranktable, default_ranktable_path):
    with open(default_ranktable_path, "w") as f:
        json.dump(ranktable, f, indent=4)
    print("gen ranktable.json done")
