# coding:utf-8


import os
# 添加根目录到 path,确保正常引入
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ray

from unified_store import UnifiedStoreManager


def initialize_ray():
    if ray.is_initialized():
        print(f"Ray 已经在当前进程中运行。")
        print(f"Dashboard URL: {ray.worker.get_dashboard_url()}")
        return

    try:
        context = ray.init(address="auto", ignore_reinit_error=True)
        print("成功连接到现有的 Ray 实例。")

    except ConnectionError:
        print("未检测到现有 Ray 实例，正在启动新的本地实例...")
        context = ray.init()
        print("新的本地 Ray 实例已启动。")


# --- 使用示例 ---
if __name__ == "__main__":
    initialize_ray()

    print("Ray resources:", ray.cluster_resources())
    mgr = UnifiedStoreManager.remote()

    # 启动目标存储
    mgr.start_inmem_store.remote()
    mgr.start_file_store.remote()
    mgr.start_message_queue_service.remote(prompts_num=1024, n_samples_per_prompt=10)
