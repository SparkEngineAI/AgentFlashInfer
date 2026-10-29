import os
import asyncio
import logging
from datetime import datetime, time
import aiofiles
from pathlib import Path

from config import config
from agent_infer.unified_store.unified_store import UnifiedStore, UnifiedStoreManager
# 配置上传失败日志
os.makedirs(config.WORKSPACE_PATH, exist_ok=True)
UPLOAD_FAIL_LOG = os.path.realpath(os.path.join(config.WORKSPACE_PATH, 'upload_workspace.log'))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
upload_logger = logging.getLogger("upload_fail")
handler = logging.FileHandler(UPLOAD_FAIL_LOG)
handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
upload_logger.addHandler(handler)
upload_logger.propagate = False

# store_manager.py

import ray
from threading import Lock
from typing import Optional


class StoreManager:
    _instance: Optional["StoreManager"] = None
    _lock = Lock()
    _store: Optional[UnifiedStore] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_store(self, ray_init_kwargs=None) -> UnifiedStore:
        """
        Obtain the global UnifiedStore singleton.

        Args:
        ray_init_kwargs (dict, optional): A parameter passed to ray.init(). For example, {'address': 'auto'} or {'num_cpus': 4}. It only takes effect when Ray is first initialized.
        """
        if self._store is not None:
            return self._store

        with self._lock:
            if self._store is not None:
                return self._store

            # 检查 Ray 是否已连接
            try:
                ray.get_runtime_context().get_worker_id()
                ray_already_started = True
            except Exception:
                ray_already_started = False

            # 如果 Ray 未启动，则初始化
            if not ray_already_started:
                init_args = ray_init_kwargs or {}
                # 避免重复 init 报错
                try:
                    ray.init(**init_args)
                except RuntimeError as e:
                    if "Ray has already been initialized" in str(e):
                        pass  # 可能被其他线程初始化了
                    else:
                        raise

            # 创建或获取 UnifiedStoreManager Actor
            # 方案1：通过命名 Actor（推荐，便于跨模块共享）
            actor_name = "UnifiedStoreManager"

            try:
                mgr = ray.get_actor(actor_name)
            except ValueError:
                # Actor 不存在，创建它
                mgr = UnifiedStoreManager.options(
                    name=actor_name,
                    # lifetime="detached",  # 关键：即使 driver 退出，Actor 仍存在（适用于长时间运行应用）
                    max_concurrency=1000
                ).remote()
                # 初始化文件存储根目录（如果需要）
                ray.get(mgr.start_file_store.remote())

            self._store = UnifiedStore(mgr)
            self._store.add_store(["file"])  # 初始化 file backend
            self._initialized = True

            return self._store

    # def shutdown(self):
    #     """手动关闭资源（可选）"""
    #     if ray.is_initialized():
    #         ray.shutdown()
    #         # shutil.rmtree(root, ignore_errors=True)
    #     self._instance = None
    # #     ray.shutdown()
    # #     shutil.rmtree(root, ignore_errors=True)


# 全局便捷函数
def get_file_store(ray_init_kwargs=None) -> UnifiedStore:
    """
    Global unified entry: Obtain the UnifiedStore singleton.

    examples：
        store = get_store()
        store.file.create("test.txt", "hello")
    """

    return StoreManager().get_store(ray_init_kwargs=ray_init_kwargs)


def upload_session_to_dist_storage(session_id):
    """
    Upload the local workspace with session_id='123' to the remote storage.
    Supported file types:.json,.jsonl,.md,.txt and any UTF-8 text files.
    Path structure: "ddv2/workspace/YYYY/MM/DD/{session_id}/<relative_path>
    """
    store = get_file_store()
    workspace_path = Path(os.path.realpath(os.path.join(config.WORKSPACE_PATH, session_id)))
    local_session_dir = workspace_path
    today_partition = datetime.utcnow().strftime("%Y/%m/%d")
    remote_base_prefix = f"ddv2/workspace/{today_partition}/{session_id}"
    success_count = 0
    total_files = 0
    try:
        uploaded_paths = []  # Record all the remote paths that should be uploaded
        for local_file in local_session_dir.rglob("*"):
            if not local_file.is_file():
                continue
            total_files += 1
            rel_path = local_file.relative_to(local_session_dir)
            remote_filepath = f"{remote_base_prefix}/{rel_path.as_posix()}"
            try:
                # Suppose all the target files are UTF-8 text （json/jsonl/md/txt）
                content = local_file.read_text(encoding="utf-8")
                store.file.create(remote_filepath, content)
                uploaded_paths.append(remote_filepath)
                success_count += 1
            except UnicodeDecodeError as e:
                # Defensive log
                error_msg = f"[SESSION UPLOAD PARTIAL FAILURE] Non-UTF-8 file skipped: {local_file} ({e})"
                upload_logger.error(error_msg)
            except Exception as e:
                error_msg = f"[SESSION UPLOAD PARTIAL FAILURE] Upload failed for {local_file} → {remote_filepath}: {e}"
                upload_logger.error(error_msg)

        flat = store.file.list_files(remote_base_prefix, recursive=True)
        names_flat = {info.name for info in flat if '.' in info.name }
        if(len(names_flat) < len(uploaded_paths) or success_count < total_files ):
            upload_logger.error(f"[SESSION UPLOAD FAIL] {session_id}: {success_count}/{total_files} files uploaded failed. Uploaded_paths: {uploaded_paths},  Store.file.list_files: {names_flat}")
        upload_logger.info(f"[SESSION UPLOAD SUCCESSFUL] {session_id}: {success_count}/{total_files} files uploaded successfully")
    except Exception as e:
        upload_logger.exception(f"[SESSION UPLOAD FAIL] Unexpected error during upload of session {session_id}: {e}")