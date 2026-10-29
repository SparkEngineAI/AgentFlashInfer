import os
from typing import Any, List, Dict, Tuple, Union, Optional
from pathlib import Path
import ray


from agent_infer.unified_store.inmemory_store import InMemoryStoreClient, InMemoryStoreServer, TTLConfig
from agent_infer.unified_store.file_store import FileStoreClient
from agent_infer.unified_store.message_queue import MessageQueueClient


class UnifiedStore:
    def __init__(self, mgr: ray.actor.ActorHandle):
        self._mgr = mgr
        self._sub_clients: Dict[str, Any] = {}

    def add_store(self, names: List[str]) -> None:

        # Fetch all server addresses or local paths
        endpoint_map = ray.get(self._mgr.get_store_endpoints.remote(names))

        for name in names:
            if name in self._sub_clients:
                raise RuntimeError(f"Store '{name}' has already been added")

            endpoint = endpoint_map.get(name)
            if not endpoint:
                raise RuntimeError(f"No endpoint found for store '{name}'")

            if name == "file":
                client = FileStoreClient(endpoint)
            elif name == "mem":
                client = InMemoryStoreClient(endpoint)
            elif name == "mq":
                client = MessageQueueClient(endpoint)  
            else:
                raise RuntimeError(f"No handler defined for store '{name}'")

            self._sub_clients[name] = client
            setattr(self, name, client)

    def __getattr__(self, name: str):
        if name in self._sub_clients:
            return self._sub_clients[name]
        raise AttributeError(f"Unknown store '{name}'. Call add_store first.")

    def __dir__(self):
        return super().__dir__() + list(self._sub_clients.keys())


@ray.remote(name="UnifiedStoreManager")
class UnifiedStoreManager:
    def __init__(self):
        # store_name -> (server instance, endpoint)
        self._servers: Dict[str, Tuple[Any, Any]] = {}
        # TODO add client list


    def start_inmem_store(
        self,
        port: int = 4242,
        config: Optional[TTLConfig] = None  # 可选 TTLConfig，可传入自定义配置
    ) -> str:
        if "mem" in self._servers:
            raise RuntimeError("InMemoryStore has been started")
        # 传递 port 与可选配置到远程 Actor
        srv = InMemoryStoreServer.remote(port, config)
        addr = ray.get(srv.get_address.remote())
        srv.run.remote()
        self._servers["mem"] = (srv, addr)
        return addr


    def start_file_store(self, root_path: str = "/tmp/unified_store") -> str:
        # 1. 防止重复启动
        if "file" in self._servers:
            raise RuntimeError("FileStore has already been started")

        # 2. 类型及绝对路径校验
        if not isinstance(root_path, str) or not os.path.isabs(root_path):
            raise ValueError(f"Invalid root_path: {root_path!r}. Must be an absolute path string")

        # 3. 规范化（去掉多余斜杠、处理 . .. 等）
        normalized = os.path.normpath(root_path)
        base = Path(normalized)

        # 4. 父目录权限检查
        parent = base.parent if base.parent.suffix or base.parent.name else Path("/")
        if not os.access(str(parent), os.R_OK | os.W_OK):
            raise PermissionError(f"No right to mk_dir root_path for file store in Unified Store: {parent!r}")

        # 5. 创建目录（含多级）
        base.mkdir(parents=True, exist_ok=True)

        # 6. 对新建目录再做一次读写权限校验
        if not os.access(str(base), os.R_OK | os.W_OK):
            raise PermissionError(f"No read/write access to directory: {base!r}")

        # 7. 注册并返回
        self._servers["file"] = (None, str(base))
        return str(base)


    def start_message_queue_service(
            self,
            prompts_num: int,
            n_samples_per_prompt: int,
            metrics: Any = None,
            addition_columns: Union[List[str], None] = None,
            addition_consumers: Union[List[str], None] = None,
            timeout: Union[int, None] = None,
            timeout_interval: Union[int, None] = None,
        ) -> ray.actor.ActorHandle:  
            """
            启动一个 GRPOTransferDock Ray Actor 并注册为 'mq' 存储。
            参数与 GRPOTransferDock.__init__ 完全一致。
            """
            if "mq" in self._servers:
                raise RuntimeError("MessageQueue store has already been started")
            from agent_infer.unified_store.utils.transfer_dock import GRPOTransferDock
            actor = GRPOTransferDock.remote(
                prompts_num,
                n_samples_per_prompt,
                metrics,
                addition_columns,
                addition_consumers,
                timeout,
                timeout_interval
            )  
            self._servers["mq"] = (actor, actor)
            return actor

        
    def get_store_endpoints(self, names: List[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name in names:
            if name not in self._servers:
                raise RuntimeError(f"Store '{name}' has not been started")
            result[name] = self._servers[name][1]
        return result
    


def get_store() -> UnifiedStore:
    mgr = ray.get_actor("UnifiedStoreManager")
    return UnifiedStore(mgr)


