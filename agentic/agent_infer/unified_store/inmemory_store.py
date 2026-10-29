import pickle
import threading
import time
from concurrent import futures
from typing import Any, Dict, Optional, Tuple, TypedDict

import grpc
import ray

from agent_infer.unified_store.protos import inmemory_store_pb2, inmemory_store_pb2_grpc
from agent_infer.unified_store.store_future import FutureType, StoreFuture


class TTLConfig(TypedDict, total=False):
    # 可选配置类型
    # default_ttl: 默认TTL（秒），不指定时默认为不过期（None）
    default_ttl: int
    # default_reset_ttl: 默认get时是否重置TTL，不指定时默认为 False
    default_reset_ttl: bool
    # cleanup_interval: 后台清理线程的间隔（秒），不指定时默认为 300 秒（5 分钟）
    cleanup_interval: int


class InMemoryStore:
    """
    A simple in-memory store supporting put (with partial updates),
    get (with optional TTL reset), delete operations, and optional TTL per entry.
    支持通过配置指定默认TTL、get自动刷新TTL，以及后台清理频率。
    """

    _PICKLED_NONE = pickle.dumps(None)

    def __init__(
            self,
            config: Optional[TTLConfig] = None  # 可选配置，不指定时使用内置默认
    ) -> None:
        # 应用配置
        cfg = config or {}
        self._default_ttl: Optional[int] = cfg.get("default_ttl", None)
        self._default_reset_ttl: bool = cfg.get("default_reset_ttl", False)
        self._cleanup_interval: int = cfg.get("cleanup_interval", 300)

        # _data maps namespace -> topic -> data payload
        self._data: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        # _expiry maps namespace -> topic -> expiry timestamp
        self._expiry: Dict[Tuple[str, ...], Dict[str, float]] = {}
        # _ttls maps namespace -> topic -> original TTL duration
        self._ttls: Dict[Tuple[str, ...], Dict[str, int]] = {}
        self._lock = threading.RLock()

        # start background cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_expired,
            daemon=True
        )
        self._cleanup_thread.start()

    def put(
            self,
            namespace: Tuple[str, ...],
            topic: str,
            data: Any,
            ttl: Optional[int] = None
    ) -> None:
        """
        Store or update data under the specified namespace and topic.
        If ttl is provided (in seconds) or存在默认TTL，则条目将过期；
        数据合并同前逻辑。
        """
        with self._lock:
            ns = tuple(namespace)
            now = time.time()

            # 初始化命名空间容器
            if ns not in self._data:
                self._data[ns] = {}
                self._expiry[ns] = {}
                self._ttls[ns] = {}

            # 清理已过期条目
            self._remove_if_expired(ns, topic)

            # 合并或替换数据
            existing = self._data[ns].get(topic)
            if isinstance(existing, dict) and isinstance(data, dict):
                for key, value in data.items():
                    if value == self._PICKLED_NONE:
                        existing.pop(key, None)
                    else:
                        existing[key] = value
            else:
                self._data[ns][topic] = data

            # 计算实际TTL
            actual_ttl = ttl if ttl is not None else self._default_ttl
            # 设置或清理过期与TTL元信息
            if actual_ttl is not None:
                self._expiry[ns][topic] = now + actual_ttl
                self._ttls[ns][topic] = actual_ttl
            else:
                self._expiry[ns].pop(topic, None)
                self._ttls[ns].pop(topic, None)

    def get(
            self,
            namespace: Tuple[str, ...],
            topic: str,
            reset_ttl: Optional[bool] = None
    ) -> Optional[Any]:
        """
        Retrieve the data payload under the specified namespace and topic.
        Returns None if not找到或已过期。
        若 reset_ttl 未指定，则使用默认配置；当为 True 时，会重置TTL。
        """
        with self._lock:
            ns = tuple(namespace)
            if ns not in self._data:
                return None
            # 清理过期条目
            if self._remove_if_expired(ns, topic):
                return None

            value = self._data[ns].get(topic)
            # 决定是否重置TTL
            do_reset = reset_ttl if reset_ttl is not None else self._default_reset_ttl
            if value is not None and do_reset:
                ttl = self._ttls.get(ns, {}).get(topic)
                if ttl is not None:
                    self._expiry[ns][topic] = time.time() + ttl
            return value

    def delete(self, namespace: Tuple[str, ...], topic: str) -> None:
        """
        Delete the data under the specified namespace and topic.
        """
        with self._lock:
            ns = tuple(namespace)
            self._data.get(ns, {}).pop(topic, None)
            self._expiry.get(ns, {}).pop(topic, None)
            self._ttls.get(ns, {}).pop(topic, None)
            if ns in self._data and not self._data[ns]:
                self._data.pop(ns, None)
                self._expiry.pop(ns, None)
                self._ttls.pop(ns, None)

    def _remove_if_expired(self, ns: Tuple[str, ...], topic: str) -> bool:
        """
        Check expiry for given entry, remove if expired.
        Returns True if expired and removed.
        """
        exp_map = self._expiry.get(ns, {})
        if topic in exp_map and time.time() >= exp_map[topic]:
            self._data[ns].pop(topic, None)
            exp_map.pop(topic, None)
            self._ttls.get(ns, {}).pop(topic, None)
            if not self._data[ns]:
                self._data.pop(ns, None)
                self._expiry.pop(ns, None)
                self._ttls.pop(ns, None)
            return True
        return False

    def _cleanup_expired(self) -> None:
        """
        Background thread: periodically remove expired entries.
        """
        while True:
            with self._lock:
                now = time.time()
                for ns, topics in list(self._expiry.items()):
                    for topic, exp_time in list(topics.items()):
                        if now >= exp_time:
                            self._data[ns].pop(topic, None)
                            topics.pop(topic, None)
                            self._ttls.get(ns, {}).pop(topic, None)
                    if ns in self._data and not self._data[ns]:
                        self._data.pop(ns, None)
                        self._expiry.pop(ns, None)
                        self._ttls.pop(ns, None)
            time.sleep(self._cleanup_interval)  # 使用配置的清理间隔


# Ray remote server actor for gRPC InMemoryStore
@ray.remote(num_cpus=1)
class InMemoryStoreServer:
    def __init__(self, port: int, config: Optional[TTLConfig] = None):
        self._port = port
        # 存储可选配置，不指定时将在 InMemoryStore 中使用默认配置
        self._config = config

    def get_address(self) -> str:
        ip = ray.util.get_node_ip_address()
        return f"{ip}:{self._port}"

    def run(self) -> None:
        """
        Start gRPC server exposing InMemoryStore service directly.
        使用配置或默认值：
          - default_ttl = config.default_ttl or 300 秒
          - default_reset_ttl = config.default_reset_ttl or False
          - cleanup_interval = config.cleanup_interval or 300 秒
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        # 构造 InMemoryStore，传入可选配置
        store = InMemoryStore(config=self._config)
        servicer = InMemoryStoreServicer(store)
        inmemory_store_pb2_grpc.add_InMemoryStoreServicer_to_server(servicer, server)
        server.add_insecure_port(f"0.0.0.0:{self._port}")

        server.start()
        server.wait_for_termination()


class InMemoryStoreServicer(inmemory_store_pb2_grpc.InMemoryStoreServicer):
    def __init__(self, store: InMemoryStore):
        self._store = store

    def grpcPut(self, request, context):
        namespace = tuple(request.namespace)
        topic = request.topic
        data: Dict[str, bytes] = dict(request.data)
        # Protobuf v3 整数字段默认值为 0，只有正数才当作有效 TTL
        raw_ttl = request.ttl
        actual_ttl = raw_ttl if raw_ttl > 0 else None
        self._store.put(namespace, topic, data, ttl=actual_ttl)
        return inmemory_store_pb2.putResponse()

    def grpcGet(self, request, context):
        namespace = tuple(request.namespace)
        topic = request.topic
        reset = getattr(request, 'reset_ttl', None)
        result = self._store.get(namespace, topic, reset_ttl=reset)

        resp = inmemory_store_pb2.getResponse()
        if result is not None:
            resp.exists = True
            resp.data.update(result)
        else:
            resp.exists = False
        return resp

    def grpcDelete(self, request, context):
        namespace = tuple(request.namespace)
        topic = request.topic
        self._store.delete(namespace, topic)
        return inmemory_store_pb2.deleteResponse()


class InMemoryStoreClient:
    """
    Client wrapper for InMemoryStore gRPC service.
    用户实际调用的接口。参数的合法性检查应该都在这里完成, 而不是grpc之后。
    同一客户端里同时提供：
    1) 同步阻塞的 put/get/delete
    2) 基于 Future 的异步接口 put_async/get_async/delete_async
    """

    def __init__(self, addr: str) -> None:
        self._channel = grpc.insecure_channel(addr)
        self._stub = inmemory_store_pb2_grpc.InMemoryStoreStub(self._channel)

    def put(
            self,
            namespace: Tuple[str, ...],
            topic: str,
            data: Dict[str, Any],
            ttl: Optional[int] = None
    ) -> None:
        """
        Serialize each value in 'data' with pickle and invoke remote Put.
        ttl: Optional[int] - seconds (>0); if None, server default used.
        """
        self._validate_namespace(namespace)
        if not isinstance(data, dict):
            raise TypeError("Client put(): data must be a dict[str, Any]")
        for k in data.keys():
            if not isinstance(k, str):
                raise TypeError("Client put(): all keys in data must be strings")
        # TTL 校验
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be > 0")
        # 每个 value 分别 pickle
        data_bytes = {k: pickle.dumps(v) for k, v in data.items()}
        req_kwargs: Dict[str, Any] = {
            'namespace': list(namespace),
            'topic': topic,
            'data': data_bytes
        }
        if ttl is not None:
            req_kwargs['ttl'] = ttl
        req = inmemory_store_pb2.putRequest(**req_kwargs)
        self._stub.grpcPut(req)

    def get(
            self,
            namespace: Tuple[str, ...],
            topic: str,
            reset_ttl: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Invoke remote Get, optionally reset TTL, unpickle each value and return result or None.
        reset_ttl: Optional[bool] - if True, server will reset TTL countdown.
        """
        self._validate_namespace(namespace)
        req_kwargs = {
            'namespace': list(namespace),
            'topic': topic
        }
        if reset_ttl is not None:
            req_kwargs['reset_ttl'] = reset_ttl
        req = inmemory_store_pb2.getRequest(**req_kwargs)
        resp = self._stub.grpcGet(req)
        if resp.exists:
            return {k: pickle.loads(v) for k, v in resp.data.items()}
        return None

    def delete(self, namespace: Tuple[str, ...], topic: str) -> None:
        """
        Invoke remote Delete.
        """
        self._validate_namespace(namespace)
        req = inmemory_store_pb2.deleteRequest(
            namespace=list(namespace),
            topic=topic
        )
        self._stub.grpcDelete(req)

    def __del__(self):
        self._channel.close()

    def put_async(
            self,
            namespace: Tuple[str, ...],
            topic: str,
            data: Dict[str, Any],
            ttl: Optional[int] = None
    ) -> StoreFuture:
        """
        异步 put：立即返回 StoreFuture，future.get() -> None
        """
        self._validate_namespace(namespace)
        if not isinstance(data, dict):
            raise TypeError("Client put_async(): data must be a dict[str, Any]")
        for k in data.keys():
            if not isinstance(k, str):
                raise TypeError("Client put_async(): all keys in data must be strings")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be > 0")

        data_bytes = {k: pickle.dumps(v) for k, v in data.items()}
        req = inmemory_store_pb2.putRequest(
            namespace=list(namespace),
            topic=topic,
            data=data_bytes,
            ttl=ttl or 0
        )
        fut: FutureType = self._stub.grpcPut.future(req)
        return StoreFuture(fut, extractor=lambda _: None)

    def get_async(
            self,
            namespace: Tuple[str, ...],
            topic: str,
            reset_ttl: Optional[bool] = None
    ) -> StoreFuture:
        """
        异步 get：立即返回 StoreFuture，future.get() -> Optional[Dict[str, Any]]
        """
        self._validate_namespace(namespace)
        req_kwargs = {
            'namespace': list(namespace),
            'topic': topic
        }
        if reset_ttl is not None:
            req_kwargs['reset_ttl'] = reset_ttl
        req = inmemory_store_pb2.getRequest(**req_kwargs)

        fut: FutureType = self._stub.grpcGet.future(req)

        def extractor(resp):
            if not resp.exists:
                return None
            return {k: pickle.loads(v) for k, v in resp.data.items()}

        return StoreFuture(fut, extractor=extractor)

    def delete_async(
            self,
            namespace: Tuple[str, ...],
            topic: str
    ) -> StoreFuture:
        """
        异步 delete：立即返回 StoreFuture，future.get() -> None
        """
        self._validate_namespace(namespace)
        req = inmemory_store_pb2.deleteRequest(
            namespace=list(namespace),
            topic=topic
        )
        fut: FutureType = self._stub.grpcDelete.future(req)
        return StoreFuture(fut, extractor=lambda _: None)

    @staticmethod
    def _validate_namespace(namespace: Tuple[str, ...]) -> None:
        if not namespace:
            raise ValueError("Namespace cannot be empty.")
        for label in namespace:
            if not isinstance(label, str) or not label:
                raise ValueError(f"Invalid namespace label: '{label}'")
            if "." in label:
                raise ValueError(f"Namespace labels cannot contain periods: '{label}'")
