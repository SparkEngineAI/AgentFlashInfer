# tests/agentic_rl/unified_store/test_unified_store.py
import shutil
import socket
import pytest
import ray
import time
from agent_infer.unified_store.unified_store import UnifiedStoreManager, get_store


class CustomClass:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return isinstance(other, CustomClass) and self.x == other.x

def pick_unused_port() -> int:
    """让操作系统分配一个空闲端口"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture(scope="module", autouse=True)
def setup_ray_and_manager():
    # 启动 Ray 并创建全局 Manager
    ray.init()
    mgr = UnifiedStoreManager.remote()
    # 用动态端口启动 InMemoryStoreServer
    port = pick_unused_port()
    addr = ray.get(mgr.start_inmem_store.remote(port))
    # 地址格式应包含 ':' 分隔 IP 和端口
    assert isinstance(addr, str) and ":" in addr, "Store server 地址格式异常"
    # 等待服务器就绪
    time.sleep(1)

    yield addr
    ray.shutdown()
    shutil.rmtree(addr, ignore_errors=True)


def test_add_and_readd_store_and_lifecycle():
    store = get_store()

    # 首次添加 mem 应成功，重复添加应抛 RuntimeError
    store.add_store(["mem"])
    with pytest.raises(RuntimeError):
        store.add_store(["mem"])

    namespace = ("ns1",)
    topic = "topicX"
    payload = {"a": 1, "b": [2, 3]}

    # 在写入前，get 返回 None
    assert store.mem.get(namespace, topic) is None

    # put 返回 None
    assert store.mem.put(namespace, topic, payload) is None

    # get 应直接返回原始字典
    assert store.mem.get(namespace, topic) == payload

    # delete 不应报错，即使删除不存在的项
    store.mem.delete(namespace, topic)
    store.mem.delete(("other_ns",), "no_topic")
    assert store.mem.get(namespace, topic) is None


def test_put_data_type_checks():
    store = get_store()

    # 确保 mem store 已添加过一次，重复 add 会触发 RuntimeError，但不影响后续测试
    store.add_store(["mem"])
    with pytest.raises(RuntimeError):
        store.add_store(["mem"])

    ns = ("ns",)
    topic = "t"

    # 非 dict 应 TypeError
    with pytest.raises(TypeError):
        store.mem.put(ns, topic, [1, 2, 3])

    # dict key 非 str 应 TypeError
    with pytest.raises(TypeError):
        store.mem.put(ns, topic, {123: "bad_key"})


def test_store_singleton_and_attribute_errors():
    s1 = get_store()
    s2 = get_store()
    # get_store 多次调用应使用同一 Manager
    assert s1._mgr == s2._mgr

    # 未 add_store 前，访问属性 mem 抛 AttributeError
    new_store = get_store()
    with pytest.raises(AttributeError):
        _ = new_store.mem

    # 添加后访问未知 store 名称也应抛 AttributeError
    store = get_store()
    store.add_store(["mem"])
    with pytest.raises(AttributeError):
        _ = store.nonexistent


def test_cross_actor_data_sharing():
    @ray.remote
    class ProxyActor:
        def __init__(self):
            self.s = get_store()
            self.s.add_store(["mem"])

        def write(self, ns, tp, d):
            return self.s.mem.put(ns, tp, d)

        def read(self, ns, tp):
            return self.s.mem.get(ns, tp)

        def remove(self, ns, tp):
            return self.s.mem.delete(ns, tp)

    ns = ("actor_ns",)
    topic = "actor_topic"
    data = {"key": "value"}

    writer = ProxyActor.remote()
    reader = ProxyActor.remote()

    # 写入并读取验证
    assert ray.get(writer.write.remote(ns, topic, data)) is None
    assert ray.get(reader.read.remote(ns, topic)) == data

    # 删除并确认
    ray.get(reader.remove.remote(ns, topic))
    assert ray.get(reader.read.remote(ns, topic)) is None




def test_threaded_actor_access():
    import threading

    @ray.remote
    class ThreadedActor:
        def __init__(self):
            self.s = get_store()
            self.s.add_store(["mem"])

        def threaded_put(self, ns, tp, d):
            # 在线程中执行 put
            def task():
                self.s.mem.put(ns, tp, d)

            th = threading.Thread(target=task)
            th.start()
            th.join()
            return True

        def read(self, ns, tp):
            return self.s.mem.get(ns, tp)

    ns = ("thread_ns",)
    topic = "thread_topic"
    payload = {"k": "v"}

    t_actor = ThreadedActor.remote()
    proxy = ThreadedActor.remote()

    # 使用线程池写入
    assert ray.get(t_actor.threaded_put.remote(ns, topic, payload)) is True
    # 主线程中读取
    assert ray.get(proxy.read.remote(ns, topic)) == payload

    # 清理
    ray.get(t_actor.read.remote(ns, topic))
    t_actor._actor_id  # ensure actor stays referenced


def test_multi_actor_threaded_access():
    import threading

    @ray.remote
    class ThreadedActor:
        def __init__(self):
            self.s = get_store()
            self.s.add_store(["mem"])

        def threaded_put(self, ns, tp, d):
            def task():
                self.s.mem.put(ns, tp, d)
            th = threading.Thread(target=task)
            th.start()
            th.join()
            return True

        def threaded_get(self, ns, tp):
            result = {}
            def task():
                result['v'] = self.s.mem.get(ns, tp)
            th = threading.Thread(target=task)
            th.start()
            th.join()
            return result.get('v')

    # 主线程创建两个 Actor，每个在后台线程执行 put/get
    actor1 = ThreadedActor.remote()
    actor2 = ThreadedActor.remote()

    ns1 = ("multi1",)
    topic1 = "t1"
    data1 = {"x": 1}

    ns2 = ("multi2",)
    topic2 = "t2"
    data2 = {"y": 2}

    # actor1 put & get
    assert ray.get(actor1.threaded_put.remote(ns1, topic1, data1)) is True
    assert ray.get(actor1.threaded_get.remote(ns1, topic1)) == data1

    # actor2 put & get
    assert ray.get(actor2.threaded_put.remote(ns2, topic2, data2)) is True
    assert ray.get(actor2.threaded_get.remote(ns2, topic2)) == data2
    assert ray.get(actor2.threaded_get.remote(ns1, topic1)) == data1


def test_actor_nested_threadpools_access():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    @ray.remote
    class NestedPoolActor:
        def __init__(self):
            self.s = get_store()
            self.s.add_store(["mem"])

        def nested_pool_ops(self, namespace, topic, data):
            # 外层线程池
            def outer_task():
                # 内层线程池
                def inner_task():
                    self.s.mem.put(namespace, topic, data)
                    return self.s.mem.get(namespace, topic)
                with ThreadPoolExecutor(max_workers=1) as inner_pool:
                    inner_future = inner_pool.submit(inner_task)
                    return inner_future.result()

            with ThreadPoolExecutor(max_workers=1) as outer_pool:
                outer_future = outer_pool.submit(outer_task)
                return outer_future.result()

    # 实例化同一个 Actor 测试多个 namespace/topic
    actor = NestedPoolActor.remote()
    ns1, tp1, d1 = ("nestedA",), "tA", {"a": 1}
    ns2, tp2, d2 = ("nestedB",), "tB", {"b": 2}

    res1 = ray.get(actor.nested_pool_ops.remote(ns1, tp1, d1))
    res2 = ray.get(actor.nested_pool_ops.remote(ns2, tp2, d2))

    assert res1 == d1
    assert res2 == d2

def test_nested_custom_class_storage():

    store = get_store()
    store.add_store(["mem"])

    namespace = ("nested_custom",)
    topic = "topic_custom"

    payload = {
        "level1": {
            "level2": {
                "custom_obj": CustomClass(42)
            }
        }
    }

    # 存入
    assert store.mem.put(namespace, topic, payload) is None

    # 读取
    retrieved_payload = store.mem.get(namespace, topic)

    # 验证读取结果
    assert isinstance(retrieved_payload, dict)
    assert isinstance(retrieved_payload["level1"], dict)
    assert isinstance(retrieved_payload["level1"]["level2"], dict)
    assert retrieved_payload["level1"]["level2"]["custom_obj"] == CustomClass(42)


def test_put_with_none_removes_key():
    store = get_store()

    # 确保 mem store 已添加过一次，重复 add 会触发 RuntimeError，但不影响后续测试
    store.add_store(["mem"])
    
    # 插入初始数据
    namespace = ("ns1",)
    topic = "topic1"
    payload = {"key1": b"value1", "key2": b"value2"}
    
    # 首先插入数据
    assert store.mem.put(namespace, topic, payload) is None
    assert store.mem.get(namespace, topic) == payload
    
    # 更新数据，key2 设置为 None，应该删除 key2
    updated_payload = {"key1": b"new_value", "key2": None}
    
    # put 时，如果 key2 的值为 None，key2 应该从字典中删除
    assert store.mem.put(namespace, topic, updated_payload) is None
    updated_data = store.mem.get(namespace, topic)
    
    # 确认 key2 被删除
    assert "key2" not in updated_data
    # 确认 key1 被更新
    assert updated_data["key1"] == b"new_value"
    
    # 再次插入，删除 key1，应该只剩下空字典
    final_payload = {"key1": None}
    assert store.mem.put(namespace, topic, final_payload) is None
    final_data = store.mem.get(namespace, topic)
    print(f"[TEST DEBUG] final_data: {final_data}, type: {type(final_data)}")

    # 确认 key1 被删除
    assert "key1" not in final_data
    # 确认最终字典为空
    assert final_data == {}


def test_put_get_delete_async_operations():
    store = get_store()
    # 确保已添加 mem store
    store.add_store(["mem"])

    namespace = ("async_ns",)
    topic = "async_topic"
    payload = {"foo": 123, "bar": [1, 2, 3]}

    # 异步 put
    fut_put = store.mem.put_async(namespace, topic, payload, ttl=5)
    # .result 应存在且返回 None
    assert hasattr(fut_put, "result"), "put_async 返回对象需支持 .result()"
    assert fut_put.result(timeout=2) is None

    # 异步 get
    fut_get = store.mem.get_async(namespace, topic, reset_ttl=False)
    assert hasattr(fut_get, "result"), "get_async 返回对象需支持 .result()"
    got = fut_get.result(timeout=2)
    assert got == payload

    # 异步 delete
    fut_del = store.mem.delete_async(namespace, topic)
    assert fut_del.result(timeout=2) is None

    # 同步验证已删除
    assert store.mem.get(namespace, topic) is None


def test_concurrent_async_put_and_get():
    store = get_store()
    store.add_store(["mem"])

    namespace = ("concurrent_ns",)
    topics = [f"t{i}" for i in range(3)]
    payloads = [{"val": i} for i in range(3)]

    # 并发发起 put_async
    put_futs = [store.mem.put_async(namespace, tp, payloads[i]) for i, tp in enumerate(topics)]
    # 并发等待结果
    results = [f.result(timeout=2) for f in put_futs]
    assert results == [None, None, None]

    # 并发发起 get_async
    get_futs = [store.mem.get_async(namespace, tp) for tp in topics]
    got_list = [f.result(timeout=2) for f in get_futs]
    assert got_list == payloads

