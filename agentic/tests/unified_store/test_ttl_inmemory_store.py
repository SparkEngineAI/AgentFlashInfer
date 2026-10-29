import time
import pytest
from agent_infer.unified_store.inmemory_store import InMemoryStore, TTLConfig

# Use shorter intervals for testing
TEST_CONFIG: TTLConfig = {
    "default_ttl": 1,            # 1 second default TTL
    "default_reset_ttl": False,  # default do not reset on get
    "cleanup_interval": 1        # clean up every second
}

@pytest.fixture
def store():
    # Initialize store with test config
    store = InMemoryStore(config=TEST_CONFIG)
    yield store

    # Allow background thread to finish
    time.sleep(1.1)


def test_put_and_get_before_expiry(store):
    store.put(("ns",), "topic1", "value1")
    # Immediately retrieve
    result = store.get(("ns",), "topic1")
    assert result == "value1"


def test_default_ttl_expires(store):
    store.put(("ns",), "topic2", "value2")
    # Wait for expiry (default 1s)
    time.sleep(1.1)
    result = store.get(("ns",), "topic2")
    assert result is None


def test_custom_ttl_override(store):
    # Override TTL to 2 seconds
    store.put(("ns",), "topic3", "value3", ttl=2)
    time.sleep(1.1)
    # Should still exist
    assert store.get(("ns",), "topic3") == "value3"
    time.sleep(1.1)
    # Now expired
    assert store.get(("ns",), "topic3") is None


def test_reset_ttl_on_get(store):
    # Enable reset for this call
    store.put(("ns",), "topic4", "value4", ttl=1)
    time.sleep(0.6)
    # Retrieve with reset, pushes expiry 1s ahead
    assert store.get(("ns",), "topic4", reset_ttl=True) == "value4"
    # Wait less than original expiry but more than initial pause
    time.sleep(0.6)
    # Should still exist due to reset
    assert store.get(("ns",), "topic4") == "value4"
    # Finally expire
    time.sleep(1.1)
    assert store.get(("ns",), "topic4") is None


def test_no_reset_when_default_false(store):
    store.put(("ns",), "topic5", "value5", ttl=1)
    time.sleep(0.6)
    # Default_reset_ttl is False, so get should not reset
    assert store.get(("ns",), "topic5") == "value5"
    time.sleep(0.6)
    # Should now be expired
    assert store.get(("ns",), "topic5") is None



def test_default_config_no_expiry():
    default_store = InMemoryStore()
    default_store.put(("ns",), "topic_default", "persistent_value")
    time.sleep(3)
    result = default_store.get(("ns",), "topic_default")
    assert result == "persistent_value", "Data with default config should persist indefinitely"


def test_delete_functionality():
    default_store = InMemoryStore()
    default_store.put(("ns",), "topic_delete", "delete_me", ttl=5)
    default_store.delete(("ns",), "topic_delete")
    assert default_store.get(("ns",), "topic_delete") is None, "Data should be deleted immediately"


def test_none_ttl_explicitly_set():
    default_store = InMemoryStore()
    default_store.put(("ns",), "topic_none_ttl", "no_expiry", ttl=None)
    time.sleep(1.5)
    result = default_store.get(("ns",), "topic_none_ttl")
    assert result == "no_expiry", "Data explicitly set with TTL=None should persist indefinitely"


def test_tracemalloc_memory_reclamation_with_long_ttl():
    import time
    import tracemalloc
    # 1) 初始化 store：TTL 30s，cleanup 每 5s
    config = TTLConfig(
        default_ttl=30,
        default_reset_ttl=False,
        cleanup_interval=5
    )
    store = InMemoryStore(config=config)

    # 2) 启动 tracemalloc
    tracemalloc.start()

    # 3) 批量插入 ~200MB 数据 —— 每次都 new 一个独占的 bytearray
    num_entries = 20_000
    start = time.time()
    for i in range(num_entries):
        # 这里每次创建一个新的 ~10KB 块，确保 tracemalloc 能跟踪到所有分配
        buf = bytearray(10_240)
        store.put(("ns",), f"topic_large_{i}", buf)
    insert_time = time.time() - start
    assert insert_time < 30, f"Insertion took too long: {insert_time:.1f}s"

    # 4) 插入后快照
    snap_insert = tracemalloc.take_snapshot()

    # 5) 等待 TTL（30s）+ cleanup（5s）+ 余量（2s）
    time.sleep(37)

    # 6) 清理后快照并停止跟踪
    snap_cleanup = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # 7) 计算总分配量
    def total_alloc(snap):
        return sum(stat.size for stat in snap.statistics('filename'))

    total_insert = total_alloc(snap_insert)
    total_cleanup = total_alloc(snap_cleanup)

    # 8) 打印
    print(
        f"Alloc after insert: {total_insert / (1024 * 1024):.2f} MB; "
        f"Alloc after cleanup: {total_cleanup / (1024 * 1024):.2f} MB"
    )

    # 9) 断言
    assert total_cleanup < total_insert, (
        f"预期 cleanup 后 tracemalloc 分配量下降，"
        f"insert={total_insert}, cleanup={total_cleanup}"
    )