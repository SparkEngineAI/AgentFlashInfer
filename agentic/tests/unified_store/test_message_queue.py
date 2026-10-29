# 使用该单元测试前最好把trainer/__init__.py里面的内容注释掉，避免安装单元测试不需要的依赖
import shutil

import pytest
import ray
import torch
from ray._raylet import ObjectRef

from agent_infer.unified_store.unified_store import UnifiedStoreManager, get_store
from agent_infer.unified_store.utils.metrics import Metric


@pytest.fixture(scope="module", autouse=True)
def setup_ray_and_mq_store():
    # 启动 Ray 和 UnifiedStoreManager，并注册 MQ store
    ray.init()
    mgr = UnifiedStoreManager.remote()
    prompts_num = 4
    n_samples_per_prompt = 2
    metrics = Metric()
    actor = ray.get(mgr.start_message_queue_service.remote(
        prompts_num,
        n_samples_per_prompt,
        metrics,
        None,  # addition_columns
        None,  # addition_consumers
        None,  # timeout
        None,  # timeout_interval
    ))
    yield actor
    ray.shutdown()
    shutil.rmtree(actor, ignore_errors=True)


@pytest.mark.xdist_group(name="message_queue_test")
def test_put_experience_sync_and_get():
    store = get_store()
    store.add_store(["mq"])

    data_dict = {
        "prompts": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        "prompt_length": [torch.tensor([3]), torch.tensor([2])],
    }
    indexes = [0, 1]

    # 同步 put_experience
    assert store.mq.put_experience(True, data_dict, indexes) is None

    # 同步 get_experience，直接拿到 (batch, idxs)
    batch, idxs = store.mq.get_experience(
        True,
        "trainer",
        ["prompts", "prompt_length"],
        2,
        indexes
    )
    assert idxs == indexes
    assert set(batch.keys()) == set(data_dict.keys())
    assert torch.equal(batch["prompts"][0], data_dict["prompts"][0])
    assert torch.equal(batch["prompts"][1], data_dict["prompts"][1])


@pytest.mark.xdist_group(name="message_queue_test")
def test_put_and_get_from_topic():
    store = get_store()
    store.add_store(["mq"])

    data_dict1 = {
        "prompts": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        "prompt_length": [torch.tensor([3]), torch.tensor([2])],
    }
    indexes = [0, 1]

    data_dict2 = {
        "prompts": [torch.tensor([10, 20, 30]), torch.tensor([40, 50])],
        "prompt_length": [torch.tensor([30]), torch.tensor([20])],
    }
    indexes = [0, 1]

    # 同步 put_experience to default topic
    assert store.mq.put_experience(True, data_dict1, indexes) is None

    # 同步 put_experience to test_topic
    assert store.mq.put_experience(True, data_dict2, indexes, topic="test_topic") is None

    # 同步 get_experience，直接拿到 (batch, idxs)
    batch, idxs = store.mq.get_experience(
        True,
        "trainer",
        ["prompts", "prompt_length"],
        2,
        indexes
    )
    assert idxs == indexes
    assert set(batch.keys()) == set(data_dict1.keys())
    assert torch.equal(batch["prompts"][0], data_dict1["prompts"][0])
    assert torch.equal(batch["prompts"][1], data_dict1["prompts"][1])

    # 同步 get_experience，直接拿到 (batch, idxs)
    batch, idxs = store.mq.get_experience(
        True,
        "trainer",
        ["prompts", "prompt_length"],
        2,
        indexes,
        topic="test_topic"
    )
    assert idxs == indexes
    assert set(batch.keys()) == set(data_dict2.keys())
    assert torch.equal(batch["prompts"][0], data_dict2["prompts"][0])
    assert torch.equal(batch["prompts"][1], data_dict2["prompts"][1])


@pytest.mark.xdist_group(name="message_queue_test")
def test_put_experience_async_and_get():
    store = get_store()
    store.add_store(["mq"])

    data_dict = {
        "prompts": [torch.tensor([7, 7, 7]), torch.tensor([8])],
        "prompt_length": [torch.tensor([3]), torch.tensor([1])],
    }
    indexes = [0, 1]

    # 异步 put_experience
    put_ref = store.mq.put_experience(False, data_dict, indexes)
    assert isinstance(put_ref, ObjectRef)
    # get 后应为 None
    assert store.mq.get(put_ref) is None

    # 异步 get_experience 返回 Handle
    handle = store.mq.get_experience(
        False,
        "trainer",
        ["prompts", "prompt_length"],
        2,
        indexes
    )

    # 最终通过 get 拿到 (batch, idxs)
    batch, idxs = store.mq.get(handle)
    assert idxs == indexes
    assert set(batch.keys()) == set(data_dict.keys())
    assert torch.equal(batch["prompts"][0], data_dict["prompts"][0])
    assert torch.equal(batch["prompts"][1], data_dict["prompts"][1])


@pytest.mark.xdist_group(name="message_queue_test")
def test_put_prompts_experience_sync_and_sample():
    store = get_store()
    store.add_store(["mq"])

    batch = {
        "prompts": torch.tensor([[1, 0], [2, 0], [3, 0], [4, 0]], dtype=torch.int64),
        "attention_mask": torch.tensor([[1, 0]] * 4, dtype=torch.int64),
    }
    keys = ["attention_mask"]
    pad_id = 0

    # 同步 put_prompts_experience
    assert store.mq.put_prompts_experience(True, batch, keys, pad_id) is None

    # 同步采样
    batch_out, idxs = store.mq.get_experience(
        True,
        "get_rollout_lengths",
        ["prompts", "attention_mask"],
        2,
        None
    )
    assert isinstance(idxs, list) and len(idxs) == 2
    assert "prompts" in batch_out and "attention_mask" in batch_out


@pytest.mark.xdist_group(name="message_queue_test")
def test_put_prompts_experience_async_and_sample():
    store = get_store()
    store.add_store(["mq"])

    batch = {
        "prompts": torch.tensor([[9, 9], [8, 8], [7, 7], [6, 6]], dtype=torch.int64),
        "attention_mask": torch.tensor([[1, 1]] * 4, dtype=torch.int64),
    }
    keys = ["attention_mask"]
    pad_id = 0

    # 异步 put_prompts_experience
    ref = store.mq.put_prompts_experience(False, batch, keys, pad_id)
    assert isinstance(ref, ObjectRef)
    # get 后应为 None
    assert store.mq.get(ref) is None

    # 异步采样
    handle = store.mq.get_experience(
        False,
        "get_rollout_lengths",
        ["prompts", "attention_mask"],
        2,
        None
    )

    out, idxs = store.mq.get(handle)
    assert isinstance(idxs, list) and len(idxs) == 2
    assert "prompts" in out and "attention_mask" in out


@pytest.mark.xdist_group(name="message_queue_test")
def test_put_and_get_experience_both_async():
    """
    异步 put_experience + 异步 get_experience
    """
    store = get_store()
    store.add_store(["mq"])

    data_dict = {
        "prompts": [torch.tensor([5, 5, 5]), torch.tensor([6, 6])],
        "prompt_length": [torch.tensor([3]), torch.tensor([2])],
    }
    indexes = [0, 1]

    put_ref = store.mq.put_experience(False, data_dict, indexes)
    assert isinstance(put_ref, ObjectRef)
    assert store.mq.get(put_ref) is None

    get_handle = store.mq.get_experience(
        False,
        "trainer",
        ["prompts", "prompt_length"],
        2,
        indexes
    )

    batch, idxs = store.mq.get(get_handle)
    assert idxs == indexes
    assert set(batch.keys()) == set(data_dict.keys())
    assert torch.equal(batch["prompts"][0], data_dict["prompts"][0])
    assert torch.equal(batch["prompts"][1], data_dict["prompts"][1])


@pytest.mark.xdist_group(name="message_queue_test")
def test_consumer_status_and_reset():
    """
    测试 get_consumer_status, all_consumed, all_updated 及 clear 对 consumer 状态的影响
    """
    store = get_store()
    store.add_store(["mq"])
    # 确保状态干净
    store.mq.clear(True)

    # 初始 consumer status 全为 0
    status0 = store.mq.get_consumer_status(True)
    assert isinstance(status0, dict)
    assert "trainer" in status0
    assert status0["trainer"].sum().item() == 0

    # all_consumed 和 all_updated 返回 Tensor(False)
    res_consumed = store.mq.all_consumed(True, "trainer")
    assert isinstance(res_consumed, torch.Tensor)
    assert res_consumed.item() == 0
    res_updated = store.mq.all_updated(True, "prompts")
    assert isinstance(res_updated, torch.Tensor)
    assert res_updated.item() == 0

    # 模拟消费：先 put，再 get_experience(sync=True) 指定 indexes
    data_dict = {
        "prompts": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        "prompt_length": [torch.tensor([3]), torch.tensor([2])],
    }
    indexes = [0, 1]
    store.mq.put_experience(True, data_dict, indexes)
    batch_ref, idxs = store.mq.get_experience(True, "trainer", ["prompts", "prompt_length"], 2, indexes)
    _ = store.mq.get(batch_ref)  # 拆解数据，不做断言

    # Consumer 状态在这两个索引位置被标记
    status1 = store.mq.get_consumer_status(True)
    assert status1["trainer"][0].item() == 1
    assert status1["trainer"][1].item() == 1

    # 清空 consumer status
    store.mq.clear(True)
    status2 = store.mq.get_consumer_status(True)
    assert status2["trainer"].sum().item() == 0

    # 再次 all_consumed/all_updated
    res2_consumed = store.mq.all_consumed(True, "trainer")
    assert isinstance(res2_consumed, torch.Tensor)
    assert res2_consumed.item() == 0
    res2_updated = store.mq.all_updated(True, "prompts")
    assert isinstance(res2_updated, torch.Tensor)
    assert res2_updated.item() == 0
