"""Tests for SwitchStrategy and ProgressCheckStrategy."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from features.model_routing.config import ModelRoutingConfig
from features.model_routing.strategy import ProgressCheckStrategy, SwitchStrategy


# ── resolve_fixed_model ──


def test_progress_check_resolves_default_fixed_model():
    """默认配置下 PlannerAgent 应返回 large_model"""
    config = ModelRoutingConfig()
    strategy = ProgressCheckStrategy(config=config)
    assert strategy.resolve_fixed_model("PlannerAgent") == "large_model"


def test_progress_check_resolves_none_for_unknown_agent():
    """不在 fixed_agents 中的 agent 应返回 None"""
    config = ModelRoutingConfig()
    strategy = ProgressCheckStrategy(config=config)
    assert strategy.resolve_fixed_model("SearchAgent") is None


def test_progress_check_custom_fixed_agents():
    """自定义 fixed_agents 应覆盖默认值"""
    config = ModelRoutingConfig()
    custom_strategy = ProgressCheckStrategy(config=config, fixed_agents={"WriterAgent": "small_model"})
    assert custom_strategy.resolve_fixed_model("WriterAgent") == "small_model"
    assert custom_strategy.resolve_fixed_model("PlannerAgent") is None


def test_progress_check_empty_fixed_agents():
    """空 fixed_agents 表示所有 agent 参与正常路由"""
    config = ModelRoutingConfig()
    strategy = ProgressCheckStrategy(config=config, fixed_agents={})
    assert strategy.resolve_fixed_model("PlannerAgent") is None


# ── should_escalate ──
# should_escalate checks conversation_history[-2] for trigger_tool_name,
# so histories must end with [..., assistant_with_trigger, user].


@pytest.mark.asyncio
async def test_should_not_escalate_below_min_iter(short_history, mock_model, mock_model_settings):
    """迭代次数不足时不执行升级检查"""
    config = ModelRoutingConfig()
    strategy = ProgressCheckStrategy(config=config, trigger_tool_name="think", progress_check_min_iter=3)
    result = await strategy.should_escalate(
        "SearchAgent", short_history, mock_model, mock_model_settings
    )
    assert result is False


@pytest.mark.asyncio
async def test_should_not_escalate_without_trigger_tool(mock_model, mock_model_settings):
    """上一轮不包含 trigger_tool_name 时不执行升级检查"""
    config = ModelRoutingConfig()
    strategy = ProgressCheckStrategy(config=config, trigger_tool_name="think", progress_check_min_iter=3)
    # [-2] is assistant without "think"
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Regular response"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "Another regular response"},
        {"role": "user", "content": "q3"},
    ]
    result = await strategy.should_escalate(
        "SearchAgent", history, mock_model, mock_model_settings
    )
    assert result is False


@pytest.mark.asyncio
async def test_should_escalate_when_progress_is_false(
    long_history_with_think, mock_model, mock_model_settings
):
    """小模型返回 FALSE（进展不足）时应升级"""
    config = ModelRoutingConfig(
        enabled=True,
        small_model_url="http://7B:7000/v1",
    )
    strategy = ProgressCheckStrategy(config=config, trigger_tool_name="think", progress_check_min_iter=3)

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "<reason>No progress</reason><value>FALSE</value>"}}]
    }
    fake_response.raise_for_status = MagicMock()

    with patch("features.model_routing.strategy.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = fake_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await strategy.should_escalate(
            "SearchAgent", long_history_with_think, mock_model, mock_model_settings
        )
    assert result is True


@pytest.mark.asyncio
async def test_progress_check_uses_vllm_choice_without_request_template(
    long_history_with_think, mock_model, mock_model_settings
):
    """DDv2/vLLM uses a constrained choice without a request chat template."""
    config = ModelRoutingConfig(
        enabled=True,
        small_model_url="http://7B:7000/v1",
        small_model_template="untrusted request template",
    )
    strategy = ProgressCheckStrategy(
        config=config,
        trigger_tool_name="think",
        progress_check_min_iter=3,
    )
    original_history = [dict(message) for message in long_history_with_think]

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "FALSE"}}]
    }
    fake_response.raise_for_status = MagicMock()

    with patch("features.model_routing.strategy.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = fake_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await strategy.should_escalate(
            "SearchAgent", long_history_with_think, mock_model, mock_model_settings
        )

    payload = mock_client.post.call_args.kwargs["json"]
    assert "chat_template" not in payload
    assert "add_generation_prompt" not in payload
    assert payload["messages"][-1]["role"] == "user"
    assert payload["structured_outputs"] == {"choice": ["TRUE", "FALSE"]}
    assert long_history_with_think == original_history
    assert result is True


@pytest.mark.asyncio
async def test_should_not_escalate_when_progress_is_true(
    long_history_with_think, mock_model, mock_model_settings
):
    """小模型返回 TRUE（进展充足）时不应升级"""
    config = ModelRoutingConfig(
        enabled=True,
        small_model_url="http://7B:7000/v1",
    )
    strategy = ProgressCheckStrategy(config=config, trigger_tool_name="think", progress_check_min_iter=3)

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "<reason>Good progress</reason><value>TRUE</value>"}}]
    }
    fake_response.raise_for_status = MagicMock()

    with patch("features.model_routing.strategy.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = fake_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await strategy.should_escalate(
            "SearchAgent", long_history_with_think, mock_model, mock_model_settings
        )
    assert result is False


@pytest.mark.asyncio
async def test_should_escalate_on_http_error(
    long_history_with_think, mock_model, mock_model_settings
):
    """progress_check 请求异常时应兜底升级到大模型"""
    config = ModelRoutingConfig(
        enabled=True,
        small_model_url="http://7B:7000/v1",
    )
    strategy = ProgressCheckStrategy(config=config, trigger_tool_name="think", progress_check_min_iter=3)

    with patch("features.model_routing.strategy.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Connection error")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await strategy.should_escalate(
            "SearchAgent", long_history_with_think, mock_model, mock_model_settings
        )
    assert result is True


@pytest.mark.asyncio
async def test_should_escalate_when_url_not_configured(
    long_history_with_think, mock_model, mock_model_settings
):
    """small_model_url 未配置时应兜底升级"""
    config = ModelRoutingConfig(
        enabled=True,
        small_model_url="",
    )
    strategy = ProgressCheckStrategy(config=config, trigger_tool_name="think", progress_check_min_iter=3)
    result = await strategy.should_escalate(
        "SearchAgent", long_history_with_think, mock_model, mock_model_settings
    )
    assert result is True


@pytest.mark.asyncio
async def test_custom_trigger_tool_name(mock_model, mock_model_settings):
    """可配置 trigger_tool_name，如 deep research 用 'search'"""
    config = ModelRoutingConfig(
        enabled=True,
        small_model_url="http://7B:7000/v1",
    )
    strategy = ProgressCheckStrategy(config=config, trigger_tool_name="search", progress_check_min_iter=3)

    # [-2] must be assistant with "search"
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": 'Called "search" — searching...'},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": 'Called "search" — more searching...'},
        {"role": "user", "content": "q3"},
    ]

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "<value>FALSE</value>"}}]
    }
    fake_response.raise_for_status = MagicMock()

    with patch("features.model_routing.strategy.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = fake_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await strategy.should_escalate(
            "DeepResearchAgent", history, mock_model, mock_model_settings
        )
    assert result is True


# ── should_deescalate + tick_large_round ──


def test_should_deescalate_after_large_rounds():
    """大模型执行 large_rounds 轮后 should_deescalate 返回 True"""
    config = ModelRoutingConfig()
    strategy = ProgressCheckStrategy(config=config, large_rounds=2)
    strategy.on_escalated()  # _large_rounds_left = 2

    assert strategy.should_deescalate() is False
    strategy.tick_large_round()  # _large_rounds_left = 1
    assert strategy.should_deescalate() is False
    strategy.tick_large_round()  # _large_rounds_left = 0
    assert strategy.should_deescalate() is True


def test_should_deescalate_is_pure_query():
    """should_deescalate 是纯查询，不修改状态"""
    config = ModelRoutingConfig()
    strategy = ProgressCheckStrategy(config=config, large_rounds=2)
    strategy.on_escalated()

    # 多次调用不应改变结果
    for _ in range(3):
        assert strategy.should_deescalate() is False

    # 只有 tick_large_round 修改状态
    strategy.tick_large_round()
    strategy.tick_large_round()
    assert strategy.should_deescalate() is True


# ── on_escalated / on_deescalated ──


def test_on_escalated_resets_rounds():
    """on_escalated 应使用构造参数指定的 large_rounds 重置轮次计数"""
    config = ModelRoutingConfig()
    strategy = ProgressCheckStrategy(config=config, large_rounds=3)
    strategy._large_rounds_left = 0
    strategy.on_escalated()
    assert strategy._large_rounds_left == 3


def test_on_deescalated_resets_rounds():
    """on_deescalated 应重置轮次计数为 0"""
    config = ModelRoutingConfig()
    strategy = ProgressCheckStrategy(config=config, large_rounds=3)
    strategy._large_rounds_left = 2
    strategy.on_deescalated()
    assert strategy._large_rounds_left == 0


def test_tick_large_round_does_not_go_negative():
    """tick_large_round 不应将 _large_rounds_left 变为负数"""
    config = ModelRoutingConfig()
    strategy = ProgressCheckStrategy(config=config, large_rounds=1)
    strategy._large_rounds_left = 0
    strategy.tick_large_round()
    assert strategy._large_rounds_left == 0
