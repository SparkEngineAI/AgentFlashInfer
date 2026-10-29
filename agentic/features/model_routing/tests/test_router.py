"""Tests for ModelRouter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from features.model_routing.config import ModelRoutingConfig
from features.model_routing.factory import ModelFactory
from features.model_routing.router import ModelRouter
from features.model_routing.strategy import ProgressCheckStrategy, SwitchStrategy


# ── Test doubles ──


class StubFactory(ModelFactory):
    """Test factory that returns tagged MagicMock instances."""

    def __init__(self, config: ModelRoutingConfig):
        super().__init__(config)
        self.small_model = MagicMock(_model_type="small", model="pangu_small")
        self.large_model = MagicMock(_model_type="large", model="pangu_large")

    def get_or_create_small_model(self):
        return self.small_model

    def get_or_create_large_model(self):
        return self.large_model


class StubStrategy(SwitchStrategy):
    """Fully controllable strategy for router tests."""

    def __init__(
        self,
        config: ModelRoutingConfig,
        escalate: bool = False,
        deescalate: bool = False,
        fixed: dict | None = None,
    ):
        super().__init__()  # SwitchStrategy ABC has no __init__, but be safe
        self._config = config
        self._escalate = escalate
        self._deescalate = deescalate
        self._fixed = fixed or {}
        self.on_escalated_calls = 0
        self.on_deescalated_calls = 0
        self.tick_calls = 0

    async def should_escalate(self, agent_name, conversation_history, model, model_settings):
        return self._escalate

    def should_deescalate(self):
        return self._deescalate

    def resolve_fixed_model(self, agent_name):
        return self._fixed.get(agent_name)

    def on_escalated(self):
        self.on_escalated_calls += 1

    def on_deescalated(self):
        self.on_deescalated_calls += 1

    def tick_large_round(self):
        self.tick_calls += 1


# ── Fixtures ──


@pytest.fixture
def default_model():
    return MagicMock(model="default_model")


@pytest.fixture
def mock_settings():
    return MagicMock()


@pytest.fixture
def enabled_config():
    return ModelRoutingConfig(
        enabled=True,
        small_model_url="http://7B:7000/v1",
        large_model_url="http://38B:7000/v1",
        small_model_name="pangu_small",
        large_model_name="pangu_large",
    )


# ── Disabled routing ──


@pytest.mark.asyncio
async def test_resolve_model_returns_default_when_disabled(default_model, mock_settings):
    """路由关闭时 resolve_model 返回传入的默认 model"""
    config = ModelRoutingConfig(enabled=False)
    router = ModelRouter(config=config)
    result = await router.resolve_model("Agent", [], default_model, mock_settings)
    assert result is default_model


# ── Fixed model agents ──


@pytest.mark.asyncio
async def test_resolve_model_returns_fixed_model(default_model, mock_settings):
    """resolve_fixed_model 返回固定模型的 agent 不参与路由"""
    config = ModelRoutingConfig(enabled=True, large_model_url="http://38B:7000/v1")
    factory = StubFactory(config=config)
    strategy = StubStrategy(config=config, fixed={"PlannerAgent": "large_model"})
    router = ModelRouter(config=config, strategy=strategy, factory=factory)
    result = await router.resolve_model("PlannerAgent", [], default_model, mock_settings)
    assert result._model_type == "large"


@pytest.mark.asyncio
async def test_resolve_model_fixed_small_model(default_model, mock_settings):
    """固定模型可以是 small_model"""
    config = ModelRoutingConfig(enabled=True, small_model_url="http://7B:7000/v1")
    factory = StubFactory(config=config)
    strategy = StubStrategy(config=config, fixed={"WriterAgent": "small_model"})
    router = ModelRouter(config=config, strategy=strategy, factory=factory)
    result = await router.resolve_model("WriterAgent", [], default_model, mock_settings)
    assert result._model_type == "small"


# ── Default state: small model ──


@pytest.mark.asyncio
async def test_resolve_model_returns_small_by_default(enabled_config, default_model, mock_settings):
    """初始状态应返回小模型"""
    factory = StubFactory(config=enabled_config)
    strategy = StubStrategy(config=enabled_config, escalate=False)
    router = ModelRouter(config=enabled_config, strategy=strategy, factory=factory)
    result = await router.resolve_model("Agent", [], default_model, mock_settings)
    assert result._model_type == "small"
    assert router.current_model == "small_model"


# ── Escalation ──


@pytest.mark.asyncio
async def test_resolve_model_escalates_when_strategy_says_so(enabled_config, default_model, mock_settings, capsys):
    """策略判断需要升级时应切换到大模型"""
    factory = StubFactory(config=enabled_config)
    strategy = StubStrategy(config=enabled_config, escalate=True)
    router = ModelRouter(config=enabled_config, strategy=strategy, factory=factory)
    result = await router.resolve_model(
        "Agent", [], default_model, mock_settings, route_id="seeker-0"
    )
    assert result._model_type == "large"
    assert router.current_model == "large_model"
    assert strategy.on_escalated_calls == 1
    assert (
        "[model_routing] selected=large_model reason=escalated "
        "agent=Agent endpoint=http://38B:7000/v1 route_id=seeker-0"
    ) in capsys.readouterr().out


@pytest.mark.asyncio
async def test_resolve_model_escalation_skips_deescalate_check(enabled_config, default_model, mock_settings):
    """升级当轮不执行降级检查，避免 off-by-one"""
    factory = StubFactory(config=enabled_config)
    strategy = StubStrategy(config=enabled_config, escalate=True, deescalate=True)
    router = ModelRouter(config=enabled_config, strategy=strategy, factory=factory)
    # 当前是小模型，should_escalate=True → 升级
    # should_deescalate=True 但当前不检查（因为 current_model 是 small_model）
    result = await router.resolve_model("Agent", [], default_model, mock_settings)
    assert router.current_model == "large_model"
    assert strategy.on_escalated_calls == 1
    assert strategy.on_deescalated_calls == 0  # 不执行降级


# ── De-escalation ──


@pytest.mark.asyncio
async def test_resolve_model_deescalates_when_strategy_says_so(enabled_config, default_model, mock_settings, capsys):
    """策略判断需要降级时应切回小模型"""
    factory = StubFactory(config=enabled_config)
    strategy = StubStrategy(config=enabled_config, deescalate=True)
    router = ModelRouter(config=enabled_config, strategy=strategy, factory=factory)
    router._current_model = "large_model"  # 模拟已在大模型状态
    result = await router.resolve_model(
        "Agent", [], default_model, mock_settings, route_id="seeker-0"
    )
    assert result._model_type == "small"
    assert router.current_model == "small_model"
    assert strategy.on_deescalated_calls == 1
    assert (
        "[model_routing] selected=small_model reason=deescalated "
        "agent=Agent endpoint=http://7B:7000/v1 route_id=seeker-0"
    ) in capsys.readouterr().out


# ── Stay large + tick ──


@pytest.mark.asyncio
async def test_resolve_model_stays_large_when_not_deescalate(enabled_config, default_model, mock_settings):
    """大模型状态下策略判断不降级时应保持大模型，并递减轮次"""
    strategy = ProgressCheckStrategy(config=enabled_config, large_rounds=2)
    factory = StubFactory(config=enabled_config)
    router = ModelRouter(config=enabled_config, strategy=strategy, factory=factory)
    router._current_model = "large_model"
    strategy._large_rounds_left = 2
    result = await router.resolve_model("Agent", [], default_model, mock_settings)
    assert router.current_model == "large_model"
    assert strategy._large_rounds_left == 1  # tick_large_round 已递减


# ── ProgressCheckStrategy integration with router ──


@pytest.mark.asyncio
async def test_router_escalation_sets_rounds_from_strategy(default_model, mock_settings):
    """升级时应根据策略的 large_rounds 设置大模型轮次"""
    config = ModelRoutingConfig(
        enabled=True,
        small_model_url="http://7B:7000/v1",
        large_model_url="http://38B:7000/v1",
        small_model_name="pangu_small",
        large_model_name="pangu_large",
    )
    strategy = ProgressCheckStrategy(config=config, large_rounds=3)
    factory = StubFactory(config=config)
    router = ModelRouter(config=config, strategy=strategy, factory=factory)

    with patch.object(strategy, "should_escalate", return_value=True):
        await router.resolve_model("Agent", [], default_model, mock_settings)

    assert router.current_model == "large_model"
    assert strategy._large_rounds_left == 3


@pytest.mark.asyncio
async def test_router_escalation_sets_rounds_from_constructor(default_model, mock_settings):
    """升级时应使用策略构造参数指定的 large_rounds"""
    config = ModelRoutingConfig(
        enabled=True,
        small_model_url="http://7B:7000/v1",
        large_model_url="http://38B:7000/v1",
        small_model_name="pangu_small",
        large_model_name="pangu_large",
    )
    strategy = ProgressCheckStrategy(config=config, large_rounds=5)
    factory = StubFactory(config=config)
    router = ModelRouter(config=config, strategy=strategy, factory=factory)

    with patch.object(strategy, "should_escalate", return_value=True):
        await router.resolve_model("Agent", [], default_model, mock_settings)

    assert strategy._large_rounds_left == 5


@pytest.mark.asyncio
async def test_router_full_cycle_small_to_large_to_small(enabled_config, default_model, mock_settings):
    """完整周期：小模型 → 升级 → 大模型 N 轮 → 降级回小模型"""
    strategy = ProgressCheckStrategy(config=enabled_config, large_rounds=2)
    factory = StubFactory(config=enabled_config)
    router = ModelRouter(config=enabled_config, strategy=strategy, factory=factory)

    # 阶段 1：小模型，触发升级
    with patch.object(strategy, "should_escalate", return_value=True):
        result = await router.resolve_model("Agent", [], default_model, mock_settings)
    assert router.current_model == "large_model"
    assert result._model_type == "large"

    # 阶段 2：大模型第 1 轮（不降级）
    result = await router.resolve_model("Agent", [], default_model, mock_settings)
    assert router.current_model == "large_model"

    # 阶段 3：大模型第 2 轮（不降级，tick 递减到 0）
    result = await router.resolve_model("Agent", [], default_model, mock_settings)
    assert router.current_model == "large_model"

    # 阶段 4：should_deescalate=True，降级回小模型
    result = await router.resolve_model("Agent", [], default_model, mock_settings)
    assert router.current_model == "small_model"
    assert result._model_type == "small"


# ── Properties ──


def test_current_model_default():
    """初始 current_model 应为 small_model"""
    router = ModelRouter(config=ModelRoutingConfig())
    assert router.current_model == "small_model"


def test_config_property():
    """config 属性应返回传入的配置"""
    config = ModelRoutingConfig(enabled=True, small_model_url="http://7B:7000/v1")
    router = ModelRouter(config=config)
    assert router.config is config
    assert router.config.small_model_url == "http://7B:7000/v1"


# ── resolve_model_sync ──


def test_resolve_model_sync(enabled_config, default_model, mock_settings):
    """resolve_model_sync 应在无事件循环时正常工作"""
    factory = StubFactory(config=enabled_config)
    strategy = StubStrategy(config=enabled_config, escalate=False)
    router = ModelRouter(config=enabled_config, strategy=strategy, factory=factory)
    result = router.resolve_model_sync("Agent", [], default_model, mock_settings)
    assert result._model_type == "small"


def test_resolve_model_sync_escalation(enabled_config, default_model, mock_settings):
    """resolve_model_sync 应正确处理升级"""
    factory = StubFactory(config=enabled_config)
    strategy = StubStrategy(config=enabled_config, escalate=True)
    router = ModelRouter(config=enabled_config, strategy=strategy, factory=factory)
    result = router.resolve_model_sync("Agent", [], default_model, mock_settings)
    assert result._model_type == "large"
    assert router.current_model == "large_model"
