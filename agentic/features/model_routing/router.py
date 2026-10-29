"""Model router — unified entry point for model routing decisions."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents import Model, ModelSettings

    from .config import ModelRoutingConfig
    from .factory import ModelFactory
    from .strategy import SwitchStrategy

from .config import ModelRoutingConfig
from .factory import ModelFactory, OpenAIModelFactory
from .strategy import ProgressCheckStrategy, SwitchStrategy


class ModelRouter:
    """
    模型路由器 — 对外统一入口

    职责：
    1. 维护路由状态（current_model）
    2. 协调 SwitchStrategy（升降级决策）和 ModelFactory（模型创建）
    3. 提供对外的 resolve_model 接口方法
    """

    def __init__(
        self,
        config: ModelRoutingConfig,
        strategy: SwitchStrategy | None = None,
        factory: ModelFactory | None = None,
    ):
        self._config = config
        self._strategy = strategy or ProgressCheckStrategy(config=config)
        self._factory = factory or OpenAIModelFactory(config=config)
        self._current_model: str = "small_model"

    # ============================================================
    # 对外接口
    # ============================================================

    async def resolve_model(
        self,
        agent_name: str,
        conversation_history: list[dict],
        model: Model,
        model_settings: ModelSettings,
        route_id: str = "unknown",
    ) -> Model:
        """返回本轮应使用的模型实例

        在 _run_single_turn 中调用，_get_model() 之后。
        返回的 Model 实例覆盖 _get_model() 的默认结果。
        升级和降级判断统一在此方法中完成。
        """
        # 1. 路由关闭 → 返回默认 model
        if not self._config.enabled:
            return model

        # 2. 固定模型 agent → 直接返回
        fixed = self._strategy.resolve_fixed_model(agent_name)
        if fixed is not None:
            return self._factory.create(fixed)

        # 3. 当前在大模型 → 检查是否降级
        if self._current_model == "large_model":
            if self._strategy.should_deescalate():
                self._current_model = "small_model"
                self._strategy.on_deescalated()
                print(
                    f"[model_routing] selected=small_model reason=deescalated "
                    f"agent={agent_name} endpoint={self._config.small_model_url} "
                    f"route_id={route_id}"
                )
                return self._factory.get_or_create_small_model()
            else:
                self._strategy.tick_large_round()  # 不降级则递减轮次计数
                return self._factory.get_or_create_large_model()

        # 4. 当前在小模型 → 检查是否升级
        if await self._strategy.should_escalate(
            agent_name, conversation_history, model, model_settings
        ):
            self._current_model = "large_model"
            self._strategy.on_escalated()
            print(
                f"[model_routing] selected=large_model reason=escalated "
                f"agent={agent_name} endpoint={self._config.large_model_url} "
                f"route_id={route_id}"
            )
            return self._factory.get_or_create_large_model()

        # 5. 保持小模型
        return self._factory.get_or_create_small_model()

    def resolve_model_sync(
        self,
        agent_name: str,
        conversation_history: list[dict],
        model: Model,
        model_settings: ModelSettings,
        route_id: str = "unknown",
    ) -> Model:
        """同步便捷方法，供同步调用方使用（如 Deep Research）

        内部使用 loop.run_until_complete() 桥接。
        若当前线程无事件循环，则创建临时循环。
        """
        coro = self.resolve_model(
            agent_name,
            conversation_history,
            model,
            model_settings,
            route_id=route_id,
        )
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # ============================================================
    # 辅助方法
    # ============================================================

    @property
    def current_model(self) -> str:
        return self._current_model

    @property
    def config(self) -> ModelRoutingConfig:
        return self._config
