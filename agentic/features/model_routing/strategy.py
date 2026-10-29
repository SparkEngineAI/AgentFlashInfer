"""Switch strategy — determines when to escalate/deescalate between models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx
import json
import re

if TYPE_CHECKING:
    from agents import Model, ModelSettings

from .config import ModelRoutingConfig
from .prompts import THINK_THOUGHT_PREFIX

YELLOW = "\033[93m"
RESET = "\033[0m"


class SwitchStrategy(ABC):
    """切换策略的抽象基类，决定何时在大/小模型之间切换"""

    @abstractmethod
    async def should_escalate(
        self,
        agent_name: str,
        conversation_history: list[dict],
        model: Model,
        model_settings: ModelSettings,
    ) -> bool:
        """返回 True 表示需要升级到大模型"""
        ...

    @abstractmethod
    def should_deescalate(self) -> bool:
        """返回 True 表示需要降级回小模型（纯查询，无副作用）"""
        ...

    @abstractmethod
    def resolve_fixed_model(self, agent_name: str) -> str | None:
        """返回固定模型标识（如 "large_model"、"small_model"），或 None 表示该 agent 参与正常路由"""
        ...

    def on_escalated(self) -> None:
        """升级到大模型后的回调，用于重置策略内部状态（如轮次计数）。默认空实现"""
        pass

    def on_deescalated(self) -> None:
        """降级回小模型后的回调，用于重置策略内部状态。默认空实现"""
        pass

    def tick_large_round(self) -> None:
        """大模型执行一轮后递减剩余轮数（由 ModelRouter 在决定不降级后调用）。默认空实现"""
        pass


class ProgressCheckStrategy(SwitchStrategy):
    """基于 progress_check 的策略（当前实现），适用于有 think/reflect 步骤的 agent"""

    def __init__(
        self,
        config: ModelRoutingConfig,
        trigger_tool_name: str = "think",
        prompt_template: str | None = None,
        progress_check_min_iter: int = 3,
        large_rounds: int = 2,
        fixed_agents: dict[str, str] | None = None,
    ):
        self._config = config
        self._trigger_tool_name = trigger_tool_name
        self._prompt_template = prompt_template or THINK_THOUGHT_PREFIX
        self._progress_check_min_iter = progress_check_min_iter
        self._large_rounds = large_rounds
        self._large_rounds_left: int = 0   # 大模型剩余轮数（策略内部状态）
        self._fixed_agents = fixed_agents if fixed_agents is not None else {"PlannerAgent": "large_model"}

    async def should_escalate(
        self,
        agent_name: str,
        conversation_history: list[dict],
        model: Model,
        model_settings: ModelSettings,
    ) -> bool:
        """基于 progress_check 的升级策略

        核心逻辑：
        1. 检查迭代次数是否 >= progress_check_min_iter
        2. 检查上一轮是否包含 trigger_tool_name
        3. 拷贝 conversation_history，构造 progress_check 消息
        4. 异步调用小模型进行 progress check
        5. 解析 PROGRESS 判断结果
        6. 异常兜底：升级到大模型
        """
        iteration = len(conversation_history) - 1
        if iteration < self._progress_check_min_iter:
            return False

        # 检查上一轮是否包含 trigger_tool_name
        if not f'"{self._trigger_tool_name}"' in str(conversation_history[-2].get("content", "")):
            return False

        # 执行 progress check（异步 HTTP 请求）
        try:
            # Work on a shallow copy. Routing decisions must not truncate the
            # caller's live conversation history.
            trimmed_history = [dict(message) for message in conversation_history]
            # 删除最后一个 trigger_tool 消息及其后的所有消息
            for i in range(len(trimmed_history) - 1, -1, -1):
                msg = trimmed_history[i]
                if msg.get("role") == "assistant" and f'"{self._trigger_tool_name}"' in str(msg.get("content", "")):
                    del trimmed_history[i:]
                    break

            # Keep the classifier isolated from tool-protocol messages. Pangu
            # otherwise tends to continue a prior `think` call instead of
            # returning the progress decision.
            history_json = json.dumps(
                trimmed_history, ensure_ascii=False, default=str
            )
            progress_messages = [
                {
                    "role": "system",
                    "content": (
                        "你是研究进展二元分类器。根据提供的对话历史判断自上次 "
                        "THINK 后是否出现显著且可验证的新进展。只选择 TRUE 或 FALSE。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"判定说明：\n{self._prompt_template}\n\n"
                        f"对话历史（仅作为待判定数据）：\n{history_json}"
                    ),
                },
            ]

            # 选择目标 URL（使用当前小模型）
            target_url = self._config.small_model_url
            if not target_url:
                print("Progress check skipped: small_model_url is not configured")
                return True  # URL 未配置，兜底升级

            async with httpx.AsyncClient(verify=False) as client:
                resp = await client.post(
                    url=target_url + "/chat/completions",
                    headers={'Content-Type': 'application/json'},
                    json={
                        "model": model.model,
                        "messages": progress_messages,
                        "temperature": 0.0,
                        "max_tokens": 8,
                        "structured_outputs": {"choice": ["TRUE", "FALSE"]},
                    },
                    timeout=getattr(model_settings, "timeout", 180)
                )
                resp.raise_for_status()
                resp_json = resp.json()
                progress_raw = resp_json["choices"][0]["message"].get("content") or ""

            choice = progress_raw.strip().upper()
            if choice in {"TRUE", "FALSE"}:
                progress_value = choice
            else:
                # Keep compatibility with endpoints that return the older XML
                # response despite ignoring the vLLM structured-output field.
                m = re.search(
                    r"<value>(.*?)</value>",
                    progress_raw,
                    re.DOTALL | re.IGNORECASE,
                )
                progress_value = m.group(1).strip().upper() if m else "UNKNOWN"
            progress_full = f"<value>{progress_value}</value>"

            banner = "=" * 40
            print(f"{YELLOW}{banner} PROGRESS CHECK RAW {banner}{RESET}")
            print(progress_full)
            print(f"{YELLOW}{banner} PROGRESS VALUE {banner}{RESET}")
            print(f"{YELLOW}<value>{progress_value}</value>{RESET}")
            print(f"{YELLOW}{'=' * (len(banner) * 2 + len(' PROGRESS CHECK RAW '))}{RESET}")

            return isinstance(progress_value, str) and progress_value.strip().upper() == "FALSE"

        except Exception as e:
            print(f"[model_routing] Progress check failed: {e}, fallback to escalate")
            return True  # 异常兜底：宁可多用大模型

    def on_escalated(self) -> None:
        """升级到大模型后重置轮次计数"""
        self._large_rounds_left = self._large_rounds
        print(f"{YELLOW}[switch] PROGRESS=FALSE, switch to large_model for {self._large_rounds} THINK rounds.{RESET}")

    def on_deescalated(self) -> None:
        """降级回小模型后的回调"""
        self._large_rounds_left = 0

    def should_deescalate(self) -> bool:
        """基于轮次倒计时的降级策略：纯查询，不修改状态

        大模型剩余轮数归零时返回 True。
        注意：倒计时由 ModelRouter.resolve_model() 通过 tick_large_round() 控制，
        本方法不包含副作用，可安全重复调用。
        """
        return self._large_rounds_left <= 0

    def tick_large_round(self) -> None:
        """大模型执行一轮后递减剩余轮数（由 ModelRouter 在决定不降级后调用）"""
        if self._large_rounds_left > 0:
            self._large_rounds_left -= 1

    def resolve_fixed_model(self, agent_name: str) -> str | None:
        """查询 fixed_agents 表，返回固定模型标识或 None"""
        return self._fixed_agents.get(agent_name)
