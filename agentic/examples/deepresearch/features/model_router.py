"""DeepResearch-adapted model router.

Reuses ModelRouter and ProgressCheckStrategy from features/model_routing,
but returns ResolvedEndpoint tuples instead of agents.Model instances.
All progress-check calls are synchronous since DeepResearch runs its
ReAct loop in a thread-pool executor.
"""

import json
import re
from dataclasses import dataclass

import requests

from features.model_routing.config import ModelRoutingConfig
from features.model_routing.prompts import THINK_THOUGHT_PREFIX
from features.model_routing.router import ModelRouter
from features.model_routing.strategy import ProgressCheckStrategy, SwitchStrategy

YELLOW = "\033[93m"
RESET = "\033[0m"


@dataclass
class ResolvedEndpoint:
    """The endpoint information returned by the router for a single LLM call."""
    base_url: str
    model_name: str
    api_key: str
    model_type: str  # "small_model" or "large_model"


class DRProgressCheckStrategy(ProgressCheckStrategy):
    """Synchronous progress-check strategy for DeepResearch.

    Inherits should_deescalate / on_escalated / on_deescalated /
    tick_large_round / resolve_fixed_model from ProgressCheckStrategy.
    Only overrides should_escalate to use synchronous requests instead of
    async httpx, and to avoid the dependency on agents.Model / ModelSettings.
    """

    def __init__(
        self,
        config: ModelRoutingConfig,
        trigger_tool_name: str = "think",
        prompt_template: str | None = None,
        progress_check_min_iter: int = 3,
        large_rounds: int = 2,
        fixed_agents: dict[str, str] | None = None,
    ):
        # Use empty dict as default fixed_agents (DR has no PlannerAgent)
        super().__init__(
            config=config,
            trigger_tool_name=trigger_tool_name,
            prompt_template=prompt_template,
            progress_check_min_iter=progress_check_min_iter,
            large_rounds=large_rounds,
            fixed_agents=fixed_agents if fixed_agents is not None else {},
        )

    async def should_escalate(self, agent_name, conversation_history, model, model_settings):
        # Unused — sync version below is used instead.
        return False

    def should_escalate_sync(self, agent_name: str, conversation_history: list[dict]) -> bool:
        """Synchronous progress-check escalation decision.

        Mirrors ProgressCheckStrategy.should_escalate but uses requests.post
        and reads model name from config instead of agents.Model.
        """
        iteration = len(conversation_history) - 1
        if iteration < self._progress_check_min_iter:
            return False

        # Check if the last assistant message contains the trigger tool name
        last_content = ""
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                last_content = str(msg.get("content", ""))
                break

        if f'{self._trigger_tool_name}' not in last_content:
            return False

        history_json = json.dumps(
            conversation_history, ensure_ascii=False, default=str
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

        target_url = self._config.small_model_url
        if not target_url:
            return True  # URL not configured, fail-safe: escalate

        try:
            resp = requests.post(
                url=target_url.rstrip("/") + "/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self._config.small_model_name,
                    "messages": progress_messages,
                    "temperature": 0.0,
                    "max_tokens": 8,
                    "structured_outputs": {"choice": ["TRUE", "FALSE"]},
                },
                timeout=180,
                verify=False,
            )
            resp.raise_for_status()
            resp_json = resp.json()
            progress_raw = resp_json["choices"][0]["message"].get("content") or ""

            choice = progress_raw.strip().upper()
            if choice in {"TRUE", "FALSE"}:
                progress_value = choice
            else:
                m = re.search(
                    r"<value>(.*?)</value>",
                    progress_raw,
                    re.DOTALL | re.IGNORECASE,
                )
                progress_value = m.group(1).strip().upper() if m else "UNKNOWN"

            banner = "=" * 40
            print(f"{YELLOW}{banner} PROGRESS CHECK RAW {banner}{RESET}")
            print(progress_raw)
            print(f"{YELLOW}{banner} PROGRESS VALUE {banner}{RESET}")
            print(f"{YELLOW}<value>{progress_value}</value>{RESET}")
            print(f"{YELLOW}{'=' * (len(banner) * 2 + len(' PROGRESS CHECK RAW '))}{RESET}")

            return not progress_value.strip().upper() == "TRUE"

        except Exception as e:
            print(f"[model_routing] Progress check failed: {e}, fallback to escalate")
            return True


class DRModelRouter(ModelRouter):
    """Model router for DeepResearch.

    Inherits routing state and decision logic (steps 1-5) from ModelRouter,
    but returns ResolvedEndpoint tuples instead of agents.Model instances.
    All operations are synchronous since DeepResearch runs in a thread pool.
    """

    def __init__(self, config: ModelRoutingConfig, strategy: DRProgressCheckStrategy):
        # Pass factory=None — we never use it since resolve() is fully overridden
        super().__init__(config=config, strategy=strategy, factory=None)

    def resolve(self, agent_name: str, messages: list[dict]) -> ResolvedEndpoint:
        """Determine which model endpoint to use for this LLM call.

        Must be called before each call_server() invocation.
        Reimplements ModelRouter.resolve_model() logic but returns
        ResolvedEndpoint instead of Model.
        """
        # 1. Routing disabled → return small model (or default)
        if not self._config.enabled:
            return self._small_endpoint()

        # 2. Fixed model agent → return fixed endpoint
        fixed = self._strategy.resolve_fixed_model(agent_name)
        if fixed is not None:
            self._current_model = fixed
            return self._endpoint_for(fixed)

        # 3. Currently on large model → check deescalation
        if self._current_model == "large_model":
            if self._strategy.should_deescalate():
                print(f"{YELLOW}=== DEESCALATING TO SMALL MODEL ==={RESET}")
                self._current_model = "small_model"
                self._strategy.on_deescalated()
                return self._small_endpoint()
            else:
                self._strategy.tick_large_round()
                return self._large_endpoint()

        # 4. Currently on small model → check escalation
        if self._strategy.should_escalate_sync(agent_name, messages):
            print(f"{YELLOW}=== ESCALATING TO LARGE MODEL ==={RESET}")
            self._current_model = "large_model"
            self._strategy.on_escalated()
            self._strategy.tick_large_round()
            return self._large_endpoint()

        # 5. Stay on small model
        return self._small_endpoint()

    def _small_endpoint(self) -> ResolvedEndpoint:
        return ResolvedEndpoint(
            base_url=self._config.small_model_url,
            model_name=self._config.small_model_name,
            api_key=self._config.small_model_api_key or "EMPTY",
            model_type="small_model",
        )

    def _large_endpoint(self) -> ResolvedEndpoint:
        return ResolvedEndpoint(
            base_url=self._config.large_model_url,
            model_name=self._config.large_model_name,
            api_key=self._config.large_model_api_key or "EMPTY",
            model_type="large_model",
        )

    def _endpoint_for(self, model_type: str) -> ResolvedEndpoint:
        if model_type == "large_model":
            return self._large_endpoint()
        return self._small_endpoint()
