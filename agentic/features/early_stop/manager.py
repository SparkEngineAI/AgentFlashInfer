import threading
from typing import Dict, Optional

from .config import EarlyStopConfig
from .state import EarlyStopState
from .checker import EarlyStopChecker, EarlyStopResult


class EarlyStopManager:
    """早停管理器：自主维护 agent_id -> EarlyStopState 映射，对外暴露简洁接口。

    调用方只需传入 agent_id、当前轮次数据，管理器自动查找/持久化对应 agent 的早停状态。
    使用 threading.Lock 保护内部状态，确保多线程/多协程环境下的安全性。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._config = EarlyStopConfig()
        self._states: Dict[str, EarlyStopState] = {}
        self._agent_id = 0

    def check(
        self,
        agent_id: str,
        current_turn: int,
        tool_name: str,
        current_thought: str,
    ) -> EarlyStopResult:
        """执行早停检查，自动查找并更新对应 agent 的状态。"""
        with self._lock:
            state = self._states.get(agent_id, EarlyStopState())
            result = EarlyStopChecker.check(
                config=self._config,
                state=state,
                current_turn=current_turn,
                tool_name=tool_name,
                current_thought=current_thought,
            )
            # 管理器内部持久化更新后的状态
            if result.state is not None:
                self._states[agent_id] = result.state
            return result

    def refresh_config(self, config_overrides: Optional[dict] = None) -> None:
        """刷新早停配置，优先级：config_overrides > 全局 config > 默认值"""
        with self._lock:
            self._config = EarlyStopConfig.from_context(config_overrides)
    
    def refresh_config_from_env(self, early_stop_enabled: bool, early_stop_min_iter: int) -> None:
        """刷新早停配置，优先级：env > 全局 config > 默认值"""
        with self._lock:
            self._config = EarlyStopConfig.from_env(early_stop_enabled, early_stop_min_iter)

    def init_agent(self, agent_id: str) -> None:
        """初始化指定 agent 的早停状态（用于子任务上下文创建时调用）"""
        print(f"################# init_agent: {agent_id}")
        with self._lock:
            self._states[agent_id] = EarlyStopState()

    def remove_agent(self, agent_id: str) -> None:
        """移除指定 agent 的早停状态（用于 agent 运行结束时清理）"""
        print(f"################# remove_agent: {agent_id}")
        with self._lock:
            self._states.pop(agent_id, None)
    
    def gen_agent_id(self) -> str:
        with self._lock:
            self._agent_id += 1
            return str(self._agent_id)