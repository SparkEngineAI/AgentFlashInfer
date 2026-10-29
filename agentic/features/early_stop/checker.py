from dataclasses import dataclass
from typing import Optional

from openai.types.responses import EasyInputMessage

from .config import EarlyStopConfig
from .state import EarlyStopState
from .constants import (
    REPEAT_OUTPUT_PROMPT,
    MIN_ITER_ANALYSIS_PROMPT,
    LOG_MARKER_REPEAT,
    LOG_MARKER_MIN_ITER,
)


@dataclass
class EarlyStopResult:
    """早停检查结果"""
    should_inject: bool = False
    injected_message: Optional[EasyInputMessage] = None
    state: Optional[EarlyStopState] = None
    reason: str = "none"  # "repeat" | "min_iter" | "none"


class EarlyStopChecker:
    """无状态检查器：根据配置、状态和当前轮次数据判断是否触发早停。
    不修改外部状态，通过 EarlyStopResult 返回更新后的状态由调用方持久化。"""

    @staticmethod
    def check(
        config: EarlyStopConfig,
        state: EarlyStopState,
        current_turn: int,
        tool_name: str,
        current_thought: str,
    ) -> EarlyStopResult:
        
        # print(f"################ check current_turn:{current_turn}, tool_name:{tool_name}, config :{config}, state:{state}")
        if not config.enabled or tool_name != "think":
            return EarlyStopResult(state=state, reason="none")

        
        # 场景1：重复输出检测
        if state.last_thought == current_thought:
            print(f"[early_stop] {LOG_MARKER_REPEAT}")
            new_state = EarlyStopState(
                last_thought=state.last_thought,
                last_early_stop_iter=current_turn,
            )
            print(f"[early_stop] repeat output detected, iter={current_turn}")
            return EarlyStopResult(
                should_inject=True,
                injected_message=EasyInputMessage(content=REPEAT_OUTPUT_PROMPT, role="user", type="message"),
                state=new_state,
                reason="repeat",
            )

        # 场景2：最小迭代次数检查
        if current_turn >= config.min_iter and current_turn - state.last_early_stop_iter > 1:
            print(f"[early_stop] {LOG_MARKER_MIN_ITER}")
            new_state = EarlyStopState(
                last_thought=current_thought,
                last_early_stop_iter=current_turn,
            )
            print(f"[early_stop] min_iter reached, analyze progress, iter={current_turn}")
            return EarlyStopResult(
                should_inject=True,
                injected_message=EasyInputMessage(content=MIN_ITER_ANALYSIS_PROMPT, role="user", type="message"),
                state=new_state,
                reason="min_iter",
            )

        # 无早停触发，仅更新 last_thought
        new_state = EarlyStopState(
            last_thought=current_thought,
            last_early_stop_iter=state.last_early_stop_iter,
        )
        return EarlyStopResult(state=new_state, reason="none")