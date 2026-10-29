from dataclasses import dataclass


@dataclass
class EarlyStopState:
    """每个 agent 的早停状态，由 EarlyStopManager 内部管理"""
    last_thought: str = ""
    last_early_stop_iter: int = -1