from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional


@dataclass
class CompressionConfig:
    """多轮异步压缩配置"""

    # === 功能开关 ===
    enabled: bool = True

    # === 触发条件 ===
    token_threshold: int = 10              # 触发压缩的最低 token 数
    trigger_tool_names: list = field(default_factory=lambda: ["batch_web_search"])
    max_pending_tasks: int = 1           # 最大在飞压缩任务数
    staleness_threshold: int = 2       # 压缩结果过期轮次阈值

    # === 对话结构 ===
    header_item_count: int = 2           # 对话头部保留项数

    # === LLM 服务 ===
    compress_llm_url: Optional[str] = None
    compress_llm_api_key: Optional[str] = None
    compress_llm_model_name: str = "pangu_auto"
    compress_llm_temperature: float = 0.6
    compress_llm_max_tokens: int = 8192

    # === 工具名 ===
    think_tool_name: str = "think"       # 内部思考工具的名称

    # === 可选回调 ===
    kv_warmup_callback: Optional[Callable[..., Awaitable]] = None
    # KV cache 预热回调，由 Agent 注入。签名: async (messages, ...) -> list
    # 不传则跳过 KV 预热（当前仅 DDv2 的 Pangu 模型需要）