from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from features.compression.config import CompressionConfig
from features.compression.manager import CompressionManager


class SimpleToolCallItem:
    """测试用对话项 — 模拟工具调用"""

    def __init__(self, name: str, arguments: str = ""):
        self.raw_item = SimpleNamespace(name=name, arguments=arguments)


class SimpleMessageItem:
    """测试用对话项 — 模拟消息"""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class StubConversationAdapter:
    """轻量测试适配器 — 无 DDv2 依赖"""

    def __init__(self, think_tool_name: str = "think"):
        self._think_tool_name = think_tool_name

    async def to_messages(self, items, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for item in items:
            if hasattr(item, "raw_item"):
                messages.append(
                    {"role": "assistant", "content": item.raw_item.arguments}
                )
            else:
                messages.append({"role": item.role, "content": item.content})
        return messages

    def find_last_tool_call(self, items, tool_name, boundary):
        for i, item in enumerate(items[:boundary][::-1]):
            if (
                hasattr(item, "raw_item")
                and item.raw_item.name == tool_name
            ):
                return len(items[:boundary]) - 1 - i
        return None

    def is_tool_call(self, item):
        return hasattr(item, "raw_item")

    def set_tool_call_content(self, item, content):
        item.raw_item.arguments = json.dumps(
            {"thought": content}, ensure_ascii=False
        )

    def reconstruct_history(self, items, boundary, keep_header_count):
        tool_calls = [
            item
            for i, item in enumerate(items)
            if hasattr(item, "raw_item") and i < boundary
        ]
        return items[:keep_header_count] + tool_calls[1:] + items[boundary:]


def make_llm_output(summary_text: str, action_text: str) -> str:
    """构造模拟 LLM 输出"""
    return f"<summary>{summary_text}</summary>\n<action>{action_text}</action>"


def make_items(n_turns: int = 5) -> list:
    """构造 n 轮对话历史：header(2) + 交替的 tool_call/message"""
    items = [
        SimpleMessageItem("system", "system prompt"),
        SimpleMessageItem("user", "user query"),
    ]
    for i in range(n_turns):
        items.append(
            SimpleToolCallItem("think", json.dumps({"thought": f"思考{i}"}))
        )
        items.append(
            SimpleToolCallItem(
                "batch_web_search", json.dumps({"query": f"查询{i}"})
            )
        )
        items.append(SimpleMessageItem("tool", f"搜索结果{i}"))
    return items


@pytest.fixture
def adapter():
    return StubConversationAdapter()


@pytest.fixture
def config():
    return CompressionConfig(
        enabled=True,
        token_threshold=5000,
        trigger_tool_names=["batch_web_search"],
        max_pending_tasks=1,
        staleness_threshold=100,
        header_item_count=2,
    )


@pytest.fixture
def manager(config, adapter):
    return CompressionManager(config, adapter)