from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from features.compression.adapters.ddv2 import Ddv2ConversationAdapter


# ============================================================
# Mock 辅助 — 注入 agents.items 和 agents.run 模块
# ============================================================


class FakeToolCallItem:
    """模拟 agents.items.ToolCallItem"""

    def __init__(self, name: str = "", arguments: str = ""):
        self.raw_item = SimpleNamespace(name=name, arguments=arguments)

    def to_input_item(self):
        return {"type": "function", "name": self.raw_item.name, "arguments": self.raw_item.arguments}


class FakeMessageItem:
    """模拟 agents.items.MessageOutputItem"""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_input_item(self):
        return {"role": self.role, "content": self.content}


# 构建 mock agents 包结构
def _setup_mock_agents():
    """注入 mock agents.items 和 agents.run 到 sys.modules"""
    if "agents" not in sys.modules:
        agents_mod = types.ModuleType("agents")
        sys.modules["agents"] = agents_mod
    if "agents.items" not in sys.modules:
        items_mod = types.ModuleType("agents.items")
        items_mod.ToolCallItem = FakeToolCallItem
        sys.modules["agents.items"] = items_mod
        sys.modules["agents"].items = items_mod
    if "agents.run" not in sys.modules:
        run_mod = types.ModuleType("agents.run")
        mock_runner = MagicMock()
        mock_runner._maybe_filter_model_input = AsyncMock()
        run_mod.AgentRunner = mock_runner
        sys.modules["agents.run"] = run_mod
        sys.modules["agents"].run = run_mod


_setup_mock_agents()


def _make_filtered_result(input_items, instructions="system"):
    """模拟 ModelInputData"""
    return SimpleNamespace(input=input_items, instructions=instructions)


def _make_adapter(**kwargs):
    defaults = dict(
        chat_converter=MagicMock(),
        agent=MagicMock(),
        run_config=MagicMock(),
        context_wrapper=MagicMock(),
        system_prompt="你是一个压缩助手",
        think_tool_name="think",
    )
    defaults.update(kwargs)
    return Ddv2ConversationAdapter(**defaults)


def _make_items(n_turns: int = 3) -> list:
    """构造 n 轮对话历史：header(2) + 交替的 tool_call/message"""
    items = [
        FakeMessageItem("system", "system prompt"),
        FakeMessageItem("user", "user query"),
    ]
    for i in range(n_turns):
        items.append(FakeToolCallItem("think", json.dumps({"thought": f"思考{i}"})))
        items.append(FakeToolCallItem("batch_web_search", json.dumps({"query": f"查询{i}"})))
        items.append(FakeMessageItem("tool", f"搜索结果{i}"))
    return items


# ============================================================
# to_messages
# ============================================================


class TestToMessages:
    """测试 to_messages 方法"""

    @pytest.mark.asyncio
    async def test_basic_conversion(self):
        """基本转换：items → messages，包含 system_prompt"""
        items = _make_items(n_turns=2)
        input_items = [item.to_input_item() for item in items]

        mock_filter_result = _make_filtered_result(input_items)
        converter = MagicMock()
        converter.items_to_messages = MagicMock(
            side_effect=lambda x: [{"role": "user", "content": str(x[0])}]
        )

        adapter = _make_adapter(chat_converter=converter)

        runner = sys.modules["agents.run"].AgentRunner
        runner._maybe_filter_model_input = AsyncMock(return_value=mock_filter_result)

        messages = await adapter.to_messages(items)

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "你是一个压缩助手"
        assert converter.items_to_messages.called

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self):
        """自定义 system_prompt 覆盖默认值"""
        items = _make_items(n_turns=1)
        input_items = [item.to_input_item() for item in items]
        mock_filter_result = _make_filtered_result(input_items)

        converter = MagicMock()
        converter.items_to_messages = MagicMock(return_value=[{"role": "user", "content": "hi"}])

        adapter = _make_adapter(chat_converter=converter)

        runner = sys.modules["agents.run"].AgentRunner
        runner._maybe_filter_model_input = AsyncMock(return_value=mock_filter_result)

        messages = await adapter.to_messages(items, system_prompt="自定义指令")

        assert messages[0]["content"] == "自定义指令"

    @pytest.mark.asyncio
    async def test_no_system_prompt(self):
        """system_prompt 为空时不插入 system 消息"""
        items = _make_items(n_turns=1)
        input_items = [item.to_input_item() for item in items]
        mock_filter_result = _make_filtered_result(input_items)

        converter = MagicMock()
        converter.items_to_messages = MagicMock(return_value=[{"role": "user", "content": "hi"}])

        adapter = _make_adapter(chat_converter=converter, system_prompt="")

        runner = sys.modules["agents.run"].AgentRunner
        runner._maybe_filter_model_input = AsyncMock(return_value=mock_filter_result)

        messages = await adapter.to_messages(items, system_prompt="")

        assert all(m.get("role") != "system" for m in messages)


# ============================================================
# find_last_tool_call
# ============================================================


class TestFindLastToolCall:
    """测试 find_last_tool_call 方法"""

    def test_found(self):
        """正常查找：找到最后一个匹配的工具调用"""
        adapter = _make_adapter()
        items = _make_items(n_turns=3)

        result = adapter.find_last_tool_call(items, "think", boundary=len(items))

        assert result is not None
        assert isinstance(items[result], FakeToolCallItem)
        assert items[result].raw_item.name == "think"

    def test_not_found(self):
        """查找不存在的工具名 — 返回 None"""
        adapter = _make_adapter()
        items = _make_items(n_turns=2)

        result = adapter.find_last_tool_call(items, "nonexistent_tool", boundary=len(items))

        assert result is None

    def test_boundary_excludes_later_items(self):
        """boundary 之后的项不参与查找"""
        adapter = _make_adapter()
        items = _make_items(n_turns=3)

        # boundary=3 只看前 3 项（header + 第一个 think）
        result = adapter.find_last_tool_call(items, "batch_web_search", boundary=3)

        # 前 3 项没有 batch_web_search
        assert result is None

    def test_returns_last_match(self):
        """多个匹配时返回最后一个"""
        adapter = _make_adapter()
        items = _make_items(n_turns=3)

        result = adapter.find_last_tool_call(items, "think", boundary=len(items))

        # 手动找最后一个 think
        last_think_idx = None
        for i, item in enumerate(items):
            if isinstance(item, FakeToolCallItem) and item.raw_item.name == "think":
                last_think_idx = i
        assert result == last_think_idx


# ============================================================
# is_tool_call
# ============================================================


class TestIsToolCall:
    """测试 is_tool_call 方法"""

    def test_tool_call_item(self):
        """ToolCallItem 返回 True"""
        adapter = _make_adapter()
        item = FakeToolCallItem("think", "{}")

        assert adapter.is_tool_call(item) is True

    def test_message_item(self):
        """非 ToolCallItem 返回 False"""
        adapter = _make_adapter()
        item = FakeMessageItem("user", "hello")

        assert adapter.is_tool_call(item) is False


# ============================================================
# set_tool_call_content
# ============================================================


class TestSetToolCallContent:
    """测试 set_tool_call_content 方法"""

    def test_sets_thought_content(self):
        """设置 think 工具的 arguments 为摘要内容"""
        adapter = _make_adapter()
        item = FakeToolCallItem("think", json.dumps({"thought": "旧内容"}))

        adapter.set_tool_call_content(item, "新的摘要内容")

        parsed = json.loads(item.raw_item.arguments)
        assert parsed["thought"] == "新的摘要内容"

    def test_ensure_ascii_false(self):
        """中文内容不被转义"""
        adapter = _make_adapter()
        item = FakeToolCallItem("think", "{}")

        adapter.set_tool_call_content(item, "中文摘要")

        assert "中文摘要" in item.raw_item.arguments
        assert "\\u" not in item.raw_item.arguments


# ============================================================
# reconstruct_history
# ============================================================


class TestReconstructHistory:
    """测试 reconstruct_history 方法"""

    def test_basic_reconstruct(self):
        """基本重构：保留头部 + 工具调用骨架 + 边界之后"""
        adapter = _make_adapter()
        items = _make_items(n_turns=3)
        original_len = len(items)

        result = adapter.reconstruct_history(
            items, boundary=original_len - 3, keep_header_count=2
        )

        # 头部 2 项保留
        assert result[:2] == items[:2]
        # 尾部 3 项保留
        assert result[-3:] == items[-3:]

    def test_header_only(self):
        """boundary=keep_header_count 时只保留头部和尾部"""
        adapter = _make_adapter()
        items = _make_items(n_turns=3)

        result = adapter.reconstruct_history(items, boundary=2, keep_header_count=2)

        # 只有头部 + 边界之后
        assert result[:2] == items[:2]
        assert result[2:] == items[2:]

    def test_tool_calls_skeleton(self):
        """工具调用骨架：boundary 之前的 tool_call 保留（跳过第一个）"""
        adapter = _make_adapter()
        items = _make_items(n_turns=3)

        boundary = len(items) - 3
        result = adapter.reconstruct_history(items, boundary=boundary, keep_header_count=2)

        # boundary 之前的 tool_calls（跳过第一个）
        tool_calls_before_boundary = [
            item for i, item in enumerate(items[:boundary])
            if isinstance(item, FakeToolCallItem)
        ]
        # 结果中应包含 tool_calls_before_boundary[1:]
        for tc in tool_calls_before_boundary[1:]:
            assert tc in result


# ============================================================
# __init__ 导入
# ============================================================


class TestAdapterImport:
    """测试 adapters/__init__.py 的导出"""

    def test_import_ddv2_adapter(self):
        """从 adapters 包导入 Ddv2ConversationAdapter"""
        from features.compression.adapters import Ddv2ConversationAdapter as Imported

        assert Imported is Ddv2ConversationAdapter