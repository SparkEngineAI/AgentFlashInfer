from __future__ import annotations

import json5
import pytest

from features.compression.adapters.deepresearch import DeepResearchConversationAdapter


TC_START = '<tool_call>'
TC_END = '</tool_call>'
TK_START = '<think>'
TK_END = '</think>'


def _make_adapter(**kwargs):
    defaults = dict(
        system_prompt="你是一个压缩助手",
        think_tag="think",
    )
    defaults.update(kwargs)
    return DeepResearchConversationAdapter(**defaults)


def _make_messages(n_turns: int = 3) -> list:
    NL = chr(10)
    messages = [
        {"role": "system", "content": "You are a deep research assistant."},
        {"role": "user", "content": "What is the impact of AI?"},
    ]
    for i in range(n_turns):
        think_part = TK_START + "思考" + str(i) + TK_END
        tool_json = '{"name": "search", "arguments": {"query": ["query{i}"]}}'
        tool_part = TC_START + NL + tool_json + NL + TC_END
        messages.append({
            "role": "assistant",
            "content": think_part + NL + tool_part,
        })
        messages.append({
            "role": "user",
            "content": "Observation:" + NL + "搜索结果" + str(i),
        })
    return messages


def _make_messages_with_think_only(n_turns: int = 3) -> list:
    NL = chr(10)
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "question"},
    ]
    for i in range(n_turns):
        think_part = TK_START + "思考" + str(i) + TK_END
        answer_part = "<answer>答案" + str(i) + "</answer>"
        messages.append({
            "role": "assistant",
            "content": think_part + NL + answer_part,
        })
    return messages


class TestInit:
    def test_default_markers(self):
        adapter = _make_adapter()
        assert adapter._system_prompt == "你是一个压缩助手"
        assert adapter._think_start == TK_START
        assert adapter._think_end == TK_END
        assert adapter._tool_call_start == TC_START
        assert adapter._tool_call_end == TC_END

    def test_custom_think_tag(self):
        adapter = _make_adapter(think_tag="reasoning")
        assert adapter._think_start == "<reasoning>"
        assert adapter._think_end == "</reasoning>"

    def test_custom_tool_call_markers(self):
        adapter = _make_adapter(tool_call_start="<<", tool_call_end=">>")
        assert adapter._tool_call_start == "<<"
        assert adapter._tool_call_end == ">>"


class TestToMessages:
    @pytest.mark.asyncio
    async def test_basic_passthrough(self):
        adapter = _make_adapter()
        messages = _make_messages(n_turns=1)
        result = await adapter.to_messages(messages)
        assert len(result) == len(messages)
        assert result is not messages
        # system 消息被替换为 adapter 的 system_prompt，其余不变
        assert result[0]["role"] == "system"
        for r, m in zip(result[1:], messages[1:]):
            assert r == m

    @pytest.mark.asyncio
    async def test_system_prompt_replaced(self):
        adapter = _make_adapter()
        messages = _make_messages(n_turns=1)
        result = await adapter.to_messages(messages)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "你是一个压缩助手"
        assert sum(1 for m in result if m["role"] == "system") == 1

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self):
        adapter = _make_adapter()
        messages = _make_messages(n_turns=1)
        result = await adapter.to_messages(messages, system_prompt="自定义指令")
        assert result[0]["content"] == "自定义指令"

    @pytest.mark.asyncio
    async def test_no_system_prompt(self):
        adapter = _make_adapter(system_prompt="")
        messages = _make_messages(n_turns=1)
        result = await adapter.to_messages(messages, system_prompt="")
        # system_prompt 为空时，不修改原有 system 消息
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a deep research assistant."

    @pytest.mark.asyncio
    async def test_does_not_modify_original(self):
        adapter = _make_adapter()
        messages = _make_messages(n_turns=1)
        original_len = len(messages)
        await adapter.to_messages(messages, system_prompt="new")
        assert len(messages) == original_len


class TestFindLastToolCall:
    def test_find_think_uses_deepresearch_tags(self):
        adapter = _make_adapter()
        messages = _make_messages_with_think_only(n_turns=3)

        result = adapter.find_last_tool_call(
            messages, "think", boundary=len(messages)
        )

        assert result == len(messages) - 1

    def test_found(self):
        adapter = _make_adapter()
        messages = _make_messages(n_turns=3)
        result = adapter.find_last_tool_call(messages, "search", boundary=len(messages))
        assert result is not None
        assert messages[result]["role"] == "assistant"

    def test_not_found(self):
        adapter = _make_adapter()
        messages = _make_messages(n_turns=2)
        result = adapter.find_last_tool_call(messages, "nonexistent_tool", boundary=len(messages))
        assert result is None

    def test_boundary_excludes_later_items(self):
        adapter = _make_adapter()
        messages = _make_messages(n_turns=3)
        result = adapter.find_last_tool_call(messages, "search", boundary=2)
        assert result is None

    def test_returns_last_match(self):
        adapter = _make_adapter()
        messages = _make_messages(n_turns=3)
        result = adapter.find_last_tool_call(messages, "search", boundary=len(messages))
        last_idx = None
        for i, m in enumerate(messages):
            if m.get("role") == "assistant" and adapter._tool_call_start in m.get("content", ""):
                try:
                    call_json = m["content"].split(adapter._tool_call_start)[1].split(adapter._tool_call_end)[0]
                    parsed = json5.loads(call_json.strip())
                    if parsed.get("name") == "search":
                        last_idx = i
                except Exception:
                    pass
        assert result == last_idx

    def test_invalid_json_skipped(self):
        adapter = _make_adapter()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": TC_START + "invalid json" + TC_END},
            {"role": "assistant", "content": TC_START + '{"name": "search", "arguments": {}}' + TC_END},
        ]
        result = adapter.find_last_tool_call(messages, "search", boundary=len(messages))
        assert result == 3


class TestIsToolCall:
    def test_assistant_with_tool_call(self):
        adapter = _make_adapter()
        item = {"role": "assistant", "content": "thinking" + TC_START + '{"name": "search"}' + TC_END}
        assert adapter.is_tool_call(item) is True

    def test_assistant_without_tool_call(self):
        adapter = _make_adapter()
        item = {"role": "assistant", "content": "just thinking"}
        assert adapter.is_tool_call(item) is False

    def test_user_message(self):
        adapter = _make_adapter()
        item = {"role": "user", "content": "observation result"}
        assert adapter.is_tool_call(item) is False

    def test_non_dict(self):
        adapter = _make_adapter()
        assert adapter.is_tool_call("string") is False
        assert adapter.is_tool_call(42) is False
        assert adapter.is_tool_call(None) is False


class TestSetToolCallContent:
    def test_replace_think_content(self):
        adapter = _make_adapter()
        NL = chr(10)
        item = {"role": "assistant", "content": TK_START + "旧思考内容" + TK_END + NL + "rest"}
        adapter.set_tool_call_content(item, "新的摘要")
        assert "新的摘要" in item["content"]
        assert "旧思考内容" not in item["content"]
        assert TK_START in item["content"]
        assert TK_END in item["content"]
        assert "rest" in item["content"]

    def test_multiline_think_content(self):
        adapter = _make_adapter()
        NL = chr(10)
        item = {"role": "assistant", "content": TK_START + "line1" + NL + "line2" + NL + "line3" + TK_END}
        adapter.set_tool_call_content(item, "summary")
        assert "summary" in item["content"]
        assert "line1" not in item["content"]

    def test_non_assistant_ignored(self):
        adapter = _make_adapter()
        item = {"role": "user", "content": "some content"}
        adapter.set_tool_call_content(item, "summary")
        assert item["content"] == "some content"

    def test_non_dict_ignored(self):
        adapter = _make_adapter()
        adapter.set_tool_call_content("not a dict", "summary")

    def test_chinese_content(self):
        adapter = _make_adapter()
        item = {"role": "assistant", "content": TK_START + "旧内容" + TK_END}
        adapter.set_tool_call_content(item, "中文摘要内容")
        assert "中文摘要内容" in item["content"]


class TestReconstructHistory:
    def test_basic_reconstruct(self):
        adapter = _make_adapter()
        messages = _make_messages(n_turns=3)
        boundary = len(messages) - 2
        result = adapter.reconstruct_history(messages, boundary=boundary, keep_header_count=2)
        assert result[:2] == messages[:2]
        assert result[-2:] == messages[-2:]

    def test_header_only(self):
        adapter = _make_adapter()
        messages = _make_messages(n_turns=3)
        result = adapter.reconstruct_history(messages, boundary=2, keep_header_count=2)
        assert result[:2] == messages[:2]
        assert result[2:] == messages[2:]

    def test_tool_calls_skeleton(self):
        adapter = _make_adapter()
        messages = _make_messages(n_turns=3)
        boundary = len(messages) - 2
        result = adapter.reconstruct_history(messages, boundary=boundary, keep_header_count=2)
        tool_calls_before = [
            m for i, m in enumerate(messages[:boundary])
            if adapter.is_tool_call(m)
        ]
        for tc in tool_calls_before[1:]:
            assert tc in result
        if tool_calls_before:
            skeleton = result[2:len(result) - 2]
            assert tool_calls_before[0] not in skeleton

