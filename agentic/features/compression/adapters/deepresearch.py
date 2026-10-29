from __future__ import annotations

import json5
import re
from typing import Any, Dict, List, Optional

from features.compression.conversation_adapter import ConversationAdapter


class DeepResearchConversationAdapter:
    """Deep Research 的对话适配器实现

    承担的编排依赖（构造时注入）：
    - system_prompt: 压缩 LLM 需要的 system 指令
    - think_tag: think 标签名，用于构建 think_start/think_end 标记
    - tool_call_start: 工具调用开始标记
    - tool_call_end: 工具调用结束标记
    """

    def __init__(
        self,
        system_prompt: str,
        think_tag: str = "think",
        tool_call_start: str = '<tool_call>',
        tool_call_end: str = '</tool_call>',
    ):
        self._system_prompt = system_prompt
        self._think_tag = think_tag
        self._think_start = f"<{think_tag}>"
        self._think_end = f"</{think_tag}>"
        self._tool_call_start = tool_call_start
        self._tool_call_end = tool_call_end

    async def to_messages(
        self, items: List[Any], system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """将 Deep Research 的 messages 列表转换为压缩 LLM 可用的标准消息格式

        Deep Research 的 messages 本身就是标准 dict 格式，
        只需替换 system_prompt 即可直接传给压缩 LLM。

        Args:
            items: 标准 messages 列表（List[Dict["role", "content"]]）
            system_prompt: 若提供，替换原有的 system 消息

        Returns:
            标准 messages 列表，可直接传给压缩 LLM
        """
        messages = list(items)
        sp = system_prompt or self._system_prompt
        if sp:
            messages = [m for m in messages if m.get("role") != "system"]
            messages.insert(0, {"content": sp, "role": "system"})
        return messages

    def find_last_tool_call(
        self, items: List[Any], tool_name: str, boundary: int
    ) -> Optional[int]:
        """在 boundary 之前查找最后一个指定工具调用的索引

        遍历 messages[:boundary]，在 assistant 消息中解析
        tool_call_start...tool_call_end 标记内的 JSON，匹配工具名。
        """
        if tool_name == self._think_tag:
            for index in range(min(boundary, len(items)) - 1, -1, -1):
                item = items[index]
                if not isinstance(item, dict) or item.get("role") != "assistant":
                    continue
                content = item.get("content", "")
                if self._think_start in content and self._think_end in content:
                    return index
            return None

        for i, item in enumerate(items[:boundary][::-1]):
            if item.get("role") != "assistant":
                continue
            content = item.get("content", "")
            if self._tool_call_start not in content:
                continue
            try:
                call_json = content.split(self._tool_call_start)[1].split(self._tool_call_end)[0]
                parsed = json5.loads(call_json.strip())
                if parsed.get("name") == tool_name:
                    return len(items[:boundary]) - 1 - i
            except (json5.JSONDecodeError, KeyError, IndexError):
                continue
        return None

    def is_tool_call(self, item: Any) -> bool:
        """判断消息是否为工具调用（assistant 消息且包含工具调用标记）"""
        if not isinstance(item, dict):
            return False
        return (
            item.get("role") == "assistant"
            and self._tool_call_start in item.get("content", "")
        )

    def set_tool_call_content(self, item: Any, content: str) -> None:
        """替换 assistant 消息中 think_start...think_end 标签内的内容为摘要"""
        if not isinstance(item, dict) or item.get("role") != "assistant":
            return
        old_content = item.get("content", "")
        pattern = re.escape(self._think_start) + r"(.*?)" + re.escape(self._think_end)
        new_content = re.sub(
            pattern,
            self._think_start + content + self._think_end,
            old_content,
            flags=re.DOTALL,
        )
        item["content"] = new_content

    def reconstruct_history(
        self, items: List[Any], boundary: int, keep_header_count: int
    ) -> List[Any]:
        """重构对话历史：保留头部 + 工具调用骨架 + boundary 之后的项"""
        tool_calls = [
            item
            for i, item in enumerate(items)
            if self.is_tool_call(item) and i < boundary
        ]
        return items[:keep_header_count] + tool_calls[1:] + items[boundary:]
