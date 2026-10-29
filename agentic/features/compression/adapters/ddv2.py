from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from features.compression.conversation_adapter import ConversationAdapter


class Ddv2ConversationAdapter:
    """DeepDiver V2 的对话适配器实现

    承担的编排依赖（构造时注入）：
    - agent: 用于 _maybe_filter_model_input 和 model_settings 解析
    - run_config: 用于 model_settings.resolve()
    - context_wrapper: 用于 _maybe_filter_model_input
    - system_prompt: 压缩 LLM 需要的 system 指令
    - chat_converter: Ddv2ChatConverter，处理 [unused11]...[unused12] 格式转换
    """

    def __init__(
        self,
        chat_converter,
        agent,
        run_config,
        context_wrapper,
        system_prompt: str,
        think_tool_name: str = "think",
    ):
        self._converter = chat_converter
        self._agent = agent
        self._run_config = run_config
        self._context_wrapper = context_wrapper
        self._system_prompt = system_prompt
        self._think_tool_name = think_tool_name

    async def to_messages(
        self, items: List[Any], system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """将 RunItem 对象列表转换为标准 messages，并插入 system_prompt

        Args:
            items: RunItem 对象列表（generated_items）
            system_prompt: 可选的 system 指令

        Returns:
            标准 messages 列表，可直接传给压缩 LLM
        """
        from agents.run import AgentRunner

        # 1. 将 RunItem 转换为 TResponseInputItem（dict 格式）
        # 这样 _maybe_filter_model_input 才能正确处理
        input_items = [item.to_input_item() for item in items]

        # 2. 通过 _maybe_filter_model_input 过滤
        filtered = await AgentRunner._maybe_filter_model_input(
            agent=self._agent,
            run_config=self._run_config,
            context_wrapper=self._context_wrapper,
            input_items=input_items,
            system_instructions=self._system_prompt,
        )
        # 3. 逐项转换为 dict messages
        messages = []
        for item in filtered.input:
            msgs = self._converter.items_to_messages([item])
            messages.extend(msgs)
        # 4. 插入 system_prompt
        sp = system_prompt or self._system_prompt
        if sp:
            messages.insert(0, {"content": sp, "role": "system"})
        return messages

    def find_last_tool_call(
        self, items: List[Any], tool_name: str, boundary: int
    ) -> Optional[int]:
        """在 boundary 之前查找最后一个指定工具调用的索引"""
        from agents.items import ToolCallItem

        for i, item in enumerate(items[:boundary][::-1]):
            if isinstance(item, ToolCallItem) and item.raw_item.name == tool_name:
                return len(items[:boundary]) - 1 - i
        return None

    def is_tool_call(self, item: Any) -> bool:
        from agents.items import ToolCallItem

        return isinstance(item, ToolCallItem)

    def set_tool_call_content(self, item: Any, content: str) -> None:
        """设置 think 工具的 arguments 为摘要内容"""
        item.raw_item.arguments = json.dumps(
            {"thought": content}, ensure_ascii=False
        )

    def reconstruct_history(
        self, items: List[Any], boundary: int, keep_header_count: int
    ) -> List[Any]:
        """重构对话历史：保留头部 + 工具调用骨架 + 边界之后的项"""
        tool_calls = [
            item
            for i, item in enumerate(items)
            if self.is_tool_call(item) and i < boundary
        ]
        return items[:keep_header_count] + tool_calls[1:] + items[boundary:]