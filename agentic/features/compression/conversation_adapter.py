from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class ConversationAdapter(Protocol):
    """对话历史适配器 — 屏蔽不同 Agent 的对话格式差异

    职责：
    1. 对话格式转换（Item 对象 / Dict 消息 → 标准 messages）
    2. 对话结构遍历与重构（查找工具调用、注入摘要、重建历史）
    3. 承担 Agent 编排依赖（agent, run_config, context_wrapper, system_prompt 等
       在适配器构造时注入，不暴露给 CompressionManager 的调用签名）
    """

    async def to_messages(
        self, items: List[Any], system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """将 Agent 的对话项列表转换为压缩 LLM 可用的标准消息格式

        Args:
            items: Agent 的对话项列表（如 List[ToolCallItem/MessageOutputItem]）
            system_prompt: 若提供，插入为第一条 system 消息

        Returns:
            标准 messages 列表，可直接传给压缩 LLM
        """
        ...

    def find_last_tool_call(
        self, items: List[Any], tool_name: str, boundary: int
    ) -> Optional[int]:
        """在 boundary 之前查找最后一个指定工具调用的索引"""
        ...

    def reconstruct_history(
        self,
        items: List[Any],
        boundary: int,
        keep_header_count: int,
    ) -> List[Any]:
        """重构对话历史：保留头部 + 工具调用骨架 + 边界之后的项"""
        ...

    def is_tool_call(self, item: Any) -> bool:
        """判断项是否为工具调用"""
        ...

    def set_tool_call_content(self, item: Any, content: str) -> None:
        """设置工具调用项的内容（用于注入摘要到 think 工具的 arguments）"""
        ...