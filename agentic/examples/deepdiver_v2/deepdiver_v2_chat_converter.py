from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Literal, cast
from agents.models.chatcmpl_converter import Converter
from agents.models.fake_id import FAKE_RESPONSES_ID
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam, ChatCompletionMessage,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from agents.exceptions import AgentsException, UserError
from agents.handoffs import Handoff
from agents.items import TResponseInputItem, TResponseOutputItem
from agents.tool import FunctionTool, Tool
from openai.types.responses import ResponseReasoningItem, ResponseOutputMessage, ResponseOutputText, \
    ResponseOutputRefusal, ResponseFunctionToolCall
from openai.types.responses.response_reasoning_item import Summary


class Ddv2ChatConverter(Converter):
    @classmethod
    def items_to_messages(
        cls,
        items: str | Iterable[TResponseInputItem],
        preserve_thinking_blocks: bool = False,
    ) -> list[ChatCompletionMessageParam]:
        """
        Convert a sequence of 'Item' objects into a list of ChatCompletionMessageParam.

        Args:
            items: A string or iterable of response input items to convert
            preserve_thinking_blocks: Whether to preserve thinking blocks in tool calls
                for reasoning models like Claude 4 Sonnet/Opus which support interleaved
                thinking. When True, thinking blocks are reconstructed and included in
                assistant messages with tool calls.

        Rules:
        - EasyInputMessage or InputMessage (role=user) => ChatCompletionUserMessageParam
        - EasyInputMessage or InputMessage (role=system) => ChatCompletionSystemMessageParam
        - EasyInputMessage or InputMessage (role=developer) => ChatCompletionDeveloperMessageParam
        - InputMessage (role=assistant) => Start or flush a ChatCompletionAssistantMessageParam
        - response_output_message => Also produces/flushes a ChatCompletionAssistantMessageParam
        - tool calls get attached to the *current* assistant message, or create one if none.
        - tool outputs => ChatCompletionToolMessageParam
        """

        if isinstance(items, str):
            return [
                ChatCompletionUserMessageParam(
                    role="user",
                    content=items,
                )
            ]

        result: list[ChatCompletionMessageParam] = []
        current_assistant_msg: ChatCompletionAssistantMessageParam | None = None
        pending_thinking_blocks: list[dict[str, str]] | None = None

        def flush_assistant_message() -> None:
            nonlocal current_assistant_msg
            if current_assistant_msg is not None:
                # The API doesn't support empty arrays for tool_calls
                if not current_assistant_msg.get("content"):
                    del current_assistant_msg["content"]
                result.append(current_assistant_msg)
                current_assistant_msg = None

        def ensure_assistant_message() -> ChatCompletionAssistantMessageParam:
            nonlocal current_assistant_msg, pending_thinking_blocks
            if current_assistant_msg is None:
                current_assistant_msg = ChatCompletionAssistantMessageParam(role="assistant")
                current_assistant_msg["content"] = []

            return current_assistant_msg

        for item in items:
            # 1) Check easy input message
            if easy_msg := cls.maybe_easy_input_message(item):
                role = easy_msg["role"]
                content = easy_msg["content"]

                if role == "user":
                    flush_assistant_message()
                    msg_user: ChatCompletionUserMessageParam = {
                        "role": "user",
                        "content": cls.extract_all_content(content),
                    }
                    result.append(msg_user)
                elif role == "system":
                    flush_assistant_message()
                    msg_system: ChatCompletionSystemMessageParam = {
                        "role": "system",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_system)
                elif role == "developer":
                    flush_assistant_message()
                    msg_developer: ChatCompletionDeveloperMessageParam = {
                        "role": "developer",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_developer)
                elif role == "assistant":
                    flush_assistant_message()
                    msg_assistant: ChatCompletionAssistantMessageParam = {
                        "role": "assistant",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_assistant)
                else:
                    raise UserError(f"Unexpected role in easy_input_message: {role}")

            # 2) Check input message
            elif in_msg := cls.maybe_input_message(item):
                role = in_msg["role"]
                content = in_msg["content"]
                flush_assistant_message()

                if role == "user":
                    msg_user = {
                        "role": "user",
                        "content": cls.extract_all_content(content),
                    }
                    result.append(msg_user)
                elif role == "system":
                    msg_system = {
                        "role": "system",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_system)
                elif role == "developer":
                    msg_developer = {
                        "role": "developer",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_developer)
                else:
                    raise UserError(f"Unexpected role in input_message: {role}")

            # 3) response output message => assistant
            elif resp_msg := cls.maybe_response_output_message(item):
                flush_assistant_message()
                new_asst = ChatCompletionAssistantMessageParam(role="assistant")
                contents = resp_msg["content"]

                text_segments = []
                for c in contents:
                    if c["type"] == "output_text":
                        text_segments.append(c["text"])
                    elif c["type"] == "refusal":
                        new_asst["refusal"] = c["refusal"]
                    elif c["type"] == "output_audio":
                        # Can't handle this, b/c chat completions expects an ID which we dont have
                        raise UserError(
                            f"Only audio IDs are supported for chat completions, but got: {c}"
                        )
                    else:
                        raise UserError(f"Unknown content type in ResponseOutputMessage: {c}")

                if text_segments:
                    combined = "\n".join(text_segments)
                    new_asst["content"] = combined

                new_asst["content"] = []
                current_assistant_msg = new_asst

            # 4) function/file-search calls => attach to assistant
            elif file_search := cls.maybe_file_search_call(item):
                asst = ensure_assistant_message()
                tool_calls = list(asst.get("tool_calls", []))
                new_tool_call = ChatCompletionMessageFunctionToolCallParam(
                    id=file_search["id"],
                    type="function",
                    function={
                        "name": "file_search_call",
                        "arguments": json.dumps(
                            {
                                "queries": file_search.get("queries", []),
                                "status": file_search.get("status"),
                            }
                        ),
                    },
                )
                tool_calls.append(new_tool_call)
                asst["content"] =json.dumps(tool_calls, ensure_ascii=False)

            elif func_call := cls.maybe_function_tool_call(item):
                asst = ensure_assistant_message()
                # If we have pending thinking blocks, use them as the content
                # This is required for Anthropic API tool calls with interleaved thinking
                if pending_thinking_blocks:
                    # If there is a text content, save it to append after thinking blocks
                    # content type is Union[str, Iterable[ContentArrayOfContentPart], None]
                    if "content" in asst and isinstance(asst["content"], str):
                        text_content = ChatCompletionContentPartTextParam(
                            text=asst["content"], type="text"
                        )
                        asst["content"] = [text_content]

                    if "content" not in asst or asst["content"] is None:
                        asst["content"] = []

                    # Thinking blocks MUST come before any other content
                    # We ignore type errors because pending_thinking_blocks is not openai standard
                    asst["content"] = pending_thinking_blocks + asst["content"]  # type: ignore
                    pending_thinking_blocks = None  # Clear after using

                tool_calls = list(asst.get("tool_calls", []))
                arguments = func_call["arguments"] if func_call["arguments"] else "{}"
                new_tool_call = {
                         "name": func_call["name"],
                         "arguments": arguments
                }
                tool_calls.append(new_tool_call)
                def tool_calls_to_string(tool_calls) -> str:
                    import json

                    """
                    将 tool_calls 列表转换为指定格式的字符串：
                    [unused11][{"name": "...", "arguments": {...}}][unused12]
                    Args:
                        tool_calls (list): List of dict with 'name' and 'arguments'
                    Returns:
                        str: 格式化后的字符串
                    """
                    if not tool_calls:
                        return ""
                    processed_calls = []
                    for call in tool_calls:
                        name = call.get("name", "")
                        arguments = call.get("arguments")
                        if not arguments:
                            arguments = "{}"
                        elif isinstance(arguments, (dict, list)):
                            arguments = json.dumps(arguments, ensure_ascii=False)
                        else:
                            arguments = str(arguments)
                        try:
                            json.loads(arguments)
                        except json.JSONDecodeError:
                            return f"工具返回格式不正确，arguments参数并不是一个合法json object：arguments：{arguments}"

                        processed_call = {
                            "name": name,
                            "arguments": json.loads(arguments)
                        }
                        processed_calls.append(processed_call)
                    json_str = json.dumps(processed_calls, ensure_ascii=False)
                    result = f"[unused11]{json_str}[unused12]"
                    return result
                asst["content"] = tool_calls_to_string(tool_calls)
            # 5) function call output => tool message
            elif func_output := cls.maybe_function_tool_call_output(item):
                flush_assistant_message()
                msg: ChatCompletionToolMessageParam = {
                    "role": "tool",
                   # "tool_call_id": func_output["call_id"],
                    "content": func_output["output"],
                }
                result.append(msg)

            # 6) item reference => handle or raise
            elif item_ref := cls.maybe_item_reference(item):
                raise UserError(
                    f"Encountered an item_reference, which is not supported: {item_ref}"
                )

            # 7) reasoning message => extract thinking blocks if present
            elif reasoning_item := cls.maybe_reasoning_message(item):
                # Reconstruct thinking blocks from content (text) and encrypted_content (signature)
                content_items = reasoning_item.get("content", [])
                encrypted_content = reasoning_item.get("encrypted_content")
                signatures = encrypted_content.split("\n") if encrypted_content else []

                if content_items and preserve_thinking_blocks:
                    # Reconstruct thinking blocks from content and signature
                    pending_thinking_blocks = []
                    for content_item in content_items:
                        if (
                            isinstance(content_item, dict)
                            and content_item.get("type") == "reasoning_text"
                        ):
                            thinking_block = {
                                "type": "thinking",
                                "thinking": content_item.get("text", ""),
                            }
                            # Add signatures if available
                            if signatures:
                                thinking_block["signature"] = signatures.pop(0)
                            pending_thinking_blocks.append(thinking_block)

            # 8) If we haven't recognized it => fail or ignore
            else:
                raise UserError(f"Unhandled item type or structure: {item}")

        flush_assistant_message()
        return result

    @classmethod
    def function_tool_to_openai_tool(cls, func_tool) -> dict:
        """
        将自定义 FunctionTool 对象转换为 OpenAI Tool Schema
        """
        import re
        def extract_descriptions(field_names: str, arg_lines: list) -> str | None:
            result = {}
            for line in arg_lines:
                for field in field_names:
                    if field not in result:
                        pattern = rf'^{re.escape(field)}\s*(?:\([^)]*\))?\s*:\s*(.+)$'
                        match = re.search(pattern, line.strip(), re.IGNORECASE)
                        desc = match.group(1).strip() if match else None
                        if desc:
                            result[field] = desc
            return result

        lines = [line.strip() for line in func_tool.description.strip().split('\n') if line.strip()]
        arg_lines = []
        start_idx = None
        for i, line in enumerate(lines):
            if line.startswith('Args:') or 'Args:' in line:
                start_idx = i
                break
        if start_idx is not None:
            for i in range(start_idx + 1, len(lines)):
                line = lines[i]
                if not line or re.match(r'^(Returns?|Note|Example|Raises?):', line, re.IGNORECASE):
                    break
                arg_lines.append(line)
        else:
            description_lines = lines
        params_schema = func_tool.params_json_schema
        property_descriptions_map = extract_descriptions(list(params_schema["properties"].keys()), arg_lines)

        properties = {}
        required = [r for r in params_schema.get("required", [])]  # if r != "session_id"]
        for key, prop in params_schema["properties"].items():
            clean_prop = {k: v for k, v in prop.items() if k not in ("title",)}
            if "description" not in clean_prop and key in property_descriptions_map:
                clean_prop["description"] = property_descriptions_map[key]
            properties[key] = clean_prop
        return {
            "type": "function",
            "function": {
                "name": func_tool.name,
                "description": func_tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required if required else None,
                }
            }
        }