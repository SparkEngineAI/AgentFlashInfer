import copy
import logging

from collections import defaultdict
from typing import Any, Optional

from agents.lifecycle import AgentHooks
from agents.agent import Agent, AgentBase
from agents.items import ModelResponse, TResponseInputItem
from agents.lifecycle import RunHooks
from agents.run_context import RunContextWrapper, TContext
from agents.tool import Tool
from pathlib import Path
import config.config as config
import json


from typing_extensions import TypeVar
TAgent = TypeVar("TAgent", bound=AgentBase, default=AgentBase)
try:
    if config.ENABLE_MEMORY and config.MEMORY_BASE_URL:
        from fusion_mem.client import AsyncMemoryClient
        memory_client = AsyncMemoryClient(config.MEMORY_BASE_URL)
except ImportError:
    config.ENABLE_MEMORY = False


def setup_hooks_logger(log_file_path):
    """为RunHooks配置专属日志器，日志仅写入文件"""
    hooks_logger = logging.getLogger("agents.lifecycle")
    hooks_logger.setLevel(logging.INFO)
    hooks_logger.propagate = False

    hooks_logger.handlers = []

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)

    hooks_logger.addHandler(file_handler)
    return hooks_logger


class RunLogsHooks(RunHooks):
    def __init__(self):
        workspace_path = Path(config.WORKSPACE_PATH)
        workspace_path.mkdir(exist_ok=True)
        self.hooks_logger = setup_hooks_logger(
            f'{config.WORKSPACE_PATH}/RunnerLogsHook.log')
        self.events: dict[str, int] = defaultdict(int)

    def reset(self):
        self.events.clear()

    async def on_agent_start(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext]
    ) -> None:
        self.hooks_logger.info(
            f'agent name: {agent.name}; start context: {context.context}')
        self.events["on_agent_start"] += 1

    async def on_agent_end(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any
    ) -> None:
        self.hooks_logger.info(
            f'agent name: {agent.name}; end output: {output}')
        self.events["on_agent_end"] += 1

    async def on_handoff(
        self,
        context: RunContextWrapper[TContext],
        from_agent: Agent[TContext],
        to_agent: Agent[TContext],
    ) -> None:
        self.events["on_handoff"] += 1

    async def on_tool_start(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool
    ) -> None:
        self.hooks_logger.info(
            f'tool name: {tool.name}; start {context.tool_name}')
        self.events["on_tool_start"] += 1

    async def on_tool_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        tool: Tool,
        result: str,
    ) -> None:
        self.hooks_logger.info(
            f'tool name: {tool.name}; end {context.tool_name}; result: {result}')
        self.events["on_tool_end"] += 1

    async def on_llm_start(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        system_prompt: Optional[str],
        input_items: list[TResponseInputItem],
    ) -> None:
        self.hooks_logger.info(
            f'llm start: {context}; input_items: {input_items}')
        self.events["on_llm_start"] += 1

    async def on_llm_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        response: ModelResponse,
    ) -> None:
        self.hooks_logger.info(f'llm end: {context}; response: {response}')
        self.events["on_llm_end"] += 1


class AgentLogsHooks(AgentHooks):
    """A class that receives callbacks on various lifecycle events for a specific agent. You can
    set this on `agent.hooks` to receive events for that specific agent.

    Subclass and override the methods you need.
    """

    def __init__(self):
        workspace_path = Path(config.WORKSPACE_PATH)
        workspace_path.mkdir(exist_ok=True)
        self.hooks_logger = setup_hooks_logger(
            f'{config.WORKSPACE_PATH}/AgentLogsHooks.log')
        self.events: dict[str, int] = defaultdict(int)

    async def on_start(self, context: RunContextWrapper[TContext], agent: TAgent) -> None:
        """Called before the agent is invoked. Called each time the running agent is changed to this
        agent."""
        self.hooks_logger.info(f'agent name: {agent.name}; start')
        self.events["on_agent_start"] += 1

    async def on_end(
        self,
        context: RunContextWrapper[TContext],
        agent: TAgent,
        output: Any,
    ) -> None:
        """Called when the agent produces a final output."""
        self.hooks_logger.info(
            f'agent name: {agent.name}; end output: {output}')
        self.events["on_agent_end"] += 1

    async def on_handoff(
        self,
        context: RunContextWrapper[TContext],
        agent: TAgent,
        source: TAgent,
    ) -> None:
        """Called when the agent is being handed off to. The `source` is the agent that is handing
        off to this agent."""
        pass

    async def on_tool_start(
        self,
        context: RunContextWrapper[TContext],
        agent: TAgent,
        tool: Tool,
    ) -> None:
        """Called before a tool is invoked."""
        self.hooks_logger.info(
            f'agent name: {agent.name}; tool name: {tool.name}; start tool_call_id: {context.tool_call_id}')
        self.events["on_tool_start"] += 1

    async def on_tool_end(
        self,
        context: RunContextWrapper[TContext],
        agent: TAgent,
        tool: Tool,
        result: str,
    ) -> None:
        """Called after a tool is invoked."""
        self.hooks_logger.info(
            f'agent name: {agent.name}; tool name: {tool.name}; end tool_call_id: {context.tool_call_id}; result: {result}')
        self.events["on_tool_end"] += 1

    async def on_llm_start(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        system_prompt: Optional[str],
        input_items: list[TResponseInputItem],
    ) -> None:
        self.hooks_logger.info(
            f'agent name: {agent.name}; llm start: {context}; input_items: {input_items}')
        self.events["on_llm_start"] += 1

    async def on_llm_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        response: ModelResponse,
    ) -> None:
        self.hooks_logger.info(
            f'agent name: {agent.name}; llm end: {context}; response: {response}')
        self.events["on_llm_end"] += 1


class AgentLogsAndMemoryHooks(AgentHooks):
    """A class that receives callbacks on various lifecycle events for a specific agent. You can
    set this on `agent.hooks` to receive events for that specific agent.

    Subclass and override the methods you need.
    """

    def __init__(self):
        workspace_path = Path(config.WORKSPACE_PATH)
        workspace_path.mkdir(exist_ok=True)
        self.hooks_logger = setup_hooks_logger(
            f'{config.WORKSPACE_PATH}/AgentLogsHooks.log')
        self.events: dict[str, int] = defaultdict(int)

    async def on_start(self, context: RunContextWrapper[TContext], agent: TAgent) -> None:
        """Called before the agent is invoked. Called each time the running agent is changed to this
        agent."""
        self.hooks_logger.info(f'agent name: {agent.name}; start')
        self.events["on_agent_start"] += 1

    async def on_end(
        self,
        context: RunContextWrapper[TContext],
        agent: TAgent,
        output: Any,
    ) -> None:
        """Called when the agent produces a final output."""
        self.hooks_logger.info(
            f'agent name: {agent.name}; end output: {output}')
        self.events["on_agent_end"] += 1

    async def on_handoff(
        self,
        context: RunContextWrapper[TContext],
        agent: TAgent,
        source: TAgent,
    ) -> None:
        """Called when the agent is being handed off to. The `source` is the agent that is handing
        off to this agent."""
        pass

    async def on_tool_start(
        self,
        context: RunContextWrapper[TContext],
        agent: TAgent,
        tool: Tool,
    ) -> None:
        """Called before a tool is invoked."""
        self.hooks_logger.info(
            f'agent name: {agent.name}; tool name: {tool.name}; start tool_call_id: {context.tool_call_id}')
        self.events["on_tool_start"] += 1

    async def on_tool_end(
        self,
        context: RunContextWrapper[TContext],
        agent: TAgent,
        tool: Tool,
        result: str,
    ) -> None:
        """Called after a tool is invoked."""
        self.hooks_logger.info(
            f'agent name: {agent.name}; tool name: {tool.name}; end tool_call_id: {context.tool_call_id}; result: {result}')
        self.events["on_tool_end"] += 1

    async def on_llm_start(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        system_prompt: Optional[str],
        input_items: list[TResponseInputItem],
    ) -> None:
        self.hooks_logger.info(
            f'agent name: {agent.name}; llm start: {context}; input_items: {input_items}')
        self.events["on_llm_start"] += 1
        context.context["input_items"] = input_items
        if len(input_items) == 1:
            context.context["system_prompt"] = system_prompt

    async def on_llm_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        response: ModelResponse,
    ) -> None:
        self.hooks_logger.info(
            f'agent name: {agent.name}; llm end: {context}; response: {response}')
        self.events["on_llm_end"] += 1
        memories = []
        try:
            if "system_prompt" in context.context:
                system_prompt = context.context.pop("system_prompt")
                memories.append({"role":"system","content":system_prompt})
            input_item = context.context["input_items"][-1]
            if "role" in input_item:
                items= copy.deepcopy(input_item)
                content = items["content"]
                if not isinstance(items["content"], str):
                    content = json.dumps(items["content"], ensure_ascii=False)
                memories.append({"content": content, "role":items["role"]})
            else:
                memories.append({"content": json.dumps(input_item,ensure_ascii=False), "role":"assistant"})
            memories.extend([{"content": json.dumps(input_item, ensure_ascii=False), "role":"assistant"} for input_item in response.to_input_items()])
            await memory_client.add(messages=memories, session_id=context.context["session_id"], agent_id=context.context["agent_id"])
        except Exception as e:
            session_id = context.context['session_id']
            agent_id = context.context['agent_id']
            self.hooks_logger.warning(f"add memory failed: {str(e)}, message: {memories}, session_id: {session_id}, agent_id: {agent_id}")


HOOKS = RunLogsHooks()
if config.ENABLE_MEMORY and config.ENABLE_SHARED_MEMORY:
    AGENT_HOOKS = AgentLogsAndMemoryHooks()
else:
    AGENT_HOOKS = AgentLogsHooks()
