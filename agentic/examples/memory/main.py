from agent_infer.agent_specs.base_agent import IDAgent
from agent_infer.agent_specs.agent_hooks import AGENT_HOOKS
import os
from agents import Agent, Runner, gen_trace_id, trace, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from agents import set_default_openai_client
import asyncio
from typing import Any, List, Dict
from agents.run_context import RunContextWrapper
from dotenv import load_dotenv, find_dotenv
from agents import function_tool
from agents import trace
from agents import gen_trace_id
import sys
from pathlib import Path
import config.config as config
import logging
import uuid
import httpx
from examples.deepdiver_v2.deepdiver_v2_agent_runner import Ddv2AgentRunner
from agents.mcp import MCPServer, MCPServerStreamableHttp,create_static_tool_filter

load_dotenv(find_dotenv(), override=True)
logger = logging.getLogger(__name__)
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

prompt = """
# Role
You are a **Senior Research Analyst and Report Writing Expert**. You possess a persistent memory system that allows you to span long conversation cycles, assisting users in collecting information, organizing thoughts, and ultimately producing high-quality industry research reports or summary documents.

# Tools
You have access to the following core memory tools. You must actively call these tools based on the conversation context:

1.  `add_memories(memory_contents: List[str], session_id: str)`
    *   **Purpose**: To store key facts, data, insights, reference links, or report formatting requirements provided by the user.
    *   **Trigger**: When the user inputs new research materials, defines a report outline, or corrects specific viewpoints.
    *   **Best Practice**: **Do not** store raw, long-form text directly. You must **clean and summarize** the content first, splitting it into independent, semantically complete strings (List[str]).
        *   *Bad*: Storing a whole paragraph about AI funding.
        *   *Good*: `["Q4 2023 AI industry total funding was $5 billion", "User emphasized focusing on Transformer architecture in Chapter 3"]`.

2.  `search_shared_memories(query: str, session_id: str)`
    *   **Purpose**: To retrieve relevant information from the memory bank when answering questions or writing specific report sections.
    *   **Trigger**: When the user asks, "What was the data I mentioned earlier?" or requests, "Write a summary based on the materials collected so far."
    *   **Best Practice**: Before starting to write, construct a specific keyword-based `query` relevant to the topic to ensure the output is grounded in facts.

# Workflow

**Phase 1: Collection & Organization**
*   When the user provides URLs, article snippets, or raw data, you **MUST** extract the core elements (e.g., definitions, specific statistics, timelines, pros/cons).
*   **Action**: Call `add_memories` to store these specific elements.
*   **Response**: Briefly confirm the key points recorded and ask if there is more material to add.

**Phase 2: Planning**
*   When the user defines the report requirements, record the topic, target audience, and structural outline.
*   **Action**: Call `add_memories` to store the outline and requirements.

**Phase 3: Drafting & Output**
*   When the user asks to generate a full report or a specific chapter, **do not** rely solely on the immediate context or hallucinate details.
*   **Action**: Call `search_shared_memories` using queries specific to that chapter's topic to retrieve stored materials.
*   **Response**: Synthesize the retrieved memories into a coherent text. Cite sources if the source information was stored in memory.

# Constraints & Guidelines
1.  **Grounding**: Your responses should prioritize content retrieved via `search_shared_memories`, followed by your general knowledge. Strictly adhere to private data provided by the user.
2.  **Structured Storage**: When storing memories, it is recommended to add simple tags for better retrieval, e.g., "[Data] XXX..." or "[Insight] XXX...".
3.  **Proactivity**: If the user asks you to write a report section but you find the memory bank lacks information on that specific topic (empty search results), actively inform the user that material is missing rather than fabricating it.
4.  **Session ID**: Always use the current conversation's `session_id` for all operations.

# Example Interaction

**User**: "We need to research DeepSeek's market performance. Here is a news snippet: DeepSeek App topped the US iOS download charts, surpassing ChatGPT..."
**Agent (Thought)**: The user provided key research material. I need to extract the insights: 1. DeepSeek App topped US iOS charts. 2. It surpassed ChatGPT.
**Agent (Tool Call)**: `add_memories(memory_contents=["[Market Performance] DeepSeek App topped US iOS download charts", "[Competition] DeepSeek downloads surpassed ChatGPT"], session_id="current_session")`
**Agent (Response)**: "I have recorded the key information regarding DeepSeek topping the US iOS charts and surpassing ChatGPT. Do you have more materials regarding their technical specs or stock performance?"

**User**: "Based on what we discussed, write a paragraph analyzing DeepSeek's market impact."
**Agent (Thought)**: I need to write content. First, I must retrieve the relevant facts from memory.
**Agent (Tool Call)**: `search_shared_memories(query="DeepSeek market performance downloads", session_id="current_session")`
**Agent (Response)**: (Synthesizes the analysis based on the tool output...)
"""
MODEL_NAME = os.environ.get('EXAMPLE_MODEL_NAME')
BASE_URL = os.environ.get('OPENAI_BASE_URL')
API_KEY = os.environ.get('OPENAI_API_KEY')
base_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
base_agent = IDAgent(
    name='Base Agent',
    instructions=prompt,
    model=OpenAIChatCompletionsModel(
        model=MODEL_NAME, openai_client=base_client),
    hooks=AGENT_HOOKS
)

MCP_SERVER = None


def create_child_agent(agent, mcp_server):
    return agent.clone(
        mcp_servers=[mcp_server]  # 复用父agent的MCP服务，而非重新创建
    )


@function_tool(strict_mode=False)
async def assign_search_memory_tasks_to_memory_agent(
        ctx: RunContextWrapper[Any],
        query: str,
) -> Any:
    if isinstance(ctx.context, tuple):
        session_id = ctx.context[0].get("session_id")
    elif isinstance(ctx.context, dict):
        # {"session_id": session_id},  # 注入会话ID
        session_id = ctx.context.get("session_id")
    context = {"session_id": session_id}

    adder = create_child_agent(base_agent, MCP_SERVER)
    result = await Ddv2AgentRunner.run(starting_agent=adder, input=query, max_turns=config.WRITER_MAX_TURNS,
                                       context=context, conversation_id=session_id)
    return result.final_output


@function_tool(strict_mode=False)
async def assign_add_memory_tasks_to_memory_agent(
        ctx: RunContextWrapper[Any],
        memory: List[str],
) -> Any:
    if isinstance(ctx.context, tuple):
        session_id = ctx.context[0].get("session_id")
    elif isinstance(ctx.context, dict):
        # {"session_id": session_id},  # 注入会话ID
        session_id = ctx.context.get("session_id")
    context = {"session_id": session_id}

    searcher = create_child_agent(base_agent, MCP_SERVER)
    result = await Ddv2AgentRunner.run(starting_agent=searcher, input=memory, max_turns=config.WRITER_MAX_TURNS,
                                       context=context, conversation_id=session_id)
    return result.final_output


async def run(mcp_server: MCPServer):
    async with httpx.AsyncClient(verify=False) as custom_httpx_client:
        set_default_openai_client(base_client)
        global MCP_SERVER
        MCP_SERVER = mcp_server
        memory_agent = base_agent.clone(mcp_servers=[mcp_server],
                                        # tools=[assign_add_memory_tasks_to_memory_agent,
                                        #        assign_search_memory_tasks_to_memory_agent],
                                        )

        # 示例："a1b2c3d4-5678-90ef-ghij-klmnopqrstuv"
        session_id = str(uuid.uuid4())
        context = {"session_id": session_id},  # 注入会话ID

        while True:
            query = input("Enter a query: ")
            logger.info(f"\n\nRunning code:\n{'-' * 50}")
            logger.info(query)
            logger.info('-' * 50)
            result = await Ddv2AgentRunner.run(starting_agent=memory_agent, input=query, max_turns=50, context=context,
                                      conversation_id=session_id)
            logger.info(f"\n\nRunning code:\n{'-' * 50}")
            logger.info(f'\nExecution Result:{result.final_output}')


async def main():
    async with MCPServerStreamableHttp(
            name="StreamableHttp Python Server",
            params={
                "url": f"http://{config.MCP_HOST}:{config.MCP_PORT}/mcp",
            },
            tool_filter=create_static_tool_filter(
                allowed_tool_names=["search_shared_memories"],
            ),
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
    ) as server:
        trace_id = gen_trace_id()
        with trace(workflow_name="StreamableHttp Example", trace_id=trace_id, disabled=True):
            await run(server)


import logging

# 设置日志级别为 INFO
logging.basicConfig(level=logging.INFO)
asyncio.run(main())
