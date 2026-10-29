"""
Example demonstrating how to use the reasoning content feature with the Runner API.

This example shows how to extract and use reasoning content from responses when using
the Runner API, which is the most common way users interact with the Agents library.

To run this example, you need to:
1. Set your OPENAI_API_KEY environment variable
2. Use a model that supports reasoning content (e.g., deepseek-reasoner)
"""


import logging
from pathlib import Path
from agent_infer.agent_specs.base_agent import IDAgent
import sys
import asyncio
import os
import httpx
from agents import Agent, Runner, gen_trace_id, trace, OpenAIChatCompletionsModel, ItemHelpers
from openai import AsyncOpenAI
from agents import set_default_openai_client
from dotenv import load_dotenv, find_dotenv
import config.config as config
from agents.mcp import MCPServer, MCPServerStreamableHttp
load_dotenv(find_dotenv(), override=True)
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))
logger = logging.getLogger(__name__)


async def run(mcp_server: MCPServer):
    async with httpx.AsyncClient(verify=False) as custom_httpx_client:
        MODEL_NAME = os.environ.get('EXAMPLE_MODEL_NAME')
        BASE_URL = os.environ.get('OPENAI_BASE_URL')
        API_KEY = os.environ.get('OPENAI_API_KEY')
        base_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
        set_default_openai_client(base_client)
        agent = IDAgent(
            name='Base Agent',
            model=OpenAIChatCompletionsModel(
                model=MODEL_NAME, openai_client=base_client),
            mcp_servers=[mcp_server]
        )
        while True:
            query = input("Enter a financial research query: ")
            logger.info(f"\n\nRunning code:\n{'-' * 50}")
            logger.info(query)
            logger.info('-' * 50)

            # RunResultStreaming is async iterable
            stream = Runner.run_streamed(agent, query)
            logger.info("==============event===========")
            # async for event in stream:  # type: ignore
            async for event in stream.stream_events():  # type: ignore
                if event.type == "agent_updated_stream_event":
                    logger.info(f"Agent已更新: {event.new_agent.name}")
                    continue
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        logger.info(f"-- 工具被调用 {event.item.raw_item.name}")
                    elif event.item.type == "tool_call_output_item":
                        logger.info(f"-- 工具输出: {event.item.output}")
                    elif event.item.type == "message_output_item":
                        finally_result = ItemHelpers.text_message_output(
                            event.item)
            logger.info(f'最终输出: {finally_result}')


async def main():
    async with MCPServerStreamableHttp(
            name="StreamableHttp Python Server",
            params={
                "url": f"http://{config.MCP_HOST}:{config.MCP_PORT}/mcp",
            },
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
    ) as server:
        trace_id = gen_trace_id()
        with trace(workflow_name="StreamableHttp Example", trace_id=trace_id, disabled=True):
            logger.info(
                f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            await run(server)


asyncio.run(main())
