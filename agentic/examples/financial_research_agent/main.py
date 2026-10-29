import logging
from agent_infer.agent_specs.deepdiver_v2.planner import planner_agent
from pathlib import Path
import sys
import asyncio
import os
import httpx
from agents import Agent, Runner, gen_trace_id, trace, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from agents import set_default_openai_client
from dotenv import load_dotenv, find_dotenv
import config.config as config
from agents.mcp import MCPServer, MCPServerStreamableHttp
load_dotenv(find_dotenv(), override=True)
logger = logging.getLogger(__name__)
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))


async def run(mcp_server: MCPServer):
    async with httpx.AsyncClient(verify=False) as custom_httpx_client:
        agent = planner_agent.clone(
            name='planner_agent',
            mcp_servers=[mcp_server]
        )
        while True:
            query = input("Enter a financial research query: ")
            logger.info(f"\n\nRunning code:\n{'-' * 50}")
            logger.info(query)
            logger.info('-' * 50)
            result = await Runner.run(starting_agent=agent, input=query, max_turns=50)
            logger.info('\nExecution Result:', result.final_output)


async def main():
    host = config.MCP_HOST
    port = config.MCP_PORT
    async with MCPServerStreamableHttp(
            name="StreamableHttp Python Server",
            params={"url": f"http://{host}:{port}/mcp"},
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
