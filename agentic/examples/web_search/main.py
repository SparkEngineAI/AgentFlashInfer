import logging
from agent_infer.agent_specs.common.websearch import websearch_agent
from pathlib import Path
import sys
import asyncio
import os
import httpx
from agents import Agent, Runner, gen_trace_id, trace, OpenAIChatCompletionsModel
from dotenv import load_dotenv, find_dotenv
import config.config as config
from agents.mcp import MCPServer, MCPServerStreamableHttp
load_dotenv(find_dotenv(), override=True)
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))
logger = logging.getLogger(__name__)


async def run(mcp_server: MCPServer):
    async with httpx.AsyncClient(verify=False) as custom_httpx_client:
        agent = websearch_agent.clone(
            name='websearch_agent',
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
