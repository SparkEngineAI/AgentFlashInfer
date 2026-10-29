from agent_infer.agent_specs.base_agent import IDAgent
from agent_infer.agent_specs.agent_hooks import AGENT_HOOKS
import uuid
import logging
import config.config as config
from pathlib import Path
import sys
import asyncio
import os
import httpx
from agents import Agent, Runner, gen_trace_id, trace, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from agents import set_default_openai_client
from dotenv import load_dotenv, find_dotenv

from agents.mcp import MCPServer, MCPServerStreamableHttp
load_dotenv(find_dotenv(), override=True)
logger = logging.getLogger(__name__)
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))


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
            mcp_servers=[mcp_server],
            hooks=AGENT_HOOKS
        )
        # 示例："a1b2c3d4-5678-90ef-ghij-klmnopqrstuv"
        session_id = str(uuid.uuid4())
        context = {"session_id": session_id},  # 注入会话ID

        while True:
            query = input("Enter a financial research query: ")
            logger.info(f"\n\nRunning code:\n{'-' * 50}")
            logger.info(query)
            logger.info('-' * 50)
            result = await Runner.run(starting_agent=agent, input=query, max_turns=50, context=context, conversation_id=session_id)
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
