import logging
from dotenv import load_dotenv, find_dotenv
from agents.mcp import MCPServerStreamableHttp
from agent_infer.manager.research import ResearchManager
from agents.mcp import ToolFilterContext
import asyncio
import config.config as config
import os
import urllib3
from agents import gen_trace_id, trace
import sys
from pathlib import Path

current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

load_dotenv(find_dotenv(), override=True)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = logging.getLogger(__name__)


def tools_filter(context: ToolFilterContext, tool) -> bool:
    """Example of a tool filter."""
    # Filter logic based on tool name patterns
    if tool.name.startswith('sandbox'):
        return True
    return False


async def main():
    mgr = ResearchManager()
    host = config.MCP_HOST
    port = config.MCP_PORT
    async with MCPServerStreamableHttp(
            name="StreamableHttp Python Server",
            params={
                "url": f"http://{host}:{port}/mcp",
            },
            client_session_timeout_seconds=60,
            tool_filter=tools_filter
    ) as server:
        trace_id = gen_trace_id()
        with trace(workflow_name="StreamableHttp Example", trace_id=trace_id, disabled=True):
            logger.info(
                f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            await mgr.run(server)

if __name__ == '__main__':
    asyncio.run(main())
