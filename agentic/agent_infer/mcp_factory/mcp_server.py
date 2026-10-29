#!/usr/bin/env python3
"""
Standard MCP Server for Multi Agent System Tools
Implements Model Context Protocol using official Python SDK with proper decorators
Includes ALL func_tools from the original mcp_factory.py implementation
"""

import asyncio
import logging

from dotenv import load_dotenv, find_dotenv
from mcp.server import FastMCP
from tools.tool_register import ToolScanner
import uvicorn
import swanlab
from pathlib import Path
import sys
import config.config as config

from agent_infer.mcp_factory.tool_cache.redis_manager import RedisManager # Redis
RedisManager().start()
import agent_infer.mcp_factory.tool_cache.tool_cache_lib # tool_cache

swanlab.init(
    logdir=config.SWANLAB_LOG_PATH,
    mode="local",
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastMCP("DeepDiver-MultiAgent-Tools")
# 示例：扫描项目中的tools包
scanner = ToolScanner("tools")
scanner.scan_package()

logger.info("Discovered tools:")
tools_list = []
for name, func in scanner.get_tools().items():
    logger.info(f"- {name}")
    tools_list.append(func)

# 注册所有tool
for tool in tools_list:
    app.add_tool(tool)


def main():
    """Main function to run the server"""
    # Initialize FastMCP server

    # Run the FastMCP server
    host = config.MCP_HOST
    port = config.MCP_PORT
    try:

        # Use streamable HTTP transport for proper MCP over HTTP
        logger.info("🚀 Starting DeepDiver MCP Server...")
        # logger.info("📡 Available func_tools: 16 comprehensive MCP func_tools")
        logger.info("🌐 Server will start on configured host and port")
        logger.info("=" * 50)

        # Create the ASGI app for streamable HTTP
        app_instance = app.streamable_http_app()

        # Default configuration

        # Get The name and quantity of MCP tools

        async def get_tools():
            tools = await app.list_tools()
            return tools

        tools = asyncio.run(get_tools())
        tool_names = [tool.name for tool in tools]
        tool_nums = len(tools)
        logger.info(f"Available tools: {tool_names}")
        logger.info(f"Number of tools: {tool_nums}")

        logger.info(f"🌐 Starting server at http://{host}:{port}/mcp")
        logger.info(f"🚀 正在启动 DeepDiver MCP 服务器...")
        logger.info(f"📡 可用工具：{tool_nums} 个综合 MCP 工具")
        logger.info(f"🌐 服务器将在 {host}:{port}/mcp 上运行")
        logger.info("=" * 50)

        uvicorn.run(
            app_instance,
            host=host,
            port=port,
            log_level="info",
            timeout_keep_alive=3600,
        )

    except ImportError:
        logger.info(
            "❌ uvicorn not found. Please install with: pip install uvicorn")
        logger.info("🔄 Trying fallback method...")

        # Fallback: Use FastMCP's built-in server

        # Default configuration
        transport = "streamable-http"
        app.run(transport=transport)
        logger.info(f"✅ MCP 服务端已启动 ({transport}) 端口:{port}")


if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)
    main()
