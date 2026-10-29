import datetime
import os
from openai import AsyncOpenAI
from agents import set_default_openai_client
from agents import Agent, OpenAIChatCompletionsModel
from dotenv import load_dotenv, find_dotenv
import config.config as config
from .agent_hooks import AGENT_HOOKS
import logging
from agents.model_settings import ModelSettings

from agents.mcp.util import *
from agents import _debug
from agents.run_context import RunContextWrapper, TContext


logger = logging.getLogger(__name__)
logger.info(find_dotenv())
load_dotenv(find_dotenv(), override=True)


class MCPIDUtil(MCPUtil):
    @classmethod
    async def invoke_mcp_tool(
        cls, server: "MCPServer", tool: "MCPTool", context: RunContextWrapper[Any], input_json: str
    ) -> str:
        """Invoke an MCP tool and return the result as a string."""
        try:
            json_data: dict[str, Any] = json.loads(
                input_json) if input_json else {}
            try:
                session_id = 'default_session_id'
                if isinstance(context.context, tuple):
                    session_id = context.context[0].get("session_id")
                elif isinstance(context.context, dict):
                    session_id = context.context.get("session_id")
                if session_id:
                    json_data["session_id"] = session_id  # 注入会话ID到工具参数
            except:
                json_data["session_id"] = 'default_session_id'
                pass
        except Exception as e:
            if _debug.DONT_LOG_TOOL_DATA:
                logger.debug(f"Invalid JSON input for tool {tool.name}")
            else:
                logger.debug(
                    f"Invalid JSON input for tool {tool.name}: {input_json}")
            raise ModelBehaviorError(
                f"Invalid JSON input for tool {tool.name}: {input_json}"
            ) from e

        return await super().invoke_mcp_tool(server, tool, context, json.dumps(json_data))


class IDAgent(Agent):
    async def get_mcp_tools(self, run_context: RunContextWrapper[TContext]) -> list[Tool]:
        """Fetches the available tools from the MCP servers."""
        convert_schemas_to_strict = self.mcp_config.get(
            "convert_schemas_to_strict", False)
        return await MCPIDUtil.get_all_function_tools(
            self.mcp_servers, convert_schemas_to_strict, run_context, self
        )

MODEL_NAME = config.DEFAULT_MODEL_NAME
BASE_URL = config.DEFAULT_MODEL_URL
API_KEY = config.DEFAULT_MODEL_API_KEY
tracing_api_key = API_KEY
# set_tracing_export_api_key(tracing_api_key)
base_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(base_client)
logger.info("Configured base model: model=%s, base_url=%s", MODEL_NAME, BASE_URL)
base_agent = IDAgent(
    name='Base Agent',
    model=OpenAIChatCompletionsModel(
        model=MODEL_NAME, openai_client=base_client),
    hooks=AGENT_HOOKS,
    model_settings=ModelSettings(
        tool_choice="required", temperature=config.model_temperature, max_tokens=config.model_max_tokens)
)


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


cur_date = today_date()
