import logging
from agents import Runner, TResponseInputItem, trace
from agent_infer.agent_specs.base_agent import base_agent
from agent_infer.agent_specs.common.evaluator import eval_agent, EvaluationFeedback
from pathlib import Path
import sys
import asyncio
import os
import httpx
from agents import Agent, Runner, gen_trace_id, trace
from dotenv import load_dotenv, find_dotenv
import config.config as config
from agents.mcp import MCPServer, MCPServerStreamableHttp
load_dotenv(find_dotenv(), override=True)
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))
logger = logging.getLogger(__name__)


async def run(mcp_server: MCPServer):
    agent = base_agent.clone(
        name='planner_agent',
        mcp_servers=[mcp_server]
    )
    eval_agentic = eval_agent.clone(
        name='eval_agent',
        mcp_servers=[mcp_server]
    )
    async with httpx.AsyncClient(verify=False) as custom_httpx_client:
        while True:
            query = input("Enter a financial research query: ")
            logger.info(f"\n\nRunning code:\n{'-' * 50}")
            logger.info(query)
            logger.info('-' * 50)

            input_items: list[TResponseInputItem] = [
                {"content": query, "role": "user"}]

            latest_outline: str | None = None

            while True:
                story_outline_result = await Runner.run(
                    agent,
                    input_items,
                    max_turns=50
                )

                input_items = story_outline_result.to_input_list()
                latest_outline = story_outline_result.final_output
                eval_input = f'user: {query}\nassatent: {latest_outline}\nIs the above Q&A correct? If not, please provide detailed reasons.'
                logger.info("Story outline generated")

                evaluator_result = await Runner.run(eval_agentic, eval_input, max_turns=50)
                result: EvaluationFeedback = evaluator_result.final_output

                logger.info(f"Evaluator score: {result.score}")

                if result.score == "pass":
                    logger.info("Story outline is good enough, exiting.")
                    break

                logger.info(f"Re-running with feedback: {result.feedback}")

                input_items.append(
                    {"content": f"Feedback: {result.feedback}", "role": "user"})
            logger.info(f"Final story outline: {latest_outline}")


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
