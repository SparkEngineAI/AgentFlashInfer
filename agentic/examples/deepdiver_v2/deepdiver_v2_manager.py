import asyncio
from agents.mcp import create_static_tool_filter, MCPServer
from typing import Any, List, Dict
from agent_infer.agent_specs.deepdiver_v2.writer import writer_agent
from agent_infer.agent_specs.deepdiver_v2.objective_information_seeker import objective_information_seeker_agent
from agent_infer.agent_specs.deepdiver_v2.subjective_information_seeker import subjective_information_seeker_agent
from agent_infer.agent_specs.deepdiver_v2.planner import planner_agent
from agents.run_context import RunContextWrapper
from dotenv import load_dotenv, find_dotenv
import urllib3.exceptions
from agents import function_tool, RunConfig
from agents import trace
from agents.mcp import MCPServerStreamableHttp
from agents import gen_trace_id
from contextlib import AsyncExitStack
import sys
from pathlib import Path
import swanlab
import time
import config.config as config
import logging
import uuid
import httpx

from examples.deepdiver_v2.deepdiver_v2_agent_runner import Ddv2AgentRunner
from examples.deepdiver_v2.extra.server_config import mcp_hub_v2
from examples.deepdiver_v2.extra.mcp_hub import McpHubServer, MCPClient, LoadBalancer

logger = logging.getLogger(__name__)
current_dir = Path(__file__).parent.parent.parent
logger.info(current_dir)
sys.path.append(str(current_dir))


# --- Configuration and Initialization ---
# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)

# Disable SSL certificate warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

swanlab.init(
    logdir=config.SWANLAB_LOG_PATH,
    mode="local",
)


def create_child_agent(agent, mcp_server):
    return agent.clone(
        mcp_servers=[mcp_server]  # 复用父agent的MCP服务，而非重新创建
    )


@function_tool(strict_mode=False)
async def assign_multi_objective_tasks_to_info_seeker(
        ctx: RunContextWrapper[Any],
        tasks: List[Dict[str, str]],
        max_workers: int = 4
) -> Any:
    """
    Creates multiple TaskInput objects and routes them to info_seeker agents for concurrent execution.
    This tool enables the PlannerAgent to assign multiple research tasks through the MCP tool interface.

    Args:
        context: any
        tasks: List of task dictionaries with the following keys:
            - task_content (required): The specific task content
            - task_steps_for_reference: Optional reference steps for execution
            - deliverable_contents: Format of expected deliverable
            - acceptance_checking_criteria: Criteria for task completion and quality
            - workspace_id: Workspace ID for stored files and memory
            - current_task_status: Description of current task status
        max_workers: Maximum concurrent threads (default=4)

    Returns:
        MCPToolResult with execution results for all tasks
    """
    # print(f'ctx: {ctx}')
    semaphore = asyncio.Semaphore(max_workers)
    if isinstance(ctx.context, tuple):
        session_id = ctx.context[0].get("session_id")
    elif isinstance(ctx.context, dict):
        # {"session_id": session_id},  # 注入会话ID
        session_id = ctx.context.get("session_id")
    context = {"session_id": session_id}

    async def process_task_async(task: Dict[str, str], task_id:int) -> Dict[str, Any]:
        information_seeker = create_child_agent(
            objective_information_seeker_agent, mcp_server_objective)
        agent_input = str(task)
        agent_input += "\nPlease analyze this task and start your ReAct process:\n"
        agent_input += "1. Reason about what information you need to gather\n"
        agent_input += "2. Use appropriate tools to get that information\n"
        agent_input += "3. Continue reasoning and acting until you have sufficient information\n"
        agent_input += "4. Call task_done when ready to provide your complete findings\n\n"
        agent_input += "Begin with your initial reasoning about the task."
        agent_input += " /no_think"
        context["agent_id"] = f"{information_seeker.name}-{str(task_id)}"

        result = await Ddv2AgentRunner.run(starting_agent=information_seeker, input=agent_input, max_turns=config.SEEKER_MAX_TURNS, context=context, conversation_id=session_id)
        return result.final_output
    coroutines = [process_task_async(task,idx) for idx, task in enumerate(tasks)]
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    return results


@function_tool(strict_mode=False)
async def assign_multi_subjective_tasks_to_info_seeker(
        ctx: RunContextWrapper[Any],
        tasks: List[Dict[str, str]],
        max_workers: int = 4
) -> Any:
    """
    Creates multiple TaskInput objects and routes them to info_seeker agents for concurrent execution.
    This tool enables the PlannerAgent to assign multiple research tasks through the MCP tool interface.

    Args:
        context: any
        tasks: List of task dictionaries with the following keys:
            - task_content (required): The specific task content
            - task_steps_for_reference: Optional reference steps for execution
            - deliverable_contents: Format of expected deliverable
            - acceptance_checking_criteria: Criteria for task completion and quality
            - workspace_id: Workspace ID for stored files and memory
            - current_task_status: Description of current task status
        max_workers: Maximum concurrent threads (default=4)

    Returns:
        MCPToolResult with execution results for all tasks
    """

    semaphore = asyncio.Semaphore(max_workers)
    if isinstance(ctx.context, tuple):
        session_id = ctx.context[0].get("session_id")
    elif isinstance(ctx.context, dict):
        # {"session_id": session_id},  # 注入会话ID
        session_id = ctx.context.get("session_id")
    context = {"session_id": session_id}

    async def process_task_async(task: Dict[str, str], task_id:int) -> Dict[str, Any]:
        information_seeker = create_child_agent(
            subjective_information_seeker_agent, mcp_server_subjective)
        agent_input = str(task)
        agent_input += "\nPlease analyze this task and start your ReAct process:\n"
        agent_input += "1. Reason about what information you need to gather\n"
        agent_input += "2. Use appropriate tools to get that information\n"
        agent_input += "3. Continue reasoning and acting until you have sufficient information\n"
        agent_input += "4. Call info_seeker_subjective_task_done when ready to provide your complete findings\n\n"
        agent_input += "Begin with your initial reasoning about the task."
        agent_input += " /no_think"
        context["agent_id"] = f"{information_seeker.name}-{str(task_id)}"
        result = await Ddv2AgentRunner.run(starting_agent=information_seeker, input=agent_input, max_turns=config.SEEKER_MAX_TURNS, context=context, conversation_id=session_id)
        return result.final_output
    coroutines = [process_task_async(task,idx) for idx,task in enumerate(tasks)]
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    return results


@function_tool(strict_mode=False)
async def assign_subjective_task_to_writer(
        ctx: RunContextWrapper[Any],
        task_content: str,
        user_query: str,
        key_files: List[Dict[str, str]]
) -> Any:
    """
    Assign a writing or content creation task to the WriterAgent

    Args:
        context: any
        task_content: Detailed description of the writing task to be performed
        user_query: List storing previous information seeker subtask summaries intact to preserve information from each completed research task
        key_files: Curated list of relevant files with file_path and desc for each file
    Returns:
        Dictionary with task assignment results
    """
    if isinstance(ctx.context, tuple):
        session_id = ctx.context[0].get("session_id")
    elif isinstance(ctx.context, dict):
        # {"session_id": session_id},  # 注入会话ID
        session_id = ctx.context.get("session_id")
    context = {"session_id": session_id}
    agent_input = f"""task_content: {task_content}
                        user_query: {user_query}
                        key_files: {key_files}"""
    agent_input += " /no_think"
    writer = create_child_agent(writer_agent, mcp_server_write)
    context["agent_id"] = writer.name
    result = await Ddv2AgentRunner.run(starting_agent=writer, input=agent_input, max_turns=config.WRITER_MAX_TURNS, context=context, conversation_id=session_id)
    return result.final_output


async def run(mcp_server_planner: MCPServer):
    planner = planner_agent.clone(
        mcp_servers=[mcp_server_planner],
        tools=[assign_subjective_task_to_writer, assign_multi_subjective_tasks_to_info_seeker,
               assign_multi_objective_tasks_to_info_seeker],
    )
    session_id = f"session_{uuid.uuid4().hex}"
    run_config = RunConfig()
    run_config.trace_id = f"trace_{uuid.uuid4().hex}"
    # add_trace_processor(SSEProfessor)
    # if session_id is not None:
    #     SSEProfessor.bind_trace_to_session(run_config.trace_id, session_id)

    context = {"session_id": session_id, "agent_id": planner.name}
    start_time = time.perf_counter()
    async with httpx.AsyncClient(verify=False):
        while True:
            query = input("query: ")
            query = query.encode('utf-8', 'replace').decode('utf-8')
            logger.info(f"\n\nRunning code:\n{'-' * 50}")
            logger.info(query)
            logger.info('-' * 50)
            agent_input = query + "/no_think"
            result = await Ddv2AgentRunner.run(starting_agent=planner, input=agent_input, max_turns=config.PLANNER_MAX_TURNS, context=context, conversation_id=session_id, run_config=run_config)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            final_result = {
                "data": result.final_output,
                "agent_name": result.last_agent.name,
                "execution_time": execution_time
            }
            logger.info(f'\nExecution Result:{final_result}' )

async def start_deepdiver():
    """
    The main async function that initializes the servers and runs the Deepdiver Manager.
    """
    host = config.MCP_HOST
    port = config.MCP_PORT
    planner_mode = config.PLANNER_MODE
    planner_mode_tool = config.MODE_TOOL_NAMES[planner_mode]
    async with AsyncExitStack() as stack:
        global mcp_server_planner
        mcp_server_planner = await stack.enter_async_context(MCPServerStreamableHttp(
            name="Planner Server",
            params={"url": f"http://{host}:{port}/mcp"},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=config.PLANNER_ALLOWED_TOOL_NAMES + planner_mode_tool
            ),
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
            cache_tools_list=True,
        ))
        global mcp_server_objective
        mcp_server_objective = await stack.enter_async_context(MCPServerStreamableHttp(
            name="Objective Server",
            params={"url": f"http://{host}:{port}/mcp"},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=config.OBJECTIVE_SEEKER_ALLOWED_TOOL_NAMES
            ),
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
            cache_tools_list=True,
        ))
        global mcp_server_subjective
        mcp_server_subjective = await stack.enter_async_context(MCPServerStreamableHttp(
            name="Subjective Server",
            params={"url": f"http://{host}:{port}/mcp"},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=config.SUBJECTIVE_SEEKER_ALLOWED_TOOL_NAMES
            ),
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
            cache_tools_list=True,
        ))
        global mcp_server_write
        mcp_server_write = await stack.enter_async_context(MCPServerStreamableHttp(
            name="Write Server",
            params={"url": f"http://{host}:{port}/mcp"},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=config.WRITER_ALLOWED_TOOL_NAMES
            ),
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
            cache_tools_list=True,
        ))
        with trace(workflow_name="DeepDiver"):
            await run(
                mcp_server_planner
            )

async def start_deepdiver_v2():
    """
    The main async function that initializes the servers and runs the Deepdiver Manager.
    """
    print("Starting Deepdiver Manager")
    host = config.MCP_HOST
    port = config.MCP_PORT
    planner_mode = config.PLANNER_MODE
    planner_mode_tool = config.MODE_TOOL_NAMES[planner_mode]
    async with AsyncExitStack() as stack:
        global mcp_server_planner
        mcp_server_planner = await stack.enter_async_context(MCPServerStreamableHttp(
            name="Planner Server",
            params={"url": f"http://{host}:{port}/mcp"},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=config.PLANNER_ALLOWED_TOOL_NAMES + planner_mode_tool
            ),
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
            cache_tools_list=True,
        ))
        global mcp_server_planner2
        mcp_server_planner2 = await stack.enter_async_context(MCPServerStreamableHttp(
            name="Planner Server",
            params={"url": f"http://{host}:{port+1}/mcp"},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=config.PLANNER_ALLOWED_TOOL_NAMES + planner_mode_tool
            ),
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
            cache_tools_list=True,
        ))
        global mcp_server_objective
        mcp_server_objective = await stack.enter_async_context(MCPServerStreamableHttp(
            name="Objective Server",
            params={"url": f"http://{host}:{port}/mcp"},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=config.OBJECTIVE_SEEKER_ALLOWED_TOOL_NAMES
            ),
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
            cache_tools_list=True,
        ))
        global mcp_server_objective2
        mcp_server_objective2 = await stack.enter_async_context(MCPServerStreamableHttp(
            name="Objective Server",
            params={"url": f"http://{host}:{port+1}/mcp"},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=config.OBJECTIVE_SEEKER_ALLOWED_TOOL_NAMES
            ),
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
            cache_tools_list=True,
        ))
        global mcp_server_subjective
        mcp_server_subjective = await stack.enter_async_context(MCPServerStreamableHttp(
            name="Subjective Server",
            params={"url": f"http://{host}:{port}/mcp"},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=config.SUBJECTIVE_SEEKER_ALLOWED_TOOL_NAMES
            ),
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
            cache_tools_list=True,
        ))
        global mcp_server_subjective2
        mcp_server_subjective2 = await stack.enter_async_context(MCPServerStreamableHttp(
            name="Subjective Server",
            params={"url": f"http://{host}:{port+1}/mcp"},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=config.SUBJECTIVE_SEEKER_ALLOWED_TOOL_NAMES
            ),
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
            cache_tools_list=True,
        ))
        global mcp_server_write
        mcp_server_write = await stack.enter_async_context(MCPServerStreamableHttp(
            name="Write Server",
            params={"url": f"http://{host}:{port}/mcp"},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=config.WRITER_ALLOWED_TOOL_NAMES
            ),
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
            cache_tools_list=True,
        ))
        global mcp_server_write2
        mcp_server_write2 = await stack.enter_async_context(MCPServerStreamableHttp(
            name="Write Server",
            params={"url": f"http://{host}:{port+1}/mcp"},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=config.WRITER_ALLOWED_TOOL_NAMES
            ),
            client_session_timeout_seconds=None,
            max_retry_attempts=60,
            retry_backoff_seconds_base=5,
            cache_tools_list=True,
        ))

        mcp_server_1 = str(host) + ":" + str(port)
        mcp_server_2 = str(host) + ":" + str(port + 1)
        mcp_hub_v2.mcp_server_manager.add_server(mcp_server_1)
        mcp_hub_v2.mcp_server_manager.add_server(mcp_server_2)

        planner_agent = McpHubServer('PlannerAgent')
        planner_agent.mcp_client_list.append(MCPClient(mcp_server_planner, "mcp_server_planner"))
        planner_agent.server_to_servername[mcp_server_planner] = mcp_server_1
        planner_agent.servername_to_server[mcp_server_1] = mcp_server_planner
        planner_agent.mcp_client_list.append(MCPClient(mcp_server_planner2, "mcp_server_planner2"))
        planner_agent.server_to_servername[mcp_server_planner2] = mcp_server_2
        planner_agent.servername_to_server[mcp_server_2] = mcp_server_planner2
        planner_agent.load_balancer = LoadBalancer(planner_agent.mcp_client_list)

        objective_agent = McpHubServer('ObjectiveInformationSeekerAgent')
        objective_agent.mcp_client_list.append(MCPClient(mcp_server_objective, "mcp_server_objective"))
        objective_agent.server_to_servername[mcp_server_objective] = mcp_server_1
        objective_agent.servername_to_server[mcp_server_1] = mcp_server_objective
        objective_agent.mcp_client_list.append(MCPClient(mcp_server_objective2, "mcp_server_objective2"))
        objective_agent.server_to_servername[mcp_server_objective2] = mcp_server_2
        objective_agent.servername_to_server[mcp_server_2] = mcp_server_objective2
        objective_agent.load_balancer = LoadBalancer(objective_agent.mcp_client_list)

        subjective_agent = McpHubServer('SubjectiveInformationSeekerAgent')
        subjective_agent.mcp_client_list.append(MCPClient(mcp_server_subjective, "mcp_server_subjective"))
        subjective_agent.server_to_servername[mcp_server_subjective] = mcp_server_1
        subjective_agent.servername_to_server[mcp_server_1] = mcp_server_subjective
        subjective_agent.mcp_client_list.append(MCPClient(mcp_server_subjective2, "mcp_server_subjective2"))
        subjective_agent.server_to_servername[mcp_server_subjective2] = mcp_server_2
        subjective_agent.servername_to_server[mcp_server_2] = mcp_server_subjective2
        subjective_agent.load_balancer = LoadBalancer(subjective_agent.mcp_client_list)

        writer_agent = McpHubServer('WriterAgent')
        writer_agent.mcp_client_list.append(MCPClient(mcp_server_write, "mcp_server_write"))
        writer_agent.server_to_servername[mcp_server_write] = mcp_server_1
        writer_agent.servername_to_server[mcp_server_1] = mcp_server_write
        writer_agent.mcp_client_list.append(MCPClient(mcp_server_write2, "mcp_server_write2"))
        writer_agent.server_to_servername[mcp_server_write2] = mcp_server_2
        writer_agent.servername_to_server[mcp_server_2] = mcp_server_write2
        writer_agent.load_balancer = LoadBalancer(writer_agent.mcp_client_list)

        mcp_hub_v2.stateless_server['PlannerAgent'] = planner_agent
        mcp_hub_v2.stateless_server['ObjectiveInformationSeekerAgent'] = objective_agent
        mcp_hub_v2.stateless_server['SubjectiveInformationSeekerAgent'] = subjective_agent
        mcp_hub_v2.stateless_server['WriterAgent'] = writer_agent

        with trace(workflow_name="DeepDiver"):
            await run(
                mcp_server_planner
            )