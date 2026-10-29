import json
import os
import queue
import shutil
import threading
from dataclasses import field, dataclass
from datetime import datetime
import time
from functools import partial
from typing import Optional, List, AsyncGenerator, Dict, Set
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# 依赖openai-agents-sdk（需提前安装）
from contextlib import AsyncExitStack
import sys
import uuid
from pathlib import Path
import copy
from sse_starlette import EventSourceResponse

import config.config as config
from agent_infer.agent_specs.agent_hooks import setup_hooks_logger
import asyncio
from agents.mcp import MCPServerStreamableHttp, create_static_tool_filter
from typing import Any, List, Dict
from agent_infer.agent_specs.base_agent import IDAgent
from agent_infer.agent_specs.deepdiver_v2.writer import writer_agent
from agent_infer.agent_specs.deepdiver_v2.objective_information_seeker import objective_information_seeker_agent
from agent_infer.agent_specs.deepdiver_v2.subjective_information_seeker import subjective_information_seeker_agent
from agent_infer.agent_specs.deepdiver_v2.planner import planner_agent
from agents.run_context import RunContextWrapper
from agents import function_tool, RunConfig
from contextlib import AsyncExitStack
from examples.deepdiver_v2.deepdiver_v2_agent_runner import Ddv2AgentRunner
import sys
from pathlib import Path
import config.config as config
import uuid
from examples.deepdiver_v2.deepdiver_v2_agent_runner import Ddv2Runner
from examples.deepdiver_v2.deepdiver_v2_manager import assign_subjective_task_to_writer, \
    assign_multi_subjective_tasks_to_info_seeker, assign_multi_objective_tasks_to_info_seeker
from examples.deepdiver_v2.tracing.processor import register_session, event_generator, unregister_session, \
    send_sse_stop_signal, SSEProfessor
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request  # 👈 正确导入方式
from examples.deepdiver_v2.server.session import SessionManager, sessionManager, get_window_config, set_window_config, apply_config_overrides, restore_config, _original_config_store

from examples.deepdiver_v2.extra.mcp_hub import MCPClient, McpHubServer, LoadBalancer

from examples.deepdiver_v2.extra.server_config import mcp_hub_v2

from features.compression.config import CompressionConfig

from examples.deepdiver_v2.extra.storage import upload_session_to_dist_storage

# from examples.deepdiver_v2.processor import register_session, event_generator

# todo app.py统一；请求+流式；log(摘日志)+全局异常；取消router，viewer，error。
# 0. 全局日志
workspace_path = Path(config.WORKSPACE_PATH)
workspace_path.mkdir(exist_ok=True)
hooks_logger = setup_hooks_logger(
    f'{config.WORKSPACE_PATH}/AgenticAppLogs.log')

# 1. 初始化FastAPI应用
app_ddv2 = FastAPI(
    title="DeepDiver v2 Agent API",
    version="1.0",
    description="基础Agent服务：支持启动Agent、流式响应、内存历史对话（无数据库/无验证）"
)
app_ddv2.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Umi 默认端口
        allow_methods=["*"],
        allow_headers=["*"],
    )

AGENTS_REGISTRY: Dict[str, IDAgent] = {
    "assistant": planner_agent,
}

from asyncio import Queue, Task


# 3. 历史对话存储（列表形式，每个元素是一条对话记录）
CONVERSATION_HISTORY: List[Dict[str, str | float | bool]] = []

# 4. 简化版请求/响应模型

# todo TTL clean up
class AgentRequest(BaseModel):
    """用户启动Agent的请求参数"""
    user_input: str  # 必填：用户需要Agent处理的内容（如"解释什么是FastAPI"）
    agent_name: str = "assistant"  # 可选：使用的Agent（默认assistant）
    stream: bool = False  # 可选：是否流式返回（默认关闭，直接要完整结果）
    session_id: str = None
    trace_id:str = None
    concurrency:int=1
    user_token:str = None
    config_overrides: Dict[str, Any] = {}  # 可选：配置覆盖参数，来自前端modal设置的开关


class AgentResponse(BaseModel):
    """Agent处理后的响应结果"""
    task_id: str  # 唯一任务ID（毫秒级时间戳生成）
    content: str  # Agent返回的内容（或流式任务提示）
    completed: bool  # 任务是否完成（流式任务启动时为False）
    timestamp: float  # 任务时间戳（UTC秒级）


class HistoryItem(BaseModel):
    """历史对话记录的响应模型"""
    task_id: str
    user_input: str
    agent_name: str
    agent_response: str
    timestamp: float
    completed: bool


def cleanup_directories():
    """清空 workspace 和 logs 目录下的所有内容（保留目录本身）"""
    SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
    WORKSPACE_DIR = SCRIPT_DIR / "workspace"
    LOGS_DIR = SCRIPT_DIR / "logs"

    def _clean_dir(root):
        if not root.exists():
            print(f"[Cleanup] Skipped (not exists): {root}")
            return
        for item in root.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                print(f"[Cleanup] Removed: {item}")
            except Exception as e:
                print(f"[Cleanup] ERROR removing {item}: {e}")

    print("[Cleanup] Starting weekly cleanup...")
    _clean_dir(WORKSPACE_DIR)
    _clean_dir(LOGS_DIR)
    print("[Cleanup] Weekly cleanup completed.")


def start_cleanup_scheduler():
    """启动清理调度器（作为守护线程）"""
    t = threading.Thread(target=schedule_weekly_cleanup, daemon=True)
    t.start()
    print("[Cleanup] Weekly cleanup scheduler started (runs every Saturday at 02:00).")


def schedule_weekly_cleanup():
    """
    后台线程：每小时检查一次，如果当前是周六凌晨 2:00，则执行清理。
    """
    while True:
        now = datetime.now()
        # 判断是否为周六（weekday() == 5）且小时为 2，分钟为 0
        if now.weekday() == 5 and now.hour == 2 and now.minute == 0:
            cleanup_directories()
            # 等待 61 秒，避免同一分钟内重复触发
            time.sleep(61)
        else:
            # 每 60 秒检查一次
            time.sleep(60)

# def schedule_daily_cleanup():
#     """
#     后台线程：每分钟检查一次，如果当前是每天中午 12:00，则执行清理。
#     """
#     while True:
#         now = datetime.now()
#         # 判断是否为中午 12:00（小时=12，分钟=0）
#         if now.hour == 12 and now.minute == 0:
#             cleanup_directories()
#             # 等待 61 秒，避免同一分钟内重复触发
#             time.sleep(61)
#         else:
#             # 每 60 秒检查一次
#             time.sleep(60)

def create_child_agent(agent, mcp_server):
    return agent.clone(
        mcp_servers=[mcp_server]  # 复用父agent的MCP服务，而非重新创建
    )


async def start_agent(request: AgentRequest) -> AgentResponse:
    """
    核心接口：启动指定Agent处理用户输入
    - 非流式（stream=False）：直接返回完整结果，并存入历史
    - 流式（stream=True）：返回任务ID，需通过/stream/{task_id}获取实时结果
    """
    session_id = request.session_id
    trace_id = request.trace_id
    task_id = trace_id
    try:
        run_config = RunConfig()
        run_config.trace_id = trace_id
        # run_config.trace_metadata = {"Async"}

        # 🔑 关键修改：将配置副本保存到 context 中，用于模型路由等
        # 深拷贝 config_overrides 避免后续修改影响
        context_config = (request.config_overrides or {}).copy()
        # 用 config 模块的默认值补齐 config_overrides 中未设置的开关，
        # 否则 context_config.get('resum_compress_async') 在空 overrides 时返回 None，
        # 导致 CompressionConfig 不创建、async_compress 永远不触发
        for key in ('resum_compress_async', 'early_stop_enabled', 'model_routing_switch',
                     'mcp_hub_switch', 'tool_predict_switch', 'search_compress_sync_switch'):
            if key not in context_config:
                default_val = getattr(config, key, None)
                if default_val is not None:
                    context_config[key] = default_val
        context = {"session_id": session_id, "config": context_config}
        context.update({
            "compression_config": CompressionConfig(
                enabled=bool(context_config.get('resum_compress_async')),
                token_threshold=int(context_config.get('TOKEN_THRESHOLD', 10)),
                compress_llm_url=(_llm_cfg := config.get_search_compress_model_config()).get('url'),
                compress_llm_api_key=_llm_cfg.get('api_key'),
                compress_llm_model_name=_llm_cfg.get('model_name', 'pangu_auto'),
                compress_llm_temperature=_llm_cfg.get('temperature', 0.6),
                compress_llm_max_tokens=_llm_cfg.get('max_tokens', 8192),
            ) if context_config.get('resum_compress_async') else None
        })
        current_time = datetime.utcnow().timestamp()
        target_agent = AGENTS_REGISTRY[request.agent_name].clone(
            mcp_servers=[mcp_server_planner],
            tools=[assign_subjective_task_to_writer, assign_multi_subjective_tasks_to_info_seeker,
                   assign_multi_objective_tasks_to_info_seeker],
        )

        hooks_logger.info(
            f'task_id: {session_id}; user_input: {request.user_input}; agent_name: {request.agent_name}')
        print(f"task_id: {session_id}")
        # 3. 非流式处理：同步调用Agent，返回完整结果
        # 调用openai-agents的同步运行方法
        agent_result = await Ddv2Runner.run(starting_agent=target_agent, input=request.user_input, max_turns=config.PLANNER_MAX_TURNS, context=context, conversation_id=session_id,  run_config=run_config)
        # 给结果存入agent_result
        hooks_logger.info(
            f'task_id: {session_id}; agent_result: {agent_result}')


        # 存入历史对话
        CONVERSATION_HISTORY.append({
            "task_id": session_id,
            "user_input": request.user_input,
            "agent_name": request.agent_name,
            "agent_response": agent_result.final_output,
            "timestamp": current_time,
            "completed": True
        })
        # 返回结果
        return AgentResponse(
            task_id=session_id,
            content=agent_result.final_output,
            completed=True,
            timestamp=current_time
        )
    except Exception as e:
        hooks_logger.exception(f'Exception occurred in function start_agent')
        print(
            f"[agent_task] failed: task_id={task_id} type={type(e).__name__}",
            flush=True,
        )
        raise e
    finally:
        # 任务结束： 资源清理：关闭sse， 更新session状态
        print(f"end task (task_id: {session_id}/{task_id})")
        session = sessionManager.get_session_by_session_id(session_id)
        if session is not None:
            session.end_task(task_id)
            dist_storage_disabled = os.getenv(
                "DISABLE_DISTRIBUTED_STORAGE", "false"
            ).strip().lower() in {"1", "true", "yes", "on"}
            if (
                session.running_tasks <= 0 or session.completed_at is not None
            ) and not dist_storage_disabled:
                upload_session_to_dist_storage(session.session_id)
            elif dist_storage_disabled:
                print(
                    "[session_upload] skipped: DISABLE_DISTRIBUTED_STORAGE=true",
                    flush=True,
                )



@function_tool(strict_mode=False)
async def assign_multi_objective_tasks_to_info_seeker(
        ctx: RunContextWrapper[Any],
        tasks: List[Dict[str, str]],
        # session_id: str,
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
        session_id: The conversation ID
        max_workers: Maximum concurrent threads (default=4)

    Returns:
        MCPToolResult with execution results for all tasks
    """
    print(f'ctx: {ctx}')
    semaphore = asyncio.Semaphore(max_workers)
    if isinstance(ctx.context, tuple):
        session_id = ctx.context[0].get("session_id")
    elif isinstance(ctx.context, dict):
        # {"session_id": session_id},  # 注入会话ID
        session_id = ctx.context.get("session_id")

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

        # 🔑 从父 context 获取配置副本，传递给子任务
        config_config = ctx.context.get("config", {}).copy()
        context = {"session_id": session_id, "config": config_config}
        context.update({
            "compression_config": ctx.context.get("compression_config"),
            "current_model": "small_model"  # seeker 默认用小模型
        })
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
        # session_id: str,
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
        session_id: The conversation ID
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

    async def process_task_async(task: Dict[str, str],task_id:int) -> Dict[str, Any]:
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

        # 🔑 从父 context 获取配置副本，传递给子任务
        config_config = ctx.context.get("config", {}).copy()
        context = {"session_id": session_id, "config": config_config}
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
        # 🔑 从父 context 获取配置副本，传递给子任务
        config_config = ctx.context.get("config", {}).copy()
        context = {"session_id": session_id, "config": config_config}
        context.update({
            "compression_config": ctx.context.get("compression_config"),
            "current_model": "small_model"  # seeker 默认用小模型
        })
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
        # session_id: str,
        key_files: List[Dict[str, str]]
) -> Any:
    """
    Assign a writing or content creation task to the WriterAgent

    Args:
        context: any
        task_content: Detailed description of the writing task to be performed
        user_query: List storing previous information seeker subtask summaries intact to preserve information from each completed research task
        session_id: The conversation ID
        key_files: Curated list of relevant files with file_path and desc for each file
    Returns:
        Dictionary with task assignment results
    """
    if isinstance(ctx.context, tuple):
        session_id = ctx.context[0].get("session_id")
    elif isinstance(ctx.context, dict):
        # {"session_id": session_id},  # 注入会话ID
        session_id = ctx.context.get("session_id")
    # 🔑 从父 context 获取配置副本，传递给子任务
    config_config = ctx.context.get("config", {}).copy()
    context = {"session_id": session_id, "config": config_config}
    agent_input = f"""task_content: {task_content}
                        user_query: {user_query}
                        key_files: {key_files}"""
    agent_input += " /no_think"
    writer = create_child_agent(writer_agent, mcp_server_write)
    context["agent_id"] = writer.name
    result = await Ddv2AgentRunner.run(starting_agent=writer, input=agent_input, max_turns=config.WRITER_MAX_TURNS, context=context, conversation_id=session_id)
    return result.final_output

async def start_deepdiver_v2():
    """
    The main async function that initializes the servers and runs the Deepdiver Manager.
    """
    print("Starting Deepdiver Manager")
    host = config.MCP_HOST
    port = config.MCP_PORT
    planner_mode = config.PLANNER_MODE
    planner_mode_tool = config.MODE_TOOL_NAMES[planner_mode]
    global stack
    stack = AsyncExitStack()
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
    planner_agent.mcp_client_list.append(MCPClient(mcp_server_planner,"mcp_server_planner"))
    planner_agent.server_to_servername[mcp_server_planner] = mcp_server_1
    planner_agent.servername_to_server[mcp_server_1] = mcp_server_planner
    planner_agent.mcp_client_list.append(MCPClient(mcp_server_planner2,"mcp_server_planner2"))
    planner_agent.server_to_servername[mcp_server_planner2] = mcp_server_2
    planner_agent.servername_to_server[mcp_server_2] = mcp_server_planner2
    planner_agent.load_balancer = LoadBalancer(planner_agent.mcp_client_list)

    objective_agent = McpHubServer('ObjectiveInformationSeekerAgent')
    objective_agent.mcp_client_list.append(MCPClient(mcp_server_objective,"mcp_server_objective"))
    objective_agent.server_to_servername[mcp_server_objective] = mcp_server_1
    objective_agent.servername_to_server[mcp_server_1] = mcp_server_objective
    objective_agent.mcp_client_list.append(MCPClient(mcp_server_objective2,"mcp_server_objective2"))
    objective_agent.server_to_servername[mcp_server_objective2] = mcp_server_2
    objective_agent.servername_to_server[mcp_server_2] = mcp_server_objective2
    objective_agent.load_balancer = LoadBalancer(objective_agent.mcp_client_list)

    subjective_agent = McpHubServer('SubjectiveInformationSeekerAgent')
    subjective_agent.mcp_client_list.append(MCPClient(mcp_server_subjective,"mcp_server_subjective"))
    subjective_agent.server_to_servername[mcp_server_subjective] = mcp_server_1
    subjective_agent.servername_to_server[mcp_server_1] = mcp_server_subjective
    subjective_agent.mcp_client_list.append(MCPClient(mcp_server_subjective2,"mcp_server_subjective2"))
    subjective_agent.server_to_servername[mcp_server_subjective2] = mcp_server_2
    subjective_agent.servername_to_server[mcp_server_2] = mcp_server_subjective2
    subjective_agent.load_balancer = LoadBalancer(subjective_agent.mcp_client_list)

    writer_agent = McpHubServer('WriterAgent')
    writer_agent.mcp_client_list.append(MCPClient(mcp_server_write,"mcp_server_write"))
    writer_agent.server_to_servername[mcp_server_write] = mcp_server_1
    writer_agent.servername_to_server[mcp_server_1] = mcp_server_write
    writer_agent.mcp_client_list.append(MCPClient(mcp_server_write2,"mcp_server_write2"))
    writer_agent.server_to_servername[mcp_server_write2] = mcp_server_2
    writer_agent.servername_to_server[mcp_server_2] = mcp_server_write2
    writer_agent.load_balancer = LoadBalancer(writer_agent.mcp_client_list)

    mcp_hub_v2.stateless_server['PlannerAgent'] = planner_agent
    mcp_hub_v2.stateless_server['ObjectiveInformationSeekerAgent'] = objective_agent
    mcp_hub_v2.stateless_server['SubjectiveInformationSeekerAgent'] = subjective_agent
    mcp_hub_v2.stateless_server['WriterAgent'] = writer_agent



async def start_deepdiver():
    host = config.MCP_HOST
    port = config.MCP_PORT
    planner_mode = config.PLANNER_MODE
    planner_mode_tool = config.MODE_TOOL_NAMES[planner_mode]
    global stack
    stack = AsyncExitStack()
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

@app_ddv2.on_event("startup")
async def startup_event():
    """应用启动时初始化MCP长连接"""
    try:
        start_cleanup_scheduler()
        if config.mcp_hub_switch:
            await start_deepdiver_v2()
        else:
            await start_deepdiver()
    except Exception as e:
            print(f"MCP连接初始化失败: {str(e)}")
            await stack.aclose()  # 清理资源
            raise  # 终止启动流程


@app_ddv2.on_event("shutdown")
async def shutdown_event():
    """应用关闭时释放MCP连接资源"""
    global stack
    if stack:
        await stack.aclose()  # 退出所有上下文，释放连接




@app_ddv2.post("/start_agents", response_model=Dict, summary="启动Agent处理请求")
async def start_agents(request: AgentRequest):
    # 1. 生成唯一任务ID（用UTC毫秒级时间戳，避免重复）
    session_id = f"session_{uuid.uuid4().hex}"
    print(f"session_id: {session_id}")
    register_session(session_id)
    trace_id = f"trace_{uuid.uuid4().hex}"
    print(f"trace_id: {trace_id}")
    # 2. 提前注册队列
    SSEProfessor.bind_trace_to_session(trace_id, session_id)
    request.session_id = session_id
    request.trace_id = trace_id
    # 2. 启动 agent（异步，不阻塞 yield）
    async def run_in_background(request: AgentRequest):
        try:
            # ✅ 注意：start_agent 必须是 async 或不阻塞
            response = await start_agent(request)
        except Exception as e:
            pass

    # 开一个后台任务跑 agent
    asyncio.create_task(run_in_background(
        request=request
    ))
    return {"session_id": session_id, "status": "started"}


@app_ddv2.post("/api/submit")
async def submit_job(request: AgentRequest):
    concurrency = request.concurrency
    user_token = request.user_token
    agent_name = request.agent_name
    input_data = request.user_input
    if not concurrency:
        concurrency = 1
    if not user_token:
        return {"error": "user_token is required"}  # 或自动生成（不推荐）
    if agent_name not in AGENTS_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Agent不存在！当前可用Agent：{list(AGENTS_REGISTRY.keys())}"
        )
    if not input_data or not (1 <= concurrency <= 50):
        return {"error": "Invalid input"}

    if concurrency < 1 or concurrency > 50:
        return {"error": "concurrency must be between 1 and 50"}

    # 1. 生成唯一任务ID（用UTC毫秒级时间戳，避免重复）
    session = sessionManager.create_session(user_token, concurrency)
    print(f"session_id: {session.session_id}")
    register_session(session.session_id)

    # 2. 如果请求中没有配置覆盖，从窗口配置中获取
    if not request.config_overrides:
        request.config_overrides = get_window_config(user_token)

    # 打印当前会话的配置覆盖
    print(f"=== Config Debug for user {user_token} ===")
    print(f"Request config_overrides: {request.config_overrides}")
    if request.config_overrides:
        print(f"  search_compress_sync_switch = {request.config_overrides.get('search_compress_sync_switch', 'NOT_FOUND')}")
        print(f"  resum_compress_async = {request.config_overrides.get('resum_compress_async', 'NOT_FOUND')}")
        print(f"  LARGE_MODEL_URL = {request.config_overrides.get('LARGE_MODEL_URL', 'NOT_FOUND')}")
        print(f"  SMALL_MODEL_URL = {request.config_overrides.get('SMALL_MODEL_URL', 'NOT_FOUND')}")
    else:
        print(f"  No config overrides for this session")

    # 4. 启动 agent（异步，不阻塞 yield）
    async def run_in_background(request: AgentRequest):
        try:
            response = await start_agent(request)
        except Exception as e:
            raise e

    # 启动并发 traces
    trace_ids = []
    for i in range(concurrency):
        request_copy = copy.deepcopy(request)
        trace_id = f"trace_{session.session_id}_t{i}"
        trace_ids.append(trace_id)
        print(f"trace_id: {trace_id}")
        # 2. 提前注册队列
        SSEProfessor.bind_trace_to_session(trace_id, session.session_id)
        request_copy.session_id = session.session_id
        request_copy.trace_id = trace_id
        bound_func = partial(run_in_background, request_copy)  # 绑定的是地址，必须为每次调用深拷贝一个request
        session.create_task(bound_func, request_copy, i, trace_id)
    hooks_logger.info(f"🚀 Started submit {session.session_id} for user={user_token}, concurrency={concurrency}")
    return {"session_id": session.session_id, "trace_ids": trace_ids ,"status": "started"}


@app_ddv2.get("/api/tracestream/{session_id}")
@app_ddv2.get("/tracestream/{session_id}")
async def tracestream(session_id: str):
    print(f"Streaming for session: {session_id}")
    # 启动前确保已注册（或等待 agent 启动）
    register_session(session_id)  # 可选：允许提前监听

    return EventSourceResponse(event_generator(session_id))


@app_ddv2.get("/history", response_model=List[HistoryItem], summary="获取历史对话")
async def get_conversation_history(task_id: Optional[str] = None):
    """
    查询历史对话：
    - 不传task_id：返回所有历史（按时间倒序，最新在前）
    - 传task_id：仅返回该任务的历史记录
    """
    if task_id:
        # 按任务ID过滤
        filtered_history = [
            item for item in CONVERSATION_HISTORY if item["task_id"] == task_id]
        return filtered_history
    # 全量历史：按时间戳倒序
    return sorted(CONVERSATION_HISTORY, key=lambda x: x["timestamp"], reverse=True)

# 全局统计
@app_ddv2.get("/api/stats")
async def stats():
    return sessionManager.get_system_stats()

@app_ddv2.get("/api/stop/{session_id}")
async def stop(session_id: str):
    return sessionManager.stop_session(session_id)

# 网络问题/并发问题
# ──────────────── 窗口配置接口 ────────────────
@app_ddv2.get("/api/window/config/{user_token}")
async def get_window_config_endpoint(user_token: str):
    """获取指定窗口的配置"""
    from examples.deepdiver_v2.server.session import SESSION_CONFIG_STORE

    user_config = get_window_config(user_token)
    # 判断是否为用户自定义配置（检查 SESSION_CONFIG_STORE 中是否存在该用户）
    is_user_customized = user_token in SESSION_CONFIG_STORE and bool(SESSION_CONFIG_STORE[user_token])

    if user_config:
        user_config['_is_default'] = False
        return user_config
    # 如果用户没有配置，返回后端的默认配置，并标记为默认
    return {
        '_is_default': True,
        'mcp_hub_switch': config.mcp_hub_switch,
        'tool_predict_switch': config.tool_predict_switch,
        'resum_compress_async': config.resum_compress_async,
        'early_stop_enabled': config.early_stop_enabled,
        'model_routing_switch': config.model_routing_switch,
        'search_compress_sync_switch': config.search_compress_sync_switch,
        'ENABLE_MEMORY': config.ENABLE_MEMORY,
        'ENABLE_SHARED_MEMORY': config.ENABLE_SHARED_MEMORY,
        'PLANNER_MODE': config.PLANNER_MODE,
        'model_temperature': config.model_temperature,
        'model_max_tokens': config.model_max_tokens,
        'TOKEN_THRESHOLD': config.TOKEN_THRESHOLD,
        'PLANNER_MAX_TURNS': config.PLANNER_MAX_TURNS,
        'SEEKER_MAX_TURNS': config.SEEKER_MAX_TURNS,
        'WRITER_MAX_TURNS': config.WRITER_MAX_TURNS,
        'early_stop_min_iter': config.early_stop_min_iter,
    }


@app_ddv2.post("/api/window/config")
async def set_window_config_endpoint(request: Request):
    """设置指定窗口的配置"""
    body = await request.json()
    user_token = body.get("user_token")
    config_overrides = body.get("config_overrides", {})
    if not user_token:
        return {"error": "user_token is required"}, 400
    set_window_config(user_token, config_overrides)
    return {"status": "ok", "user_token": user_token}


@app_ddv2.post("/api/user/close")
async def user_leave(request: Request):
    body = await request.json()
    user_token = body.get("user_token")
    if user_token:
        # 标记为已离开（可选）
        # user_last_seen[user_token] = -1  # 特殊值表示主动离开sse_stop
        # 或直接清理（但建议统一由后台任务清理）
        sessionManager.cleanup_user_sessions(user_token)
    return {"status": "ok"}



@app_ddv2.get("/stream/{task_id}", response_class=StreamingResponse, summary="流式获取Agent结果")
async def stream_agent_result(request: AgentRequest):
    """
    流式接口：获取指定任务的实时结果（仅支持stream=True启动的任务）
    - 采用SSE（Server-Sent Events）协议，前端可实时接收片段
    - 结束标识为[data: [DONE]]，同时更新历史对话
    """
    session = sessionManager.create_session(request.user_token, request.concurrency)
    context = {"session_id": session.session_id}
    async def stream_generator() -> AsyncGenerator[str, None]:
        full_response = ""
        try:
            stream = Ddv2AgentRunner.run_streamed(
                request.agent_name, request.user_input, context=context)
            async for event in stream.stream_events():
                try:
                    full_response += event.content
                except:
                    pass
                yield f"data: {event}\n\n"
            yield f"data: [DONE]\n\n"

        except Exception as e:
            # 流式异常：返回错误信息，更新历史为失败
            error_msg = f"流式处理失败：{str(e)}"
            yield f"data: {error_msg}\n\n"
            yield "data: [DONE]\n\n"


    # 返回流式响应（媒体类型为text/event-stream）
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# 6. 服务启动入口
if __name__ == "__main__":
    import uvicorn
    # 启动前打印提示信息
    hooks_logger.info("=" * 50)
    hooks_logger.info("✅ 简化版Agent API 启动准备完成")
    hooks_logger.info(f"📌 可用Agent：{list(AGENTS_REGISTRY.keys())}")
    hooks_logger.info(f"📚 API文档：http://localhost:{os.getenv('AGENT_PORT', '8289')}/docs")
    hooks_logger.info(f"⚠️  注意：重启服务后，内存中的历史对话会丢失！")
    hooks_logger.info("=" * 50)


    # 启动FastAPI服务（默认8000端口，开发模式自动重载）
    uvicorn.run(
        app="app_ddv2:app_ddv2",
        host="0.0.0.0",  # 允许外部设备访问
        port=int(os.getenv("AGENT_PORT", "8289")),
        reload=False,
    )
