
from datetime import datetime
from typing import Optional, List, AsyncGenerator, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# 依赖openai-agents-sdk（需提前安装）
from contextlib import AsyncExitStack
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import urllib3.exceptions

# Load environment variables and disable SSL warnings (same as interactive mode)
load_dotenv(find_dotenv(), override=True)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
from agents import function_tool
from contextlib import AsyncExitStack
import sys
from pathlib import Path
import config.config as config
import uuid
from examples.deepdiver_v2.deepdiver_v2_agent_runner import Ddv2Runner as MyRunner
# todo app.py统一；请求+流式；log(摘日志)+全局异常；取消router，viewer，error。
# 0. 全局日志
workspace_path = Path(config.WORKSPACE_PATH)
workspace_path.mkdir(exist_ok=True)
hooks_logger = setup_hooks_logger(
    f'{config.WORKSPACE_PATH}/AgenticAppLogs.log')

# 1. 初始化FastAPI应用
app = FastAPI(
    title="Simplified Agent API",
    version="1.0",
    description="基础Agent服务：支持启动Agent、流式响应、内存历史对话（无数据库/无验证）"
)

AGENTS_REGISTRY: Dict[str, IDAgent] = {
    "assistant": planner_agent,
}

# 3. 历史对话存储（列表形式，每个元素是一条对话记录）
CONVERSATION_HISTORY: List[Dict[str, str | float | bool]] = []

# 4. 简化版请求/响应模型


class AgentRequest(BaseModel):
    """用户启动Agent的请求参数"""
    user_input: str  # 必填：用户需要Agent处理的内容（如“解释什么是FastAPI”）
    agent_name: str = "assistant"  # 可选：使用的Agent（默认assistant）
    stream: bool = False  # 可选：是否流式返回（默认关闭，直接要完整结果）


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

# 5. 核心API接口


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
    print(f'ctx: {ctx}')
    semaphore = asyncio.Semaphore(max_workers)
    if isinstance(ctx.context, tuple):
        session_id = ctx.context[0].get("session_id")
    elif isinstance(ctx.context, dict):
        # {"session_id": session_id},  # 注入会话ID
        session_id = ctx.context.get("session_id")
    context = {"session_id": session_id, "config": {"mcp_hub_switch": False}}

    async def process_task_async(task: Dict[str, str]) -> Dict[str, Any]:
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

        result = await MyRunner.run(starting_agent=information_seeker, input=agent_input, max_turns=config.SEEKER_MAX_TURNS, context=context, conversation_id=session_id)
        return result.final_output
    coroutines = [process_task_async(task) for task in tasks]
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
    context = {"session_id": session_id, "config": {"mcp_hub_switch": False}}

    async def process_task_async(task: Dict[str, str]) -> Dict[str, Any]:
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
        result = await MyRunner.run(starting_agent=information_seeker, input=agent_input, max_turns=config.SEEKER_MAX_TURNS, context=context, conversation_id=session_id)
        return result.final_output
    coroutines = [process_task_async(task) for task in tasks]
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
    context = {"session_id": session_id, "config": {"mcp_hub_switch": False}}
    agent_input = f"""task_content: {task_content}
                        user_query: {user_query}
                        key_files: {key_files}"""
    agent_input += " /no_think"
    writer = create_child_agent(writer_agent, mcp_server_write)
    result = await MyRunner.run(starting_agent=writer, input=agent_input, max_turns=config.WRITER_MAX_TURNS, context=context, conversation_id=session_id)
    return result.final_output


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化MCP长连接"""
    try:
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
    except Exception as e:
        print(f"MCP连接初始化失败: {str(e)}")
        await stack.aclose()  # 清理资源
        raise  # 终止启动流程


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时释放MCP连接资源"""
    global stack
    if stack:
        await stack.aclose()  # 退出所有上下文，释放连接


@app.post("/agents", response_model=AgentResponse, summary="启动Agent处理请求")
async def start_agent(request: AgentRequest):
    """
    核心接口：启动指定Agent处理用户输入
    - 非流式（stream=False）：直接返回完整结果，并存入历史
    - 流式（stream=True）：返回任务ID，需通过/stream/{task_id}获取实时结果
    """
    # 1. 验证请求的Agent是否存在
    if request.agent_name not in AGENTS_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Agent不存在！当前可用Agent：{list(AGENTS_REGISTRY.keys())}"
        )
    task_id = str(uuid.uuid4())
    context = {"session_id": task_id, "config": {"mcp_hub_switch": False}}
    current_time = datetime.utcnow().timestamp()
    target_agent = AGENTS_REGISTRY[request.agent_name].clone(
        mcp_servers=[mcp_server_planner],
        tools=[assign_subjective_task_to_writer, assign_multi_subjective_tasks_to_info_seeker,
               assign_multi_objective_tasks_to_info_seeker],
    )
    # 2. 生成唯一任务ID（用UTC毫秒级时间戳，避免重复）

    hooks_logger.info(
        f'task_id: {task_id}; user_input: {request.user_input}; agent_name: {request.agent_name}')
    # 3. 非流式处理：同步调用Agent，返回完整结果
    if not request.stream:
        # 调用openai-agents的同步运行方法
        agent_result = await MyRunner.run(starting_agent=target_agent, input=request.user_input, max_turns=config.PLANNER_MAX_TURNS, context=context, conversation_id=task_id)
        hooks_logger.info(
            f'task_id: {task_id}; agent_result: {agent_result}')
        # 存入历史对话
        CONVERSATION_HISTORY.append({
            "task_id": task_id,
            "user_input": request.user_input,
            "agent_name": request.agent_name,
            "agent_response": agent_result.final_output,
            "timestamp": current_time,
            "completed": True
        })
        # 返回结果
        return AgentResponse(
            task_id=task_id,
            content=agent_result.final_output,
            completed=True,
            timestamp=current_time
        )

    # 4. 流式处理：先记录任务，返回任务ID（结果通过流式接口获取）
    CONVERSATION_HISTORY.append({
        "task_id": task_id,
        "user_input": request.user_input,
        "agent_name": request.agent_name,
        "agent_response": "",  # 流式结束后填充
        "timestamp": current_time,
        "completed": False
    })
    return AgentResponse(
        task_id=task_id,
        content=f"流式任务已启动，请访问 /stream/{task_id} 获取实时结果",
        completed=False,
        timestamp=current_time
    )


@app.get("/stream/{task_id}", response_class=StreamingResponse, summary="流式获取Agent结果")
async def stream_agent_result(task_id: str):
    """
    流式接口：获取指定任务的实时结果（仅支持stream=True启动的任务）
    - 采用SSE（Server-Sent Events）协议，前端可实时接收片段
    - 结束标识为[data: [DONE]]，同时更新历史对话
    """
    # 1. 查找任务在历史中的索引
    task_index = next(
        (i for i, item in enumerate(CONVERSATION_HISTORY)
         if item["task_id"] == task_id),
        None
    )
    if task_index is None:
        raise HTTPException(status_code=404, detail=f"流式任务 {task_id} 不存在")

    task_info = CONVERSATION_HISTORY[task_index]
    # 2. 验证Agent是否仍存在
    target_agent = AGENTS_REGISTRY.get(task_info["agent_name"])
    if not target_agent:
        raise HTTPException(
            status_code=400, detail=f"任务关联的Agent {task_info['agent_name']} 已删除")
    context = {"session_id": task_id, "config": {"mcp_hub_switch": False}}
    # 3. 生成流式响应（SSE格式）

    async def stream_generator() -> AsyncGenerator[str, None]:
        full_response = ""
        try:
            stream = MyRunner.run_streamed(
                target_agent, task_info["user_input"], context=context)
            async for event in stream.stream_events():
                try:
                    full_response += event.content
                except:
                    pass
                yield f"data: {event}\n\n"
            yield f"data: [DONE]\n\n"

            CONVERSATION_HISTORY[task_index].update({
                "agent_response": full_response,
                "completed": True
            })
        except Exception as e:
            # 流式异常：返回错误信息，更新历史为失败
            error_msg = f"流式处理失败：{str(e)}"
            yield f"data: {error_msg}\n\n"
            yield "data: [DONE]\n\n"
            CONVERSATION_HISTORY[task_index].update({
                "agent_response": error_msg,
                "completed": True
            })

    # 返回流式响应（媒体类型为text/event-stream）
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/history", response_model=List[HistoryItem], summary="获取历史对话")
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

# 6. 服务启动入口
if __name__ == "__main__":
    import uvicorn
    # 启动前打印提示信息
    hooks_logger.info("=" * 50)
    hooks_logger.info("✅ 简化版Agent API 启动准备完成")
    hooks_logger.info(f"📌 可用Agent：{list(AGENTS_REGISTRY.keys())}")
    hooks_logger.info(f"📚 API文档：http://localhost:8000/docs")
    hooks_logger.info(f"⚠️  注意：重启服务后，内存中的历史对话会丢失！")
    hooks_logger.info("=" * 50)

    # 启动FastAPI服务（默认8000端口，开发模式自动重载）
    uvicorn.run(
        app="agent_infer.server.app:app",
        host="0.0.0.0",  # 允许外部设备访问
        port=8289,
        reload=True  # 开发模式：修改代码自动重启服务
    )
