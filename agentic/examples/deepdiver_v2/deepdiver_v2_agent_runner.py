from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, cast, List, Optional

import requests
from typing_extensions import Unpack
from agents.exceptions import (
    AgentsException,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OutputGuardrailTripwireTriggered,
    RunErrorDetails,
    UserError,
)
from agents._run_impl import (
    NextStepFinalOutput,
    NextStepHandoff,
    NextStepRunAgain,
    TraceCtxManager,
)
from agents.tool import (
    FunctionTool,
    FunctionToolResult,
)
from agents.exceptions import (
    AgentsException,
    MaxTurnsExceeded,
    RunErrorDetails,
    UserError
)
from agents.guardrail import (
    InputGuardrailResult,
)
from agents.items import (
    ItemHelpers,
)
from agents.tool_context import ToolContext
from agents.models.interface import Model

from examples.deepdiver_v2.deepdiver_v2_chat_converter import Ddv2ChatConverter
from agents.lifecycle import RunHooks
from agents.logger import logger
from agents.memory import Session, SessionInputCallback
from agents.result import RunResult, RunResultStreaming
from agents.run_context import RunContextWrapper, TContext
from agents.tracing import Span, SpanError, agent_span, get_current_trace, trace, add_trace_processor, function_span, \
    response_span
from agents.tracing.span_data import AgentSpanData, CustomSpanData
from agents.util import _coro, _error_tracing
from agents.run import AgentRunner, RunOptions, RunConfig, _copy_str_or_list, Runner
import config.config as config
from agents.items import MessageOutputItem
from openai.types.responses.response_output_message import ResponseOutputMessage, ResponseOutputText
import uuid
from agents._run_impl import (
    get_model_tracing_impl,
    RunImpl,
    ToolRunFunction
)
from openai.types.responses.response_prompt_param import (
    ResponsePromptParam,
)
from agents import _debug, custom_span
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import TResponseInputItem
from agents.logger import logger
from agents.tool import Tool
from agents.tracing.span_data import GenerationSpanData
from agents.tracing.spans import Span
from agents.util._json import _to_dump_compatible
from agents.models.chatcmpl_helpers import ChatCmplHelpers
from agents.models.fake_id import FAKE_RESPONSES_ID
from agents.models.interface import ModelTracing
from agents.models.openai_responses import Converter as OpenAIResponsesConverter
from agents import Agent, OpenAIChatCompletionsModel
from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream
from openai.types.responses import Response, ResponseFunctionToolCall, EasyInputMessage
from agents.items import RunItemBase
from features.early_stop import EarlyStopManager

es_manager = EarlyStopManager()

import time
from typing import TYPE_CHECKING, Any, Literal, overload

from openai.types.responses import (
                ResponseCompletedEvent,
                ResponseOutputItemDoneEvent,
            )
from agents.run import _TOOL_CALL_TYPES
from agents.items import (
    ItemHelpers,
    ToolCallItem,
    ToolCallItemTypes,
)
from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)

# --- 新压缩流程 ---
from features.compression.config import CompressionConfig
from features.compression.manager import CompressionManager
from features.compression.adapters.ddv2 import Ddv2ConversationAdapter


from examples.deepdiver_v2.tracing.processor import SSEBatchTracingProcessor

from examples.deepdiver_v2.tracing.create import streaming_generation_span

from features.model_routing import ModelRouter, ModelRoutingConfig, ProgressCheckStrategy

if TYPE_CHECKING:
    from agents.model_settings import ModelSettings
from agents.tracing import generation_span
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
import json
from agents.usage import Usage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from agent_infer.agent_runner.runner import MyAgentRunner
from agents.stream_events import StreamEvent
from agents._run_impl import AgentToolUseTracker,QueueCompleteSentinel,SingleStepResult
from agents.items import (
    ModelResponse,
    RunItem,
    HandoffCallItem,
    TResponseInputItem,
    TResponseOutputItem,
    ToolCallOutputItem
)
DEFAULT_MAX_TURNS = 15
from examples.deepdiver_v2.extra.server_config import mcp_hub_v2

class Ddv2AgentRunner(MyAgentRunner):

    @staticmethod
    def _find_latest_function_call(generated_items: list[RunItem]) -> Any | None:
        for item in reversed(generated_items):
            raw_item = getattr(item, "raw_item", None)
            if hasattr(raw_item, "name") and hasattr(raw_item, "arguments"):
                return raw_item
        return None

    @classmethod
    async def run(
        self,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        **kwargs: Unpack[RunOptions[TContext]],
    ) -> RunResult:
        context = kwargs.get("context")
        max_turns = kwargs.get("max_turns", DEFAULT_MAX_TURNS)
        hooks = kwargs.get("hooks")
        run_config = kwargs.get("run_config")
        previous_response_id = kwargs.get("previous_response_id")
        conversation_id = kwargs.get("conversation_id")
        session = kwargs.get("session")


        if hooks is None:
            hooks = RunHooks[Any]()
        if run_config is None:
            run_config = RunConfig()

        # Keep original user input separate from session-prepared input
        original_user_input = input
        prepared_input = await self._prepare_input_with_session(
            input, session, run_config.session_input_callback
        )

        tool_use_tracker = AgentToolUseTracker()
        # original_input: 用户query； generated_items:除了prompt以外的上下文； context_wrapper中有记录总token数（包括输入+输出）
        with TraceCtxManager(
            workflow_name=run_config.workflow_name,
            trace_id=run_config.trace_id,
            group_id=run_config.group_id,
            metadata=run_config.trace_metadata,
            disabled=run_config.tracing_disabled,
        ):
            current_turn = 0
            original_input: str | list[TResponseInputItem] = _copy_str_or_list(prepared_input)
            generated_items: list[RunItem] = []
            model_responses: list[ModelResponse] = []

            context_wrapper: RunContextWrapper[TContext] = RunContextWrapper(
                context=context,  # type: ignore
            )

            # 初始化 ModelRouter
            routing_config = ModelRoutingConfig.from_config_module(
                context.get("config", {}), config
            )
            context_wrapper.context["model_router"] = ModelRouter(
                config=routing_config,
                strategy=ProgressCheckStrategy(config=routing_config, trigger_tool_name="think"),
            )

            # 初始化早停状态
            agent_id = context_wrapper.context.get("agent_id", "-1")
            es_manager.init_agent(agent_id)

            input_guardrail_results: list[InputGuardrailResult] = []

            current_span: Span[AgentSpanData] | None = None
            current_agent = starting_agent
            should_run_agent_start_hooks = True

            # save only the new user input to the session, not the combined history
            await self._save_result_to_session(session, original_user_input, [])

            try:
                while True:
                    span_e2e = custom_span(
                        name="e2e_call_metric",
                        data={
                            "turn_submitted": current_turn
                        }
                    )
                    e2e_start = time.perf_counter()
                    span_e2e.start()
                    config_overrides = context_wrapper.context.get("config", {})
                    mcp_hub_switch = config_overrides.get("mcp_hub_switch", getattr(config, 'mcp_hub_switch', False))
                    if mcp_hub_switch:
                        # load balance
                        with mcp_hub_v2.mcp_call_lock:
                            choose_server_id = mcp_hub_v2.mcp_server_manager.select_server()  # 会有调用次数加一的操作
                        current_agent.mcp_servers = [mcp_hub_v2.stateless_server[current_agent.name].servername_to_server[choose_server_id]]
                        #tool set cache
                        if current_agent.mcp_servers[0] not in mcp_hub_v2.stateless_server[current_agent.name].tool_set_cache:
                            all_tools = await AgentRunner._get_all_tools(current_agent, context_wrapper)
                            mcp_hub_v2.stateless_server[current_agent.name].tool_set_cache[current_agent.mcp_servers[0]] = all_tools
                        else:
                            all_tools = mcp_hub_v2.stateless_server[current_agent.name].tool_set_cache[current_agent.mcp_servers[0]]
                    else:
                        all_tools = await AgentRunner._get_all_tools(current_agent, context_wrapper)
                        # Skip mcp_hub logic when mcp_hub_switch is False
                        # with mcp_hub_v2.mcp_call_lock:
                        #     current_server_id = mcp_hub_v2.stateless_server[current_agent.name].server_to_servername[current_agent.mcp_servers[0]]
                        #     mcp_hub_v2.mcp_server_manager.servers[current_server_id] += 1

                    # Start an agent span if we don't have one. This span is ended if the current
                    # agent changes, or if the agent loop ends.
                    if current_span is None:
                        handoff_names = [
                            h.agent_name
                            for h in await AgentRunner._get_handoffs(current_agent, context_wrapper)
                        ]
                        if output_schema := AgentRunner._get_output_schema(current_agent):
                            output_type_name = output_schema.name()
                        else:
                            output_type_name = "str"

                        current_span = agent_span(
                            name=current_agent.name,
                            handoffs=handoff_names,
                            output_type=output_type_name,
                        )
                        current_span.start(mark_as_current=True)
                        current_span.span_data.tools = [t.name for t in all_tools]

                    current_turn += 1

                    current_tokens = context_wrapper.usage.total_tokens
                    if current_tokens > config.TOKEN_THRESHOLD:
                        logger.warning(
                            f"Token count {current_tokens} exceeds threshold {config.TOKEN_THRESHOLD}. "
                        )


                    if current_turn == max_turns-3 :
                        followup_prompt = {
                            "PlannerAgent" : "Due to length and number of rounds restrictions, you must now call the `planner_subjective_task_done` or `planner_objective_task_done` tool to report the completion of your task. /no_think",
                            "SubjectiveInformationSeekerAgent" : "Due to length and number of rounds restrictions, you must now call the `info_seeker_subjective_task_done` tool to report the completion of your task. /no_think",
                            "ObjectiveInformationSeekerAgent" : "Due to length and number of rounds restrictions, you must now call the `info_seeker_objective_task_done` tool to report the completion of your task. /no_think",
                            "WriterAgent" : "Due to length and number of rounds restrictions, you must now call the `writer_subjective_task_done` tool to report the completion of your task. /no_think"
                        }
                        raw_message = EasyInputMessage(
                            content=followup_prompt[current_agent.name],
                            role="user",
                            type="message"
                        )
                        forced_completion_item = RunItemBase(
                            agent=current_agent,
                            raw_item=raw_message
                        )
                        generated_items.append(forced_completion_item)
                    ###################
                    if current_turn > max_turns:
                        _error_tracing.attach_error_to_span(
                            current_span,
                            SpanError(
                                message="Max turns exceeded",
                                data={"max_turns": max_turns},
                            ),
                        )
                        raise MaxTurnsExceeded(f"Max turns ({max_turns}) exceeded")

                    logger.debug(
                        f"Running agent {current_agent.name} (turn {current_turn})",
                    )

                    if current_turn == 1:
                        input_guardrail_results, turn_result = await asyncio.gather(
                            self._run_input_guardrails(
                                starting_agent,
                                starting_agent.input_guardrails
                                + (run_config.input_guardrails or []),
                                _copy_str_or_list(prepared_input),
                                context_wrapper,
                            ),

                            self._run_single_turn(
                                agent=current_agent,
                                all_tools=all_tools,
                                original_input=original_input,
                                generated_items=generated_items,
                                hooks=hooks,
                                context_wrapper=context_wrapper,
                                run_config=run_config,
                                should_run_agent_start_hooks=should_run_agent_start_hooks,
                                tool_use_tracker=tool_use_tracker,
                                previous_response_id=previous_response_id,
                                conversation_id=conversation_id,
                                current_turn=current_turn,
                                max_turns=max_turns,
                            ),
                        )
                    else:
                        turn_result = await self._run_single_turn(
                            agent=current_agent,
                            all_tools=all_tools,
                            original_input=original_input,
                            generated_items=generated_items,
                            hooks=hooks,
                            context_wrapper=context_wrapper,
                            run_config=run_config,
                            should_run_agent_start_hooks=should_run_agent_start_hooks,
                            tool_use_tracker=tool_use_tracker,
                            previous_response_id=previous_response_id,
                            conversation_id=conversation_id,
                            current_turn=current_turn,
                            max_turns = max_turns,
                        )
                    should_run_agent_start_hooks = False

                    model_responses.append(turn_result.model_response)
                    original_input = turn_result.original_input
                    generated_items = turn_result.generated_items

                    if isinstance(turn_result.next_step, NextStepFinalOutput):
                        output_guardrail_results = await self._run_output_guardrails(
                            current_agent.output_guardrails + (run_config.output_guardrails or []),
                            current_agent,
                            turn_result.next_step.output,
                            context_wrapper,
                        )
                        result = RunResult(
                            input=original_input,
                            new_items=generated_items,
                            raw_responses=model_responses,
                            final_output=turn_result.next_step.output,
                            _last_agent=current_agent,
                            input_guardrail_results=input_guardrail_results,
                            output_guardrail_results=output_guardrail_results,
                            context_wrapper=context_wrapper,
                        )
                        await self._save_result_to_session(session, [], turn_result.new_step_items)

                        return result
                    elif isinstance(turn_result.next_step, NextStepHandoff):
                        current_agent = cast(Agent[TContext], turn_result.next_step.new_agent)
                        current_span.finish(reset_current=True)
                        current_span = None
                        should_run_agent_start_hooks = True
                    elif isinstance(turn_result.next_step, NextStepRunAgain):
                        await self._save_result_to_session(session, [], turn_result.new_step_items)
                    else:
                        raise AgentsException(
                            f"Unknown next step type: {type(turn_result.next_step)}"
                        )
                    e2e_end = time.perf_counter()
                    span_e2e.span_data.data['metrics'] = e2e_end - e2e_start;
                    span_e2e.finish()
            except AgentsException as exc:
                exc.run_data = RunErrorDetails(
                    input=original_input,
                    new_items=generated_items,
                    raw_responses=model_responses,
                    last_agent=current_agent,
                    context_wrapper=context_wrapper,
                    input_guardrail_results=input_guardrail_results,
                    output_guardrail_results=[],
                )
                raise exc
            finally:
                if current_span:
                    current_span.finish(reset_current=True)
                # 清理早停状态
                es_manager.remove_agent(agent_id)


    @classmethod
    async def _run_single_turn(
        cls,
        *,
        agent: Agent[TContext],
        all_tools: list[Tool],
        original_input: str | list[TResponseInputItem],
        generated_items: list[RunItem],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        should_run_agent_start_hooks: bool,
        tool_use_tracker: AgentToolUseTracker,
        previous_response_id: str | None,
        conversation_id: str | None,
        current_turn:int | None = None,
        max_turns:int | None = None,
     ) -> SingleStepResult:
        # try to replace compressed search results:
        # 1. get finished compression task_id
        # 2. check whether this id is before/smaller than `cur_crawler_id`
        # 3. if yes, get the compressed results and replace the raw results in the `conversation_history`
        # logger.info(f"### before compression")
        # logger.info(f"### {self.config.temperature=}")

        agent_id = context_wrapper.context.get("agent_id", '-1')

        # --- 获取/创建当前 Agent 的 CompressionManager ---
        # 每个 agent_id 需要独立的 CompressionManager 实例，
        # 避免多个 seeker 共享同一个实例导致压缩任务混淆
        parent_compression_config = context_wrapper.context.get("compression_config", None)
        agents_state = context_wrapper.context.setdefault("agents_state", {})
        this_agent_state = agents_state.setdefault(agent_id, {
            "compress_id_multi_round_dict": {},
            "iteration_multi_round_compress_submit": 0,
            "active_compress_spans": {},
            "already_compressed_num": 0
        })
        pending_early_stop = this_agent_state.pop("pending_early_stop", None)
        if pending_early_stop is not None:
            pending_item = pending_early_stop["item"]
            if any(item is pending_item for item in generated_items):
                print(
                    "[early_stop] prompt observed in next turn: "
                    f"reason={pending_early_stop['reason']}, "
                    f"injected_iter={pending_early_stop['iter']}, "
                    f"current_iter={current_turn}"
                )
        compression_manager = this_agent_state.get("compression_manager", None)
        if parent_compression_config is not None and compression_manager is None:
            compression_manager = CompressionManager(
                config=parent_compression_config,
                adapter=None,  # adapter 延迟初始化（需要 agent, run_config 等）
            )
            this_agent_state["compression_manager"] = compression_manager
        # 延迟初始化 adapter（需要 agent, run_config, context_wrapper, system_prompt）
        if compression_manager is not None and compression_manager._adapter is None:
            # system_prompt 在后面才生成，这里先占位，在 on_turn_end 前更新
            compression_manager.set_adapter(Ddv2ConversationAdapter(
                chat_converter=Ddv2ChatConverter,
                agent=agent,
                run_config=run_config,
                context_wrapper=context_wrapper,
                system_prompt="",  # 占位，后面 on_turn_end 时 system_prompt 已生成
                think_tool_name="think",
            ))

        # early_stop 仍然需要的状态

        # Ensure we run the hooks before anything else
        if should_run_agent_start_hooks:
            await asyncio.gather(
                hooks.on_agent_start(context_wrapper, agent),
                (
                    agent.hooks.on_start(context_wrapper, agent)
                    if agent.hooks
                    else _coro.noop_coroutine()
                ),
            )

        system_prompt, prompt_config = await asyncio.gather(
            agent.get_system_prompt(context_wrapper),
            agent.get_prompt(context_wrapper),
        )
        if "<tools></tools>" not in system_prompt.lower():
            tool_str = str([
                Ddv2ChatConverter.function_tool_to_openai_tool(t)
                for t in all_tools
            ])
            tool_choice_prompt = f"""Below, within the <tools></tools> tags, are the descriptions of each tool and the required fields for invocation:
<tools>
{tool_str}
</tools>
For each function call, return a JSON object placed within the [unused11][unused12] tags, which includes the function name and the corresponding function arguments:
[unused11][{{\"name\": <function name>, \"arguments\": <args json object>}}][unused12] \n"""
            *head_lines, last_line = system_prompt.rstrip().rsplit('\n', 1)
            system_prompt = ''.join([*head_lines, tool_choice_prompt, last_line])


        # multi-round compression — 消费已完成的压缩结果
        # --- on_turn_start 消费压缩结果 ---
        if compression_manager is not None:
            generated_items = compression_manager.on_turn_start(current_turn, generated_items)

        output_schema = cls._get_output_schema(agent)
        handoffs = await cls._get_handoffs(agent, context_wrapper)
        input = ItemHelpers.input_to_new_input_list(original_input)
        input.extend([generated_item.to_input_item() for generated_item in generated_items])

        new_response = await cls._get_new_response(
            agent,
            system_prompt,
            input,
            output_schema,
            [],  # 这里的all_tools字段用来在调用chat/completion时作为工具参数使用，pangu7B未开启工具调用功能，置空
            handoffs,
            hooks,
            context_wrapper,
            run_config,
            tool_use_tracker,
            previous_response_id,
            conversation_id,
            prompt_config,
        )

        tool_result = await cls._get_single_step_result_from_response(
            agent=agent,
            original_input=original_input,
            pre_step_items=generated_items,
            new_response=new_response,
            output_schema=output_schema,
            all_tools=all_tools, # 这里的all_tools字段用来解析响应，不能置空
            handoffs=handoffs,
            hooks=hooks,
            context_wrapper=context_wrapper,
            run_config=run_config,
            tool_use_tracker=tool_use_tracker,
        )

        # Skip mcp_hub release when mcp_hub_switch is False
        config_overrides = context_wrapper.context.get("config", {})
        mcp_hub_switch = config_overrides.get("mcp_hub_switch", getattr(config, 'mcp_hub_switch', False))
        if mcp_hub_switch:
            current_call_server_name = mcp_hub_v2.stateless_server[agent.name].server_to_servername[agent.mcp_servers[0]]
            with mcp_hub_v2.mcp_call_lock:
                mcp_hub_v2.mcp_server_manager.release_server(current_call_server_name)

        function_call = cls._find_latest_function_call(tool_result.new_step_items)
        if function_call is None:
            return tool_result

        try:
            tool_name = function_call.name
            token_num_before_compress = new_response.usage.total_tokens
            if token_num_before_compress == 0:
                # fallback: estimate by conversation item count (1 item ≈ 200 tokens roughly)
                token_num_before_compress = len(tool_result.generated_items) * 200
                print(f"[async_compress] usage total_tokens=0, fallback estimate: {token_num_before_compress} (items={len(tool_result.generated_items)})")
            else:
                print(f"[async_compress] usage debug: {new_response.usage}")

            config_overrides = context_wrapper.context.get("config", {})
            es_manager.refresh_config(config_overrides)
            current_thought = function_call.arguments
            es_result = es_manager.check(
                agent_id=agent_id,
                current_turn=current_turn,
                tool_name=tool_name,
                current_thought=current_thought,
            )
            if es_result.should_inject:
                injected_item = RunItemBase(agent=agent, raw_item=es_result.injected_message)
                tool_result.new_step_items.append(injected_item)
                this_agent_state["pending_early_stop"] = {
                    "item": injected_item,
                    "reason": es_result.reason,
                    "iter": current_turn,
                }
                print(
                    f"[early_stop] prompt injected: reason={es_result.reason}, "
                    f"iter={current_turn}, target=new_step_items"
                )
            config_overrides = context_wrapper.context.get("config", {})
            # --- on_turn_end 提交压缩任务 ---
            if compression_manager is not None:
                # 更新 adapter 的 system_prompt（此时 system_prompt 已生成）
                if compression_manager._adapter is not None:
                    compression_manager._adapter._system_prompt = system_prompt
                token_count = token_num_before_compress if token_num_before_compress else 0
                compression_manager.on_turn_end(
                    current_turn=current_turn,
                    generated_items=tool_result.generated_items,
                    last_tool_name=tool_name,
                    token_count=token_count,
                )

            # --- 持久化 early_stop 相关状态 ---
            this_agent_state.update({
            })
        except Exception as e:
            raise e
        return tool_result

    @classmethod
    async def _get_single_step_result_from_response(
            cls,
            *,
            agent: Agent[TContext],
            all_tools: list[Tool],
            original_input: str | list[TResponseInputItem],
            pre_step_items: list[RunItem],
            new_response: ModelResponse,
            output_schema: AgentOutputSchemaBase | None,
            handoffs: list[Handoff],
            hooks: RunHooks[TContext],
            context_wrapper: RunContextWrapper[TContext],
            run_config: RunConfig,
            tool_use_tracker: AgentToolUseTracker,
            event_queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel] | None = None,
    ) -> SingleStepResult:
        processed_response = Ddv2RunImpl.process_model_response(
            agent=agent,
            all_tools=all_tools,
            response=new_response,
            output_schema=output_schema,
            handoffs=handoffs,
        )

        tool_use_tracker.add_tool_use(agent, processed_response.tools_used)

        # Send handoff items immediately for streaming, but avoid duplicates
        if event_queue is not None and processed_response.new_items:
            handoff_items = [
                item for item in processed_response.new_items if isinstance(item, HandoffCallItem)
            ]
            if handoff_items:
                Ddv2RunImpl.stream_step_items_to_queue(cast(list[RunItem], handoff_items), event_queue)

        return await Ddv2RunImpl.execute_tools_and_side_effects(
            agent=agent,
            original_input=original_input,
            pre_step_items=pre_step_items,
            new_response=new_response,
            processed_response=processed_response,
            output_schema=output_schema,
            hooks=hooks,
            context_wrapper=context_wrapper,
            run_config=run_config,
        )

    @classmethod
    async def _get_new_response(
            cls,
            agent: Agent[TContext],
            system_prompt: str | None,
            input: list[TResponseInputItem],
            output_schema: AgentOutputSchemaBase | None,
            all_tools: list[Tool],
            handoffs: list[Handoff],
            hooks: RunHooks[TContext],
            context_wrapper: RunContextWrapper[TContext],
            run_config: RunConfig,
            tool_use_tracker: AgentToolUseTracker,
            previous_response_id: str | None,
            conversation_id: str | None,
            prompt_config: ResponsePromptParam | None,
    ) -> ModelResponse:

        # Allow user to modify model input right before the call, if configured
        filtered = await cls._maybe_filter_model_input(
            agent=agent,
            run_config=run_config,
            context_wrapper=context_wrapper,
            input_items=input,
            system_instructions=system_prompt,
        )


        model_settings = agent.model_settings.resolve(run_config.model_settings)
        # delete configuration "tool_choice="required" to adapt pangu7B
        model_settings.tool_choice = None
        model_settings = Ddv2RunImpl.maybe_reset_tool_choice(agent, tool_use_tracker, model_settings)

        # If we have run hooks, or if the agent has hooks, we need to call them before the LLM call
        await asyncio.gather(
            hooks.on_llm_start(context_wrapper, agent, filtered.instructions, filtered.input),
            (
                agent.hooks.on_llm_start(
                    context_wrapper,
                    agent,
                    filtered.instructions,  # Use filtered instructions
                    filtered.input,  # Use filtered input
                )
                if agent.hooks
                else _coro.noop_coroutine()
            ),
        )
        model = cls._get_model(agent, run_config)
        # 路由决策（升降级统一在此，_get_model 之后、get_response 之前）
        router = context_wrapper.context.get("model_router")
        if router and router.config.enabled:
            converted_messages = Ddv2ChatConverter.items_to_messages(filtered.input)
            if system_prompt:
                converted_messages.insert(0, {"role": "system", "content": system_prompt})
            route_id = str(context_wrapper.context.get("agent_id", "unknown"))
            model = await router.resolve_model(
                agent.name,
                converted_messages,
                model,
                model_settings,
                route_id=route_id,
            )

        raw_response = await cls.get_response(
            model = model,
            system_instructions=filtered.instructions,
            input=filtered.input,
            model_settings=model_settings,
            tools=all_tools,
            output_schema=output_schema,
            handoffs=handoffs,
            tracing=get_model_tracing_impl(
                run_config.tracing_disabled, run_config.trace_include_sensitive_data
            ),
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt_config,
            context_wrapper=context_wrapper,
            agent=agent
        )

        def convert_response(raw_response: ModelResponse) -> ModelResponse:
            import json
            from openai.types.responses import  ResponseFunctionToolCall
            content = raw_response.output[0].content[0].text
            if content:
                # 废弃functioncall的方式，改为自行从提示词中提取工具后解析调用
                def extract_tool_calls(content) -> list:
                    import re
                    from json_repair import repair_json
                    from openai.types.chat import ChatCompletionMessageToolCall
                    from openai.types.chat.chat_completion_message_tool_call import Function

                    if not content:
                        return []
                    matches = re.findall(r'\[unused\d{2}\]\s*([\s\S]*?)\s*\[unused\d{2}\]', content)
                    if not matches:
                        return []
                    json_candidate = matches[-1].strip()
                    try:
                        raw_calls = repair_json(json_candidate, return_objects=True)
                        if isinstance(raw_calls, dict):
                            raw_calls = [raw_calls]
                        elif not isinstance(raw_calls, list):
                            return []
                    except Exception as e:
                        print(f"repair_json 失败: {e}")
                        return []
                    tool_calls = []
                    for i, item in enumerate(raw_calls):
                        if not isinstance(item, dict) or "name" not in item:
                            continue
                        args = item.get("arguments", {})
                        if isinstance(args, (dict, list)):
                            args_str = json.dumps(args, ensure_ascii=False)
                        else:
                            args_str = str(args)
                        tool_call = ChatCompletionMessageToolCall(
                            type="function",
                            id=f"call_{i}_{abs(hash(args_str)) % 1000000}",
                            function=Function(
                                name=item["name"],
                                arguments=args_str
                            )
                        )
                        tool_calls.append(tool_call)
                    return tool_calls

                tool_calls_list = extract_tool_calls(content)
                items: list[TResponseOutputItem] = []

                if not tool_calls_list:
                    return raw_response
                for tool_call in tool_calls_list:
                    if tool_call.type == "function":
                        items.append(
                            ResponseFunctionToolCall(
                                id=raw_response.output[0].id,
                                call_id=tool_call.id,
                                arguments=tool_call.function.arguments,
                                name=tool_call.function.name,
                                type="function_call",
                            )
                        )
                    elif tool_call.type == "custom":
                        pass
                return ModelResponse(
                    output=items,
                    usage=raw_response.usage,
                    response_id=None,
                )
        try:
            new_response = convert_response(raw_response)
            context_wrapper.usage.add(new_response.usage)
        except Exception as e:
            raise e

        # If we have run hooks, or if the agent has hooks, we need to call them after the LLM call
        await asyncio.gather(
            (
                agent.hooks.on_llm_end(context_wrapper, agent, new_response)
                if agent.hooks
                else _coro.noop_coroutine()
            ),
            hooks.on_llm_end(context_wrapper, agent, new_response),
        )

        return new_response

    @classmethod
    async def get_response(
        cls,
        model: Model,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,  # unused
        conversation_id: str | None = None,  # unused
        prompt: ResponsePromptParam | None = None,
        context_wrapper: RunContextWrapper[TContext] | None = None,
        agent: Agent[TContext] | None = None
    ) -> ModelResponse:

        router = None
        if context_wrapper is not None and isinstance(context_wrapper.context, dict):
            router = context_wrapper.context.get("model_router")

        with generation_span(
            model=str(model),
            model_config=model_settings.to_json_dict() | {"base_url": str(model._client.base_url)} | {"current_model": router.current_model if router else "small_model"},
            disabled=tracing.is_disabled(),
            input = input[0].get('content') if len(input)==1 else ( input if isinstance(input, str) else input[-1].get("output") if isinstance(input, list) and input else '')
        ) as span_generation:
            response = await cls._fetch_response(
                model,
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=False,
                prompt=prompt,
                context_wrapper=context_wrapper,
                agent = agent
            )

            message: ChatCompletionMessage | None = None
            first_choice: Choice | None = None
            if response.choices and len(response.choices) > 0:
                first_choice = response.choices[0]
                message = first_choice.message

            if _debug.DONT_LOG_MODEL_DATA:
                logger.debug("Received model response")
            else:
                if message is not None:
                    logger.debug(
                        "LLM resp:\n%s\n",
                        json.dumps(message.model_dump(), indent=2, ensure_ascii=False),
                    )
                else:
                    finish_reason = first_choice.finish_reason if first_choice else "-"
                    logger.debug(f"LLM resp had no message. finish_reason: {finish_reason}")

            usage = (
                Usage(
                    requests=1,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    input_tokens_details=InputTokensDetails(
                        cached_tokens=getattr(
                            response.usage.prompt_tokens_details, "cached_tokens", 0
                        )
                        or 0,
                    ),
                    output_tokens_details=OutputTokensDetails(
                        reasoning_tokens=getattr(
                            response.usage.completion_tokens_details, "reasoning_tokens", 0
                        )
                        or 0,
                    ),
                )
                if response.usage
                else Usage()
            )
            if tracing.include_data():
                span_generation.span_data.output = (
                    [message.model_dump()] if message is not None else []
                )
            span_generation.span_data.usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }

            items = Ddv2ChatConverter.message_to_output_items(message) if message is not None else []
            return ModelResponse(
                output=items,
                usage=usage,
                response_id=None,
            )

    @classmethod
    async def _fetch_response(
        cls,
        model: Model,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: bool = False,
        prompt: ResponsePromptParam | None = None,
        context_wrapper: RunContextWrapper[TContext] | None = None,
        agent: Agent[TContext] | None = None
    ) -> ChatCompletion | tuple[Response, AsyncStream[ChatCompletionChunk]]:

        converted_messages = []
        for item in input:
            msgs = Ddv2ChatConverter.items_to_messages([item])
            converted_messages.extend(msgs)

        if system_instructions:
            converted_messages.insert(
                0,
                {
                    "content": system_instructions,
                    "role": "system",
                },
            )
        converted_messages = _to_dump_compatible(converted_messages)

        if tracing.include_data():
            span.span_data.input = converted_messages

        parallel_tool_calls = (
            True
            if model_settings.parallel_tool_calls and tools and len(tools) > 0
            else False
            if model_settings.parallel_tool_calls is False
            else NOT_GIVEN
        )
        tool_choice = Ddv2ChatConverter.convert_tool_choice(model_settings.tool_choice)
        response_format = Ddv2ChatConverter.convert_response_format(output_schema)

        converted_tools = [Ddv2ChatConverter.tool_to_openai(tool) for tool in tools] if tools else []

        for handoff in handoffs:
            converted_tools.append(Ddv2ChatConverter.convert_handoff_tool(handoff))

        converted_tools = _to_dump_compatible(converted_tools)

        if _debug.DONT_LOG_MODEL_DATA:
            logger.debug("Calling LLM")
        else:
            messages_json = json.dumps(
                converted_messages,
                indent=2,
                ensure_ascii=False,
            )
            tools_json = json.dumps(
                converted_tools,
                indent=2,
                ensure_ascii=False,
            )
            logger.debug(
                f"{messages_json}\n"
                f"Tools:\n{tools_json}\n"
                f"Stream: {stream}\n"
                f"Tool choice: {tool_choice}\n"
                f"Response format: {response_format}\n"
            )

        reasoning_effort = model_settings.reasoning.effort if model_settings.reasoning else None
        store = ChatCmplHelpers.get_store_param(model._get_client(), model_settings)

        stream_options = ChatCmplHelpers.get_stream_options_param(
            model._get_client(), model_settings, stream=True
        )
        # Force include_usage so that the last stream chunk contains token usage
        if stream_options is None:
            stream_options = {"include_usage": True}
        elif isinstance(stream_options, dict):
            stream_options["include_usage"] = True

        ret = await cls.call_and_measure(
            model_settings,
            converted_messages,
            model,
            converted_tools,
            tool_choice,
            response_format,
            parallel_tool_calls,
            stream_options,
            store,
            reasoning_effort,
            tracing
        )
        if isinstance(ret, ChatCompletion):
            return ret
        responses_tool_choice = OpenAIResponsesConverter.convert_tool_choice(
            model_settings.tool_choice
        )
        if responses_tool_choice is None or responses_tool_choice == NOT_GIVEN:
            responses_tool_choice = "auto"

        response = Response(
            id=FAKE_RESPONSES_ID,
            created_at=time.time(),
            model=model,
            object="response",
            output=[],
            tool_choice=responses_tool_choice,  # type: ignore[arg-type]
            top_p=model_settings.top_p,
            temperature=model_settings.temperature,
            tools=[],
            parallel_tool_calls=parallel_tool_calls or False,
            reasoning=model_settings.reasoning,
        )
        return response, ret

    @classmethod
    async def call_and_measure(
            self,
            model_settings: ModelSettings,
            converted_messages,
            model,
            converted_tools,
            tool_choice,
            response_format,
            parallel_tool_calls,
            stream_options,
            store,
            reasoning_effort,
            tracing: ModelTracing,
    ) -> tuple:
        start_time = time.perf_counter()
        def _non_null_or_not_given(value: Any) -> Any:
            return value if value is not None else NOT_GIVEN
            # if 大小模型切换开启，则新增runConfig配置，新增model_setting配置
        try:
            ret = await model._get_client().chat.completions.create(
                model=model.model,
                messages=converted_messages,
                tools=converted_tools or NOT_GIVEN,
                temperature=_non_null_or_not_given(model_settings.temperature),
                top_p=_non_null_or_not_given(model_settings.top_p),
                frequency_penalty=_non_null_or_not_given(model_settings.frequency_penalty),
                presence_penalty=_non_null_or_not_given(model_settings.presence_penalty),
                max_tokens=_non_null_or_not_given(model_settings.max_tokens),
                tool_choice=tool_choice,
                response_format=response_format,
                parallel_tool_calls=parallel_tool_calls,
                stream=True,
                stream_options=_non_null_or_not_given(stream_options),
                store=_non_null_or_not_given(store),
                reasoning_effort=_non_null_or_not_given(reasoning_effort),
                verbosity=_non_null_or_not_given(model_settings.verbosity),
                top_logprobs=_non_null_or_not_given(model_settings.top_logprobs),
                extra_headers=model._merge_headers(model_settings),
                extra_query=model_settings.extra_query,
                extra_body=model_settings.extra_body,
                metadata=_non_null_or_not_given(model_settings.metadata),
                **(model_settings.extra_args or {}),
            )
        except Exception as e:
            print(f"[call_and_measure] API call failed: {e}")
            raise  # 必须重新抛出异常，不能静默失败
        chunks: List[ChatCompletionChunk] = []
        token_times: List[float] = []  # 每个 token 到达的时间（相对 start_time）
        first_token_time = None
        with streaming_generation_span() as span:
            # 2. 遍历流，记录时间和 chunk
            async for chunk in ret:
                now = time.perf_counter()
                token = chunk.choices[0].delta.content if chunk.choices and hasattr(chunk.choices[0].delta, 'content') else None
                if token:
                    # 实时推送 Token 到前端 ---
                    span.push_chunk(chunk_type="token", delta=token)

                chunks.append(chunk)
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)

        # 3. 还原完整响应（非流式）
        def reconstruct_chat_completion(chunks: List["ChatCompletionChunk"]) -> ChatCompletion:
            if not chunks:
                raise ValueError("No chunks provided")

            # 聚合内容
            full_content = ""
            tool_calls = []
            role = "assistant"
            finish_reason: Optional[str] = None
            usage = None

            for chunk in chunks:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.role:
                    role = delta.role
                if delta.content:
                    full_content += delta.content
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        # 简化：假设按 index 顺序追加
                        while len(tool_calls) <= tc.index:
                            tool_calls.append({"function": {"arguments": ""}, "id": None, "type": "function"})
                        if tc.function:
                            if tc.function.name:
                                tool_calls[tc.index]["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments
                        if tc.id:
                            tool_calls[tc.index]["id"] = tc.id
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            # 构造 message 字典
            message = {
                "role": role,
                "content": full_content or None,
            }
            if tool_calls:
                message["tool_calls"] = tool_calls

            # 使用 .construct() 绕过验证（因为 usage 可能不全）
            return ChatCompletion.construct(
                id=chunks[0].id,
                choices=[
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason or "stop",
                    }
                ],
                created=chunks[0].created,
                model=chunks[0].model,
                object="chat.completion",
                # usage 可选，流式通常没有真实值
                usage=usage
            )
        full_response = reconstruct_chat_completion(chunks)

        # 4. 计算指标
        with custom_span(
                name="llm_call_metric",
                data={
                    "model": model.model,
                    "model_config": model_settings.to_json_dict() | {"base_url": str(model._client.base_url)}
                },
                disabled=tracing.is_disabled(),
        ) as span_custom:
            if not token_times:
                raise ValueError("No content tokens received")

            ttft = (first_token_time - start_time) * 1000  # ms
            total_decode_time = token_times[-1] - first_token_time  # 秒
            token_count = len(token_times)

            if token_count == 1:
                tpot = 0.0
                itls = [0.0]
            else:
                # ITL: inter-token latencies (between consecutive tokens)
                itls = [
                    (token_times[i] - token_times[i - 1]) * 1000  # ms
                    for i in range(1, len(token_times))
                ]
                tpot = total_decode_time * 1000 / (token_count - 1)  # ms per token

            p99_itl = sorted(itls)[int(0.99 * len(itls))] if itls else 0.0

            mcp_server_call_list = []
            # Skip mcp_hub when servers is empty
            if mcp_hub_v2.mcp_server_manager.servers:
                for server_name in mcp_hub_v2.mcp_server_manager.servers:
                    mcp_server_call_list.append(mcp_hub_v2.mcp_server_manager.servers[server_name])

            metrics = {
                "ttft_ms": round(ttft, 2),
                "tpot_ms": round(tpot, 2),
                "p99_itl_ms": round(p99_itl, 2),
                "token_count": token_count,
                "itls_ms": [round(x, 2) for x in itls],  # 完整序列，用于 sparkline
                "mcp_server_call": mcp_server_call_list,
            }
            span_custom.span_data.data.update({
                'metrics': metrics,
            })
        return full_response

    @classmethod
    def _get_model(cls, agent: Agent[Any], run_config: RunConfig) -> Model:
        model = None
        if isinstance(run_config.model, Model):
            model = run_config.model
        elif isinstance(run_config.model, str):
            model = run_config.model_provider.get_model(run_config.model)
        elif isinstance(agent.model, Model):
            model = agent.model
        return model


class Ddv2Runner(Runner):
    @classmethod
    async def run(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        *,
        context: TContext | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        session: Session | None = None,
    ) -> RunResult:
        """Run a workflow starting at the given agent. The agent will run in a loop until a final
        output is generated. The loop runs like so:
        1. The agent is invoked with the given input.
        2. If there is a final output (i.e. the agent produces something of type
            `agent.output_type`, the loop terminates.
        3. If there's a handoff, we run the loop again, with the new agent.
        4. Else, we run tool calls (if any), and re-run the loop.
        In two cases, the agent may raise an exception:
        1. If the max_turns is exceeded, a MaxTurnsExceeded exception is raised.
        2. If a guardrail tripwire is triggered, a GuardrailTripwireTriggered exception is raised.
        Note that only the first agent's input guardrails are run.
        Args:
            starting_agent: The starting agent to run.
            input: The initial input to the agent. You can pass a single string for a user message,
                or a list of input items.
            context: The context to run the agent with.
            max_turns: The maximum number of turns to run the agent for. A turn is defined as one
                AI invocation (including any tool calls that might occur).
            hooks: An object that receives callbacks on various lifecycle events.
            run_config: Global settings for the entire agent run.
            previous_response_id: The ID of the previous response, if using OpenAI models via the
                Responses API, this allows you to skip passing in input from the previous turn.
            conversation_id:  The conversation ID (https://platform.openai.com/docs/guides/conversation-state?api-mode=responses).
                If provided, the conversation will be used to read and write items.
                Every agent will have access to the conversation history so far,
                and it's output items will be written to the conversation.
                We recommend only using this if you are exclusively using OpenAI models;
                other model providers don't write to the Conversation object,
                so you'll end up having partial conversations stored.
        Returns:
            A run result containing all the inputs, guardrail results and the output of the last
            agent. Agents may perform handoffs, so we don't know the specific type of the output.
        """
        runner = Ddv2AgentRunner()
        return await runner.run(
            starting_agent,
            input,
            context=context,
            max_turns=max_turns,
            hooks=hooks,
            run_config=run_config,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            session=session,
        )

class Ddv2RunImpl(RunImpl):
    @classmethod
    async def execute_function_tool_calls(
            cls,
            *,
            agent: Agent[TContext],
            tool_runs: list[ToolRunFunction],
            hooks: RunHooks[TContext],
            context_wrapper: RunContextWrapper[TContext],
            config: RunConfig,
    ) -> list[FunctionToolResult]:
        async def run_single_tool(
                func_tool: FunctionTool, tool_call: ResponseFunctionToolCall
        ) -> Any:
            result = ''
            with function_span(func_tool.name) as span_fn:
                tool_context = ToolContext.from_agent_context(
                    context_wrapper,
                    tool_call.call_id,
                    tool_call=tool_call,
                )
                if config.trace_include_sensitive_data:
                    span_fn.span_data.input = tool_call.arguments
                try:
                    _, _, result = await asyncio.gather(
                        hooks.on_tool_start(tool_context, agent, func_tool),
                        (
                            agent.hooks.on_tool_start(tool_context, agent, func_tool)
                            if agent.hooks
                            else _coro.noop_coroutine()
                        ),
                        func_tool.on_invoke_tool(tool_context, tool_call.arguments),
                    )

                    await asyncio.gather(
                        hooks.on_tool_end(tool_context, agent, func_tool, result),
                        (
                            agent.hooks.on_tool_end(tool_context, agent, func_tool, result)
                            if agent.hooks
                            else _coro.noop_coroutine()
                        ),
                    )
                except Exception as e:
                    _error_tracing.attach_error_to_current_span(
                        SpanError(
                            message="Error running tool",
                            data={"tool_name": func_tool.name, "error": str(e)},
                        )
                    )
                    if isinstance(e, AgentsException):
                        raise e
                    raise UserError(f"Error running tool {func_tool.name}: {e}") from e

                if config.trace_include_sensitive_data:
                    span_fn.span_data.output = result

            # del performance monitoring metadata to lower tokens usage
            if tool_run.function_tool.name == "document_qa":
                result_obj = json.loads(json.loads(result).get("text"))
                for i in range(0, len(result_obj.get('data',{}))):
                    result_obj.get('data')[i].pop('metadata')
                result = json.dumps({"type": "text", "text": result_obj}, ensure_ascii=False)
            return result

        tasks = []
        for tool_run in tool_runs:
            function_tool = tool_run.function_tool
            tasks.append(run_single_tool(function_tool, tool_run.tool_call))

        results = await asyncio.gather(*tasks)

        return [
            FunctionToolResult(
                tool=tool_run.function_tool,
                output=result,
                run_item=ToolCallOutputItem(
                    output=result,
                    raw_item=ItemHelpers.tool_call_output_item(tool_run.tool_call, str(result)),
                    agent=agent,
                ),
            )
            for tool_run, result in zip(tool_runs, results)
        ]
