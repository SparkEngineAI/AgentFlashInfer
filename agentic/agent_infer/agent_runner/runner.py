from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, cast, get_args

from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseOutputItemDoneEvent,
)
from openai.types.responses.response_prompt_param import (
    ResponsePromptParam,
)
from typing_extensions import NotRequired, TypedDict, Unpack

from agents._run_impl import (
    AgentToolUseTracker,
    NextStepFinalOutput,
    NextStepHandoff,
    NextStepRunAgain,
    QueueCompleteSentinel,
    RunImpl,
    SingleStepResult,
    TraceCtxManager,
    get_model_tracing_impl,
)
from agents.agent import Agent
from agents.agent_output import AgentOutputSchema, AgentOutputSchemaBase
from agents.exceptions import (
    AgentsException,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OutputGuardrailTripwireTriggered,
    RunErrorDetails,
    UserError,
)
from agents.guardrail import (
    InputGuardrail,
    InputGuardrailResult,
    OutputGuardrail,
    OutputGuardrailResult,
)
from agents.handoffs import Handoff, HandoffInputFilter, handoff
from agents.items import (
    ItemHelpers,
    ModelResponse,
    RunItem,
    ToolCallItem,
    ToolCallOutputItem,
    ToolCallItemTypes,
    TResponseInputItem,
)
from agents.lifecycle import RunHooks
from agents.logger import logger
from agents.memory import Session, SessionInputCallback
from agents.model_settings import ModelSettings
from agents.models.interface import Model, ModelProvider
from agents.models.multi_provider import MultiProvider
from agents.result import RunResult, RunResultStreaming
from agents.run_context import RunContextWrapper, TContext
from agents.stream_events import AgentUpdatedStreamEvent, RawResponsesStreamEvent, RunItemStreamEvent
from agents.tool import Tool
from agents.tracing import Span, SpanError, agent_span, get_current_trace, trace
from agents.tracing.span_data import AgentSpanData
from agents.usage import Usage
from agents.util import _coro, _error_tracing
from agents.util._types import MaybeAwaitable
from agents.run import AgentRunner, RunOptions, DEFAULT_MAX_TURNS, RunConfig, _copy_str_or_list, Runner
from transformers import AutoTokenizer
import tiktoken
import config.config as config
import json
from agents.items import MessageOutputItem
from openai.types.responses.response_output_message import ResponseOutputMessage, ResponseOutputText
import uuid

class MyAgentRunner(AgentRunner):
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

            input_guardrail_results: list[InputGuardrailResult] = []

            current_span: Span[AgentSpanData] | None = None
            current_agent = starting_agent
            should_run_agent_start_hooks = True

            # save only the new user input to the session, not the combined history
            await self._save_result_to_session(session, original_user_input, [])

            try:
                while True:
                    all_tools = await AgentRunner._get_all_tools(current_agent, context_wrapper)

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

                    # 倒数第三轮以及达到最大token数则增加提示
                    current_tokens = context_wrapper.usage.total_tokens
                    if current_turn == max_turns-3 or current_tokens > config.TOKEN_THRESHOLD:
                        if current_tokens > config.TOKEN_THRESHOLD:
                            logger.warning(
                                f"Token count {current_tokens} exceeds threshold {config.TOKEN_THRESHOLD}. "
                                f"Attempting to rollback."
                            )
                            # 一轮会产生两个item，分别是ToolCallItem和ToolCallOutputItem，pop两次来清除上一轮对话
                            if isinstance(generated_items[-1], (ToolCallItem, ToolCallOutputItem)):
                                generated_items.pop()
                            if isinstance(generated_items[-1], (ToolCallItem, ToolCallOutputItem)):
                                generated_items.pop()
                        followup_prompt = {
                            "PlannerAgent" : "Due to length and number of rounds restrictions, you must now call the `planner_subjective_task_done` or `planner_objective_task_done` tool to report the completion of your task. /no_think",
                            "SubjectiveInformationSeekerAgent" : "Due to length and number of rounds restrictions, you must now call the `info_seeker_subjective_task_done` tool to report the completion of your task. /no_think",
                            "ObjectiveInformationSeekerAgent" : "Due to length and number of rounds restrictions, you must now call the `info_seeker_objective_task_done` tool to report the completion of your task. /no_think",
                            "WriterAgent" : "Due to length and number of rounds restrictions, you must now call the `writer_subjective_task_done` tool to report the completion of your task. /no_think"
                        }

                        text_object = ResponseOutputText(
                                text=followup_prompt[current_agent.name],
                                annotations=[],  
                                type="output_text"  
                            )
                        raw_message = ResponseOutputMessage(
                           id=f"msg_{uuid.uuid4()}",
                           content=[text_object],
                           role="assistant",  
                           status="completed",
                           type="message"
                       )
                        injected_item = MessageOutputItem(
                           agent=current_agent,
                           raw_item=raw_message
                       )
                        generated_items.append(injected_item)
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
                raise
            finally:
                if current_span:
                    current_span.finish(reset_current=True)

class MyRunner(Runner):
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
        runner = MyAgentRunner()
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


