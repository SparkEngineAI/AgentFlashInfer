import json
import json5
import os
import random
import re
import datetime
import time
import asyncio
from typing import Dict, List, Optional, Union, Callable
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.tools import BaseTool

from examples.deepresearch.config.deepresearch_config import (
    VLLM_PORTS, MAX_LLM_CALL_PER_RUN, MAX_TIMEOUT_MINUTES, MAX_TOKEN_LIMIT,
    GENERATION_TEMPERATURE, GENERATION_TOP_P, GENERATION_PRESENCE_PENALTY, GENERATION_MAX_TOKENS,
    MODEL_ROUTING_SWITCH, SMALL_MODEL_URL, LARGE_MODEL_URL, SMALL_MODEL_NAME, LARGE_MODEL_NAME,
    SMALL_MODEL_API_KEY, LARGE_MODEL_API_KEY, PROGRESS_CHECK_MIN_ITER, LARGE_ROUNDS, EARLY_STOP_ENABLED, 
    EARLY_STOP_MIN_ITER,
    COMPRESSION_ENABLED, COMPRESSION_TOKEN_THRESHOLD, COMPRESSION_TRIGGER_TOOLS,
    COMPRESSION_LLM_URL, COMPRESSION_LLM_API_KEY, COMPRESSION_LLM_MODEL_NAME
)
from examples.deepresearch.features.model_router import DRModelRouter, DRProgressCheckStrategy
from features.model_routing.config import ModelRoutingConfig
from features.compression.config import CompressionConfig
from features.compression.manager import CompressionManager
from features.compression.adapters.deepresearch import DeepResearchConversationAdapter
from examples.deepresearch.tools.prompt import SYSTEM_PROMPT
from examples.deepresearch.tools.tool_file import FileParser
from examples.deepresearch.tools.tool_scholar import Scholar
from examples.deepresearch.tools.tool_python import PythonInterpreter
from examples.deepresearch.tools.tool_search import Search
from examples.deepresearch.tools.tool_visit import Visit
from features.early_stop import EarlyStopManager

es_manager = EarlyStopManager()

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

# MAX_LLM_CALL_PER_RUN is imported from config

TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

# Round-robin port selection
_port_index = 0


def _get_next_port() -> int:
    global _port_index
    port = VLLM_PORTS[_port_index % len(VLLM_PORTS)]
    _port_index += 1
    return port


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


def _normalize_model_output(content: str) -> str:
    """Normalize recoverable Qwen output quirks at the API boundary."""
    content = content.strip()
    if "</think>" in content and "<think>" not in content:
        content = "<think>\n" + content

    tool_start = content.find("<tool_call>")
    tool_end = content.find("</tool_call>", tool_start)
    if tool_start == -1 or tool_end == -1:
        return content

    payload_start = tool_start + len("<tool_call>")
    payload = content[payload_start:tool_end]
    try:
        json5.loads(payload)
        return content
    except ValueError:
        pass

    base = re.sub(
        r'([{,]\s*)"name="\s*',
        r'\1"name": "',
        payload.strip(),
        count=1,
    )
    valid_candidates = []
    for brace_delta in range(-2, 3):
        if brace_delta < 0:
            removed = -brace_delta
            if not base.endswith("}" * removed):
                continue
            candidate = base[:-removed]
        else:
            candidate = base + ("}" * brace_delta)
        try:
            parsed = json5.loads(candidate)
        except ValueError:
            continue
        if (
            isinstance(parsed, dict)
            and isinstance(parsed.get("name"), str)
            and isinstance(parsed.get("arguments"), dict)
        ):
            valid_candidates.append(candidate)

    valid_candidates = list(dict.fromkeys(valid_candidates))
    if len(valid_candidates) != 1:
        return content
    repaired_payload = valid_candidates[0] + "\n"
    return content[:payload_start] + repaired_payload + content[tool_end:]


class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 compression_config: Optional[CompressionConfig] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self._tokenizer = None
        self.model_router = None
        self._compression_config = compression_config
        self._compression_manager: Optional[CompressionManager] = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path, trust_remote_code=True)
        return self._tokenizer

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content
    
    def call_server(self, msgs, planning_port, max_tries=10):

        if self.model_router and self.model_router.config.enabled:
            endpoint = self.model_router.resolve("DeepResearch", msgs)
            openai_api_key = endpoint.api_key
            openai_api_base = endpoint.base_url
            model_to_use = endpoint.model_name
        else:
            openai_api_key = "EMPTY"
            openai_api_base = f"http://127.0.0.1:{planning_port}/v1"
            model_to_use = self.model

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1 
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=model_to_use,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    logprobs=True,
                    max_tokens=10000,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
                )
                content = chat_response.choices[0].message.content

                # OpenRouter provides API calling. If you want to use OpenRouter, you need to uncomment line 89 - 90.
                # reasoning_content = "<think>\n" + chat_response.choices[0].message.reasoning.strip() + "\n</think>"
                # content = reasoning_content + content                
                
                if content and content.strip():
                    print("--- Service call successful, received a valid response ---")
                    return _normalize_model_output(content)
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30) 
                
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")
        
        return f"vllm server error!!!"

    def count_tokens(self, messages):
        tokenizer = self.tokenizer
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(full_prompt, return_tensors="pt")
        token_count = len(tokens["input_ids"][0])
        return token_count

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        start_time = time.time()
        planning_port = data['planning_port']
        answer = data['item']['answer']
        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
 
        # Initialize CompressionManager if config is provided
        compression_manager = None
        if self._compression_config is not None and self._compression_config.enabled:
            adapter = DeepResearchConversationAdapter(
                system_prompt=system_prompt,
                think_tag="think",
            )
            compression_manager = CompressionManager(
                config=self._compression_config,
                adapter=adapter,
            )
            print(f"CompressionManager initialized: threshold={self._compression_config.token_threshold}, "
                  f"trigger_tools={self._compression_config.trigger_tool_names}")
 
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > MAX_TIMEOUT_MINUTES * 60:
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages, planning_port)
            print(f'Round {round}: {content}')
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    if "python" in tool_call.lower():
                        try:
                            code_raw=content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                            result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                        except:
                            result = "[Python Interpreter Error]: Formatting error."

                    else:
                        tool_call = json5.loads(tool_call)
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('arguments', {})
                        result = self.custom_call_tool(tool_name, tool_args)

                except:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                # print(result)
                messages.append({"role": "user", "content": result})
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = 110 * 1024
            token_count = self.count_tokens(messages)
            print(f"Round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                content = self.call_server(messages, planning_port)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }
        return result

    
    def _run_with_events(self, data: dict, model: str, sse_emitter: Callable, **kwargs) -> dict:
        """Run the ReAct loop, emitting SSE events at each step."""
        self.model = model
        try:
            question = data['item']['question']
        except:
            raw_msg = data['item']['messages'][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

        start_time = time.time()
        planning_port = data['planning_port']
        answer = data['item'].get('answer', '')
        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round_num = 0

        compression_manager = None
        if self._compression_config is not None and self._compression_config.enabled:
            adapter = DeepResearchConversationAdapter(
                system_prompt=system_prompt,
                think_tag="think",
            )
            compression_manager = CompressionManager(
                config=self._compression_config,
                adapter=adapter,
            )
            print(f"CompressionManager initialized: threshold={self._compression_config.token_threshold}, "
                  f"trigger_tools={self._compression_config.trigger_tool_names}")

        if MODEL_ROUTING_SWITCH:
            routing_config = ModelRoutingConfig(
                enabled=True,
                small_model_url=SMALL_MODEL_URL,
                large_model_url=LARGE_MODEL_URL,
                small_model_name=SMALL_MODEL_NAME,
                large_model_name=LARGE_MODEL_NAME,
                small_model_api_key=SMALL_MODEL_API_KEY,
                large_model_api_key=LARGE_MODEL_API_KEY,
            )
            strategy = DRProgressCheckStrategy(
                config=routing_config,
                progress_check_min_iter=PROGRESS_CHECK_MIN_ITER,
                large_rounds=LARGE_ROUNDS,
                trigger_tool_name="<think>"
            )
            self.model_router = DRModelRouter(config=routing_config, strategy=strategy)
        agent_id = es_manager.gen_agent_id()
        es_manager.init_agent(agent_id)
        pending_early_stop = None
        while num_llm_calls_available > 0:
            if time.time() - start_time > MAX_TIMEOUT_MINUTES * 60:
                sse_emitter("on_react_step", {"type": "termination", "reason": "timeout", "round": round_num})
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": 'No answer found after timeout',
                    "termination": 'No answer found after timeout'
                }
                es_manager.remove_agent(agent_id)
                return result

            round_num += 1
            num_llm_calls_available -= 1

            # Consume compression results before LLM call
            if compression_manager is not None:
                messages = compression_manager.on_turn_start(round_num, messages)

            if pending_early_stop is not None:
                if any(
                    message.get("role") == "user"
                    and message.get("content") == pending_early_stop["content"]
                    for message in messages
                ):
                    print(
                        "[early_stop] prompt observed in next turn: "
                        f"reason={pending_early_stop['reason']}, "
                        f"injected_iter={pending_early_stop['iter']}, "
                        f"current_iter={round_num}"
                    )
                pending_early_stop = None
 
            content = self.call_server(messages, planning_port)
            print(f'Round {round_num}: {content}')

            if OBS_START in content:
                pos = content.find(OBS_START)
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})     

            # Emit thinking event
            sse_emitter("on_react_step", {
                "type": "thinking",
                "content": content[:2000],
                "round": round_num,
                "model": self.model_router.current_model if self.model_router else "default"
            })

            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                tool_name = ''
                tool_args = {}
                try:
                    if "python" in tool_call.lower():
                        tool_name = 'PythonInterpreter'
                        tool_args = {"code": tool_call}
                        sse_emitter("on_react_step", {
                            "type": "tool_call",
                            "tool_name": tool_name,
                            "tool_args": {},
                            "round": round_num
                        })
                        try:
                            code_raw = content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                            result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                        except:
                            result = "[Python Interpreter Error]: Formatting error."
                    else:
                        tool_call_parsed = json5.loads(tool_call)
                        tool_name = tool_call_parsed.get('name', '')
                        tool_args = tool_call_parsed.get('arguments', {})
                        sse_emitter("on_react_step", {
                            "type": "tool_call",
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "round": round_num
                        })
                        result = self.custom_call_tool(tool_name, tool_args)

                except:
                    tool_name = 'unknown'
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'

                # Emit tool_result event
                sse_emitter("on_react_step", {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "content": result[:2000],
                    "round": round_num
                })

                result = OBS_START + "\n" + result + "\n" + OBS_END
                messages.append({"role": "user", "content": result})

                # Submit compression task after tool execution
                if compression_manager is not None:
                    token_count = self.count_tokens(messages)
                    compression_manager.on_turn_end(
                        current_turn=round_num,
                        generated_items=messages,
                        last_tool_name=tool_name,
                        token_count=token_count,
                    )
 
            if '<answer>' in content and '</answer>' in content:
                prediction = content.split('<answer>')[1].split('</answer>')[0]
                sse_emitter("on_react_step", {
                    "type": "answer",
                    "content": prediction,
                    "round": round_num
                })
                if compression_manager is not None:
                    messages = compression_manager.on_turn_start(
                        round_num + 1, messages
                    )
                termination = 'answer'
                break

            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = MAX_TOKEN_LIMIT
            token_count = self.count_tokens(messages)
            print(f"Round: {round_num}, token count: {token_count}")

            # When compression is enabled, skip direct truncation — compression handles it
            if token_count > max_tokens and compression_manager is None:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")
                messages[-1]['content'] = (
                    "You have now reached the maximum context length you can handle. "
                    "You should stop making tool calls and, based on all the information above, "
                    "think again and provide what you consider the most likely answer in the following format:"
                    "your final thinking\n<answer>your answer</answer>"
                )
                content = self.call_server(messages, planning_port)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                sse_emitter("on_react_step", {
                    "type": "answer",
                    "content": prediction,
                    "round": round_num
                })
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                es_manager.remove_agent(agent_id)
                return result
            
            if '<think>' in content and '</think>' in content:
                think_content = content.split('<think>')[1].split('</think>')[0]
                es_manager.refresh_config_from_env(EARLY_STOP_ENABLED, EARLY_STOP_MIN_ITER)
                es_result = es_manager.check(
                    agent_id=agent_id,
                    current_turn=round_num,
                    tool_name="think",
                    current_thought=think_content,
                )
                if es_result.should_inject:
                    injected_item = {
                        "role": "user",
                        "content": es_result.injected_message.content,
                    }
                    messages.append(injected_item)
                    pending_early_stop = {
                        "content": injected_item["content"],
                        "reason": es_result.reason,
                        "iter": round_num,
                    }
                    print(
                        f"[early_stop] prompt injected: reason={es_result.reason}, "
                        f"iter={round_num}, target=new_step_items"
                    )

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'

        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }
        es_manager.remove_agent(agent_id)
        return result

    async def async_run(self, data: dict, model: str, sse_emitter: Callable = None, **kwargs) -> dict:
        """Run agent in thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        if sse_emitter:
            result = await loop.run_in_executor(
                None, self._run_with_events, data, model, sse_emitter
            )
        else:
            result = await loop.run_in_executor(
                None, self._run, data, model
            )
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            if "python" in tool_name.lower():
                result = TOOL_MAP['PythonInterpreter'].call(tool_args)
            elif tool_name == "parse_file":
                params = {"files": tool_args["files"]}
                
                raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
                result = raw_result

                if not isinstance(raw_result, str):
                    result = str(raw_result)
            else:
                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
                result = raw_result
            return result

        else:
            return f"Error: Tool {tool_name} not found"
