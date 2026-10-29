from __future__ import annotations

import json
import logging
import re
import time
import uuid
from html import escape
from typing import Any, List, Optional, Protocol

from features.compression.config import CompressionConfig
from features.compression.conversation_adapter import ConversationAdapter
from features.compression.queue_manager import AsyncQueueManager

logger = logging.getLogger("compression")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_handler)


# ============================================================
# 内部工具函数
# ============================================================


def extract_summary_regex(text: str, start_tag: str, end_tag: str) -> list[str]:
    """使用正则表达式提取两个标签之间的文本"""
    try:
        escaped_start_tag = re.escape(start_tag)
        escaped_end_tag = re.escape(end_tag)
        pattern = f"{escaped_start_tag}(.*?){escaped_end_tag}"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]
    except Exception as e:
        logger.error(f"正则表达式匹配过程中出错：{e}")
        return []


# ============================================================
# 内部协议 + 默认实现
# ============================================================


class SummaryPromptTemplate(Protocol):
    """摘要 Prompt 模板协议（内部）

    CompressionManager 内部自动构建默认实现（ResearchSummaryPrompt），
    构造方法不暴露 prompt_template 参数。
    协议保留供未来扩展，当前无需自行实现。
    """

    def build_prompt(
        self,
        conversation_history: list[dict[str, str]],
        last_action: str,
    ) -> str:
        """构建摘要 Prompt"""
        ...

    def extract_summary(self, llm_output: str) -> str:
        """从 LLM 输出中提取摘要文本"""
        ...

    def extract_action(self, llm_output: str) -> str:
        """从 LLM 输出中提取行动决策"""
        ...


class ResearchSummaryPrompt:
    """默认摘要 Prompt 实现 — 从 compress.py:async_multi_round_compress 迁移"""

    _PROMPT_TEMPLATE = """
    请基于以下多轮对话历史，按照指定的结构化格式进行内容总结。

    要求：
    - 子标题数量根据对话内容决定（至少2个，最多5个）
    - 每个子标题下的具体信息描述应基于对话中的实际搜索结果
    - 候选答案应从对话中合理推断得出
    - 保持客观、准确的总结风格

    【对话历史开始】
    {deep_research_agent_dialog_history}
    【对话历史结束】

    请严格按照以下模板格式进行总结，不要添加额外解释：

    <summary>
    从搜索结果中，我发现了一些有价值的信息，并且已经经过了严谨和充分的验证：

    1. [请从对话中提取第一个关键主题作为子标题]
    - [基于该主题的具体发现1]
    - [基于该主题的具体发现2]
    ...

    2. [请从对话中提取第二个关键主题作为子标题]
    - [基于该主题的具体发现1]
    - [基于该主题的具体发现2]
    ...

    3. [请从对话中提取第三个关键主题作为子标题]
    - [基于该主题的具体发现1]
    - [基于该主题的具体发现2]
    ...

    [根据需要继续添加更多子标题...]

    从这些结果来看，[从对话中推断出的最可能的候选答案]最符合条件：

    - [支持该候选答案的理由1]
    - [支持该候选答案的理由2]
    ...

    </summary>

    <action>
    [从对话历史的最后一轮：
    {last_round_action}
    中复制最终的决策（例如，"现在我需要……"），保持与原始决策完全一致，不需要添加额外解释]
    ...
    </action>

    """

    def build_prompt(
        self,
        conversation_history: list[dict[str, str]],
        last_action: str,
    ) -> str:
        return self._PROMPT_TEMPLATE.format(
            deep_research_agent_dialog_history=conversation_history,
            last_round_action=last_action,
        )

    def extract_summary(self, llm_output: str) -> str:
        results = extract_summary_regex(llm_output, "<summary>", "</summary>")
        if len(results) != 1 or results[0] == "":
            raise ValueError(
                f"Expected exactly 1 non-empty <summary> tag, got {len(results)} results"
            )
        return results[0]

    def extract_action(self, llm_output: str) -> str:
        results = extract_summary_regex(llm_output, "<action>", "</action>")
        if len(results) == 0:
            return ""
        return results[0]


class CompressionTrigger(Protocol):
    """压缩触发条件协议（内部）

    CompressionManager 内部自动从 CompressionConfig 构建
    DefaultCompressionTrigger，构造方法不暴露 trigger 参数。
    协议保留供未来扩展，当前无需自行实现。
    """

    def should_compress(
        self,
        turn: int,
        token_count: int,
        last_tool_name: Optional[str],
        has_pending_compression: bool,
    ) -> bool:
        """判断是否应触发压缩"""
        ...


class DefaultCompressionTrigger:
    """默认压缩触发策略 — 从 agent_runner.py 迁移"""

    def __init__(self, config: CompressionConfig):
        self._token_threshold = config.token_threshold
        self._trigger_tool_names = config.trigger_tool_names

    def should_compress(
        self,
        turn: int,
        token_count: int,
        last_tool_name: Optional[str],
        has_pending_compression: bool,
    ) -> bool:
        if last_tool_name not in self._trigger_tool_names:
            logger.info(
                f"should_compress=False: tool_name={last_tool_name} not in trigger_tools={self._trigger_tool_names}"
            )
            return False
        if token_count <= self._token_threshold:
            logger.info(
                f"should_compress=False: token_count={token_count} <= threshold={self._token_threshold}"
            )
            return False
        if has_pending_compression:
            logger.info(
                f"should_compress=False: THERE ARE UNFINISHED COMPRESSION TASK, SKIP THIS COMPRESSION!!"
            )
            return False
        logger.info(
            f"should_compress=True: tool_name={last_tool_name} in trigger_tools, token_count={token_count} > threshold={self._token_threshold}, no pending task"
        )
        return True


# ============================================================
# LLM 调用
# ============================================================


async def send_compress_request(
    prompt: str, config: CompressionConfig
) -> tuple[str, dict]:
    """发送压缩 LLM 请求 — 从 async_queue_manager.py:send_request 迁移

    改用 CompressionConfig 中的 LLM 配置，替代全局 config 单例。
    """
    headers = {"Content-Type": "application/json"}
    if config.compress_llm_api_key:
        headers["Authorization"] = config.compress_llm_api_key

    data = {
        "model": config.compress_llm_model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": config.compress_llm_temperature,
        "max_tokens": config.compress_llm_max_tokens,
        "stream": False,
    }
    if "qwen" in config.compress_llm_model_name.lower():
        data["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        import aiohttp

        logger.info(
            f"Sending LLM request to {config.compress_llm_url}, "
            f"model={config.compress_llm_model_name}"
        )
        async with aiohttp.ClientSession() as session:
            last_content = ""
            last_usage: dict = {}
            for attempt in range(2):
                request_data = dict(data)
                if attempt == 1:
                    request_data["temperature"] = 0.0
                    request_data["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "compression_result",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "summary": {"type": "string"},
                                    "action": {"type": "string"},
                                },
                                "required": ["summary", "action"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        },
                    }
                    request_data["messages"] = [
                        {
                            "role": "system",
                            "content": (
                                "你是结构化压缩格式器。只输出一个 JSON 对象，"
                                "其中 summary 是非空摘要字符串，action 是下一步"
                                "行动字符串；不要输出 Markdown 或额外解释。"
                            ),
                        },
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": last_content},
                        {
                            "role": "user",
                            "content": (
                                "上一条回复未遵守格式。请严格按 JSON Schema 重写，"
                                "只返回 summary 和 action 字段。"
                            ),
                        },
                    ]

                async with session.post(
                    config.compress_llm_url, headers=headers, json=request_data
                ) as response:
                    result_text = await response.text()
                    result_dict = json.loads(result_text)
                    last_content = result_dict["choices"][0]["message"]["content"]
                    last_usage = result_dict.get("usage", {})
                    if attempt == 1:
                        try:
                            structured = json.loads(last_content)
                            summary = structured.get("summary", "").strip()
                            action = structured.get("action", "").strip()
                            if summary:
                                last_content = (
                                    f"<summary>{escape(summary)}</summary>\n"
                                    f"<action>{escape(action)}</action>"
                                )
                        except (json.JSONDecodeError, AttributeError):
                            logger.warning(
                                "Structured compression repair returned invalid JSON"
                            )
                    summary_tags = extract_summary_regex(
                        last_content, "<summary>", "</summary>"
                    )
                    logger.info(
                        "LLM request succeeded, attempt=%s, summary_tags=%s, usage=%s",
                        attempt + 1,
                        len(summary_tags),
                        last_usage,
                    )
                    if len(summary_tags) == 1 and summary_tags[0]:
                        return last_content, last_usage
                    if attempt == 0:
                        logger.warning(
                            "Compression output missed the required summary tag; "
                            "retrying once with a strict repair prompt"
                        )

            return last_content, last_usage
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return "", {}


# ============================================================
# CompressionManager — 核心编排器
# ============================================================


class CompressionManager:
    """多轮异步压缩管理器 — Agent 唯一需要交互的接口

    对外接口仅两个方法：
        on_turn_start() — 轮次开始时调用，消费已完成的压缩结果
        on_turn_end()   — 轮次结束时调用，检查触发条件并提交压缩任务

    设计决策：
    - 不暴露 shutdown：AsyncQueueManager 的后台线程为 daemon 线程，
      进程退出时自动清理，无需调用者管理生命周期
    - AsyncQueueManager 并发控制：max_pending_tasks 由 QM 构造方法接收，
      submit_async_task 超限时返回 False，由 CompressionManager 决定如何处理
    - AsyncQueueManager 返回值：get_result 返回 AsyncTaskResult 命名元组，
      替代脆弱的 5-tuple 隐式契约
    - callback 保留：submit_async_task 的 callback 参数为 KV 预热设计，
      当前未启用但保留，CompressionManager 提交时传入 config.kv_warmup_callback
    - 不暴露 trigger 参数：内部自动从 config 构建 DefaultCompressionTrigger，
      协议保留供未来扩展，当前无需自行实现
    - 不暴露 prompt_template 参数：内部自动构建 ResearchSummaryPrompt，
      协议保留供未来扩展，当前无需自行实现
    - Tracing 内部管理：压缩相关的 span 创建/更新/finish 均在 Manager 内部完成，
      调用者无需关心 last_think_idx / final_summary / excluded_indices 等元数据
    - 编排依赖通过 ConversationAdapter 承担：agent, run_config, context_wrapper,
      system_prompt 等在适配器构造时注入，不污染 Manager 的调用签名

    用法：
        manager = CompressionManager(config, adapter)

        # 在 Agent 主循环中：
        generated_items = manager.on_turn_start(current_turn, generated_items)
        # ... LLM 推理 + 工具执行 ...
        manager.on_turn_end(current_turn, generated_items, tool_name, token_count)
    """

    def __init__(
        self,
        config: CompressionConfig,
        adapter: ConversationAdapter,
    ):
        self._config = config
        self._adapter = adapter
        self._prompt = ResearchSummaryPrompt()
        self._trigger = DefaultCompressionTrigger(config)
        self._queue = AsyncQueueManager(max_pending_tasks=config.max_pending_tasks)

        # 压缩状态（直接作为属性管理，不单独建类）
        self._compress_id_multi_round_dict: dict = {}
        self._iteration_multi_round_compress_submit: int = 0
        self._active_compress_spans: dict = {}
        self._already_compressed_num: int = 0

    def set_adapter(self, adapter: ConversationAdapter) -> None:
        """设置对话适配器（支持延迟初始化）

        当 adapter 在构造时无法提供（依赖 agent, run_config 等），
        可先传 None，后续通过此方法设置。
        """
        self._adapter = adapter

    def on_turn_start(self, current_turn: int, generated_items: list) -> list:
        """消费已完成的压缩结果，重构对话历史

        Args:
            current_turn: 当前轮次编号
            generated_items: 当前对话历史（Agent 的对话项列表）

        Returns:
            修改后的 generated_items（压缩后重构的对话历史）

        内部处理：
        - 检查 config.enabled 开关，未启用则直接返回原列表
        - 遍历已完成的压缩任务（从 AsyncQueueManager 获取结果）
        - 过期判断：current_turn - submit_turn <= staleness_threshold
        - 调用 adapter.find_last_tool_call() 定位摘要注入点
        - 调用 adapter.set_tool_call_content() 注入摘要
        - 调用 adapter.reconstruct_history() 重构对话
        - 内部管理 tracing span（创建/更新/finish），不暴露给调用者
        - 更新内部 state（compress_id_multi_round_dict, already_compressed_num）
        """
        if not self._config.enabled:
            return generated_items

        finished_task_ids = set(self._queue.get_finished_task_ids())
        pending_count = len(self._compress_id_multi_round_dict)
        consumable_count = len([t for t in self._compress_id_multi_round_dict if t in finished_task_ids])
        logger.info(
            f"on_turn_start: turn={current_turn}, pending_tasks={pending_count}, "
            f"consumable_tasks={consumable_count}, items_len={len(generated_items)}"
        )

        for tid in list(self._compress_id_multi_round_dict.keys()):
            if tid not in finished_task_ids:
                continue

            generated_items = self._consume_compress_result(
                tid, current_turn, generated_items
            )

        return generated_items

    def on_turn_end(
        self,
        current_turn: int,
        generated_items: list,
        last_tool_name: str,
        token_count: int,
    ) -> None:
        """检查触发条件，提交新的压缩任务

        Args:
            current_turn: 当前轮次编号
            generated_items: 当前对话历史
            last_tool_name: 本轮最后执行的工具名
            token_count: 本轮 token 数

        内部处理：
        - 调用 trigger.should_compress() 判断是否触发
        - 调用 adapter.to_messages(items, system_prompt) 转换对话格式
          （system_prompt 由 adapter 构造时持有，无需每次传入）
        - 构造 task_id，记录元数据到 state
        - 创建 tracing span
        - 提交异步压缩任务到 AsyncQueueManager
        - 若 config.kv_warmup_callback 非空，执行 KV 预热回调
        - 更新内部 state（iteration_multi_round_compress_submit）
        """
        if not self._config.enabled:
            return

        # 防御：流式响应可能不返回 usage，导致 token_count=0，
        # 永远不满足阈值，压缩永远不触发
        if token_count == 0:
            token_count = len(generated_items) * 200
            logger.info(
                f"token_count=0, fallback estimate: {token_count} "
                f"(items={len(generated_items)})"
            )

        has_pending = len(self._compress_id_multi_round_dict) > 0
        logger.info(
            f"[async_compress] submit check: tool_name={last_tool_name}, "
            f"token_num={token_count}, pending_tasks={len(self._compress_id_multi_round_dict)}"
        )
        if not self._trigger.should_compress(
            turn=current_turn,
            token_count=token_count,
            last_tool_name=last_tool_name,
            has_pending_compression=has_pending,
        ):
            return

        self._submit_compress_task(current_turn, generated_items, token_count)

    def _submit_compress_task(self, current_turn: int, items: list, token_count: int = 0):
        """提交异步压缩任务（内部方法）"""
        # 浅拷贝 items，切断闭包引用，防止异步任务执行时 items 被外部修改
        items = list(items)
        logger.info(
            f"_submit_compress_task enter: turn={current_turn}, items_len={len(items)}, token_count={token_count}"
        )
        task_id = f"comp_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        # 记录元数据
        self._compress_id_multi_round_dict[task_id] = {
            "compress_till_idx_not_inc": len(items),
            "token_num_before_compress": token_count,
            "ts_before_compress": time.time(),
        }

        # 创建 tracing span
        try:
            from agents.tracing import custom_span

            span = custom_span(
                "generation_resum_compress",
                data={
                    "task_id": task_id,
                    "turn_submitted": current_turn,
                    "compress_till_idx_not_inc": len(items) - 1,
                    "already_compressed_num": self._already_compressed_num,
                },
            )
            span.start()
            self._active_compress_spans[task_id] = span
            logger.info(
                f"tracing span created: task_id={task_id}"
            )
        except Exception:
            logger.warning(f"Failed to create tracing span")

        # 提交异步任务
        async def _do_compress():
            logger.info(
                f"_do_compress start: task_id={task_id}"
            )
            try:
                messages = await self._adapter.to_messages(items)
            except Exception as e:
                logger.error(
                    f"_do_compress to_messages failed: task_id={task_id}, error={e}",
                    exc_info=True,
                )
                raise
            last_action = messages[-2] if len(messages) >= 2 else ""
            logger.info(
                f"Compress task {task_id}: converted to {len(messages)} messages, "
                f"last_action_len={len(str(last_action))}"
            )
            prompt = self._prompt.build_prompt(messages, str(last_action))
            result, _ = await send_compress_request(prompt, self._config)
            logger.info(
                f"_do_compress done: task_id={task_id}, result_len={len(result) if result else 0}"
            )
            return result

        callback = self._config.kv_warmup_callback
        logger.info(
            f"submitting to queue: task_id={task_id}, has_callback={callback is not None}"
        )
        submitted = self._queue.submit_async_task(_do_compress, task_id, callback)
        if not submitted:
            logger.warning(
                f"task {task_id} rejected: max_pending_tasks reached"
            )
            self._compress_id_multi_round_dict.pop(task_id, None)
            self._active_compress_spans.pop(task_id, None)
            return

        self._iteration_multi_round_compress_submit = current_turn
        logger.info(
            f"[async_compress] task submitted: {task_id}, "
            f"token_num_before={token_count}, turn={current_turn}"
        )

    def _consume_compress_result(
        self, task_id: str, current_turn: int, generated_items: list
    ) -> list:
        """消费单个压缩结果，重构对话历史（内部方法）"""
        logger.info(
            f"_consume_compress_result enter: task_id={task_id}, turn={current_turn}"
        )
        task_result = self._queue.get_result(task_id)
        logger.info(
            f"get_result: task_id={task_id}, has_result={task_result.result is not None}, has_error={task_result.error is not None}"
        )

        if task_result.result is not None:
            if task_result.result == "":
                logger.warning(
                    f"task {task_id} returned empty result"
                )
            # 过期判断
            if (
                current_turn - self._iteration_multi_round_compress_submit
                > self._config.staleness_threshold
            ):
                logger.info(
                    f"result for {task_id} is stale "
                    f"(turns since submit: {current_turn - self._iteration_multi_round_compress_submit}), "
                    f"skipping"
                )
            elif task_result.result != "":
                try:
                    generated_items, excluded_count = self._apply_compression(
                        task_id, task_result.result, generated_items
                    )
                    logger.info(
                        f"[async_compress] task completed: {task_id}, "
                        f"turn={current_turn}, excluded_items={excluded_count}"
                    )
                except Exception as e:
                    logger.error(f"Failed to apply compression result: {e}")

        if task_result.error is not None:
            logger.error(
                f"task {task_id} failed: {task_result.error}"
            )

        # 清理 state
        self._compress_id_multi_round_dict.pop(task_id, None)

        # 更新 tracing span
        span = self._active_compress_spans.pop(task_id, None)
        if span:
            try:
                span.span_data.data["turn_finished"] = current_turn
                span.span_data.data["compression_success"] = (
                    task_result.result is not None
                )
            except Exception:
                pass
            finally:
                span.finish()

        return generated_items

    def _apply_compression(
        self, task_id: str, llm_output: str, generated_items: list
    ) -> tuple[list, int]:
        """应用压缩结果到对话历史（内部方法）

        Returns:
            (重构后的 generated_items, 本次排除的项数)
        """
        logger.info(
            f"_apply_compression enter: task_id={task_id}, items_len={len(generated_items)}, llm_output_len={len(llm_output)}"
        )
        # 提取 summary 和 action
        summ_text = self._prompt.extract_summary(llm_output)
        act_text = self._prompt.extract_action(llm_output)
        final_summary = f"{summ_text}\n\n{act_text}"

        # 定位摘要注入点
        meta = self._compress_id_multi_round_dict[task_id]
        compress_till_idx_not_inc = meta["compress_till_idx_not_inc"]

        last_think_idx = self._adapter.find_last_tool_call(
            generated_items,
            self._config.think_tool_name,
            compress_till_idx_not_inc,
        )
        if last_think_idx is None:
            raise ValueError(
                f"No '{self._config.think_tool_name}' tool call found before index {compress_till_idx_not_inc}"
            )

        # 注入摘要
        self._adapter.set_tool_call_content(
            generated_items[last_think_idx], final_summary
        )

        # 计算被排除的项（在重构前计算，基于原始 items）
        excluded_indices = [
            i
            for i, item in enumerate(generated_items)
            if i < compress_till_idx_not_inc
            and not self._adapter.is_tool_call(item)
        ]
        # 第一个 excluded index 通常是 user message（items[1]），
        # 不属于搜索结果，不计入已压缩数量
        if excluded_indices:
            excluded_indices.pop(0)
        self._already_compressed_num += len(excluded_indices)

        # 重构对话历史
        items_len_before = len(generated_items)
        generated_items = self._adapter.reconstruct_history(
            generated_items,
            compress_till_idx_not_inc,
            self._config.header_item_count,
        )
        logger.info(
            f"applied: task_id={task_id}, boundary={compress_till_idx_not_inc}, "
            f"last_think_idx={last_think_idx}, excluded_count={len(excluded_indices)}, "
            f"items_len: {items_len_before} -> {len(generated_items)}"
        )

        return generated_items, len(excluded_indices)
