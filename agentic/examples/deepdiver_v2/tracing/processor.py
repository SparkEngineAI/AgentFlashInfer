import asyncio
import queue
import threading
import time
from typing import Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from agents.tracing import Span, trace, Trace, add_trace_processor
from datetime import datetime
from typing import Dict, List
from agents.tracing.processor_interface import TracingProcessor
import json
import logging

from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{config.WORKSPACE_PATH}/RunnerLogsHook.log', encoding='utf-8'),
        ]
)
logger = logging.getLogger(__name__)

@dataclass
class SSEEvent:
    session_id: str
    trace_id: str
    event_data: str
    is_volatile: bool = False # True for instant token stream; False for event stream.



class TraceSummaryStats:
    """
    Self-managed class, real-time parsing of the openAI-Agent's trace/span events, and update of trace statistics information
    """
    _lock = asyncio.Lock()

    # session-> [unique instance identifier key->All the TraceSummaryStats instance of per session]
    _instance_map: Dict[str, Dict[str, "TraceSummaryStats"]] = {}

    # session-> [unique instance identifier key] :  save the order of instance creation
    _order: Dict[str, List[str]] = {}

    # session-> [span_id -> parant_name]: save the father agent name for each span
    _id_to_agent_name_map: Dict[str, Dict[str, str]] = {}

    # session-> the duration of the current trace
    _query_latency: Dict[str, float] = {}

    def __init__(self, metrics: str, agent: str, tool: str):
        self.metrics = metrics
        self.agent = agent
        self.tool = tool
        self._call_latency: List[float] = []

    def __repr__(self):
        return (
            f"TraceSummaryStats("
            f"field1={self.metrics}, "
            f"field2={self.agent}, "
            f"field2={self.tool}, "
            f"tools_list={self._call_latency!r}"
            f")"
        )

    def to_dict(self):
        return {"metrics": self.metrics, "agent": self.agent, "tool": self.tool, "_call_latency": self._call_latency}

    @classmethod
    def parsing_session_stats(cls, event: Dict[str, Any], session_id: str):
        """Parse the current event in the current event stream and update the trace statistics"""
        span_id = event.get("span_id", "")
        if event.get("span_type", "") == "agent":
            if session_id not in cls._id_to_agent_name_map:
                cls._id_to_agent_name_map[session_id] = {}
            cls._id_to_agent_name_map[session_id][span_id] = event["data"]["agent_name"]
        parent_span_id = event.get("parent_id", "")
        event["parent_agent"] = cls._id_to_agent_name_map.get(session_id, {}).get(parent_span_id, "")

        if event["event"] == "on_trace_start":
            cls._query_latency[session_id] = event["timestamp"]
        elif event["event"] == "on_trace_end":
            cls._query_latency[session_id] = float(( event["timestamp"] - cls._query_latency[session_id] )/ 1000)
            TraceSummaryStats._log_stats_table(session_id=session_id)
        else:
            type = event.get("span_type", "")
            if type == "agent":
                cls._init_agent_call_stats(event, session_id)
            elif type == "generation":
                cls._init_agent_llm_stats(event, session_id)
            elif type == "function":
                output = event["data"]["output"]
                if output and 'llm_calls' in output:
                    cls._init_agent_llm_stats(event, session_id)
                cls._init_agent_tool_stats(event, session_id)

    @classmethod
    def _init_agent_call_stats(cls, event_data_obj, session_id: str):
        """Update the trace statistics of the agent_call type"""
        event = event_data_obj["event"]
        metric = "agent_call"
        agent = event_data_obj["data"]["agent_name"]
        tool = ''
        if event == "on_span_start":
            cls._create_stats(metric,agent,tool,session_id)
        else:
            duration = event_data_obj.get("duration_s", -1)
            cls._update_stats(metric, agent, tool, session_id, duration)

    @classmethod
    def _init_agent_llm_stats(cls, event_data_obj, session_id: str):
        """Update the trace statistics of the agent_llm type"""
        event = event_data_obj["event"]
        metric = "agent_llm"
        agent = event_data_obj["parent_agent"]
        tool = "main"
        if event_data_obj.get("span_type") == "function":
            event = "sub_task"

        if event == "on_span_start":
            cls._create_stats(metric,agent,tool,session_id)
        elif event == "on_span_end":
            duration = event_data_obj.get("duration_s", -1)
            cls._update_stats(metric, agent, tool, session_id, duration)
        elif event == "sub_task":
            text = json.loads(event_data_obj["data"]["output"]).get("text")
            name = event_data_obj["data"]["name"]
            function_output = json.loads(text)
            for data in function_output.get("data", {}):
                meta_data = data.get("metadata",{})
                if "llm_calls" in meta_data:
                    st = meta_data.get("llm_calls",{})[0].get("start_time",0.0)
                    et = meta_data.get("llm_calls",{})[0].get("end_time",0.0)
                    tool = name + "_sub_task"
                    cls._update_stats(metric, agent, tool, session_id, float((et - st)/1000))

    @classmethod
    def _init_agent_tool_stats(cls, event_data_obj, session_id: str):
        """Update the trace statistics of the agent_tool type"""
        event = event_data_obj.get("event", "")
        agent = event_data_obj["parent_agent"]
        metric = "agent_tool"
        tool = event_data_obj["data"]["name"]
        if event == "on_span_start":
            cls._create_stats(metric,agent,tool,session_id)
        elif event == "on_span_end":
            duration = event_data_obj.get("duration_s", -1)
            cls._update_stats(metric, agent, tool, session_id,duration)

    @classmethod
    def _generate_key(cls, metric, agent, tool):
        """get unique install key"""
        return f"{metric}_{agent}_{tool}"

    @classmethod
    def _create_stats(cls, metric, agent, tool, session_id: str):
        if cls._generate_key(metric, agent, tool) in cls._order.get(session_id, []):
            return
        stats = cls(metric, agent, tool)
        cls._set_by_key(stats, session_id)

    @classmethod
    def _update_stats(cls, metric, agent, tool, session_id: str, duration_s:float) -> None:
        if cls._generate_key(metric, agent, tool) not in cls._order.get(session_id, []):
            cls._create_stats(metric, agent, tool, session_id)
        stats = cls._get_by_key(metric, agent, tool, session_id)
        stats._call_latency.append(duration_s)

    @classmethod
    def _set_by_key(cls, stats: "TraceSummaryStats", session_id: str) -> None:
        key = cls._generate_key(stats.metrics,stats.agent,stats.tool)
        if session_id not in cls._instance_map:
            cls._instance_map[session_id] = {}
        cls._instance_map[session_id][key] = stats
        if session_id not in cls._order:
            cls._order[session_id] = []
        if key not in cls._order[session_id]:
            cls._order[session_id].append(key)

    @classmethod
    def _get_by_key(cls, a: str, b: str, c: str, session_id: str) -> "TraceSummaryStats":
        key = cls._generate_key(a, b, c)
        return cls._instance_map.get(session_id).get(key)

    @classmethod
    def get_all_instances_by_creation_order(cls, session_id: str) -> List["TraceSummaryStats"]:
        alive_instances = []
        for key in cls._order.get(session_id, []):
            obj = cls._instance_map.get(session_id).get(key)
            if obj is not None:
                alive_instances.append(obj)
        return alive_instances

    @classmethod
    def cleanup_session(cls, session_id: str):
        """cleanup instances belonging to the session to avoid OOM, when _log_stats_table is called"""
        if session_id in cls._instance_map:
            del cls._instance_map[session_id]
        if session_id in cls._order:
            del cls._order[session_id]
        if session_id in cls._query_latency:
            del cls._query_latency[session_id]
        if session_id in cls._id_to_agent_name_map:
            del cls._id_to_agent_name_map[session_id]

    @classmethod
    def _log_stats_table(cls,
                        session_id: str,
                        w_metrics: int = 14,
                        w_agent: int = 36,
                        w_tool: int = 46,
                        w_latency_group: int = 42,
                        float_format: str = ".3f"):
        """print the statistical information as a log in table form."""
        import pprint
        pprint.pprint(cls._instance_map)

        stats_list = cls.get_all_instances_by_creation_order(session_id)
        header_metrics = "METRICS"
        header_agent = "AGENT"
        header_tool = "TOOL"
        header_latency = f"_call_latency count latency proportion"

        line = (
                "+" + "-" * w_metrics +
                "+" + "-" * w_agent +
                "+" + "-" * w_tool +
                "+" + "-" * w_latency_group + "+"
        )

        header_row = (
            f"|{header_metrics:<{w_metrics}}"
            f"|{header_agent:<{w_agent}}"
            f"|{header_tool:<{w_tool}}"
            f"|{header_latency:<{w_latency_group}}|"
        )
        logger.info(line)
        logger.info(header_row)
        logger.info(line)

        query_latency = cls._query_latency.get(session_id, 0.0)
        for stat in stats_list:
            lat = stat._call_latency
            count = len(lat)
            total = sum(lat) if count > 0 else 0.0
            avg = total / count if count > 0 else 0.0
            proportion = total / query_latency if query_latency > 0 else 0.0
            latency_str = f"{avg:{float_format}}       {count:5d}   {total:{float_format}}    {proportion:{float_format}}"

            row = (
                f"|{stat.metrics:<{w_metrics}}"
                f"|{stat.agent:<{w_agent}}"
                f"|{stat.tool:<{w_tool}}"
                f"|{latency_str:<{w_latency_group}}|"
            )
            logger.info(row)
        logger.info(line)
        cls.cleanup_session(session_id)



# global trace statistical event queue
_statistics_active_queue: "queue.Queue[SSEEvent]" = queue.Queue(maxsize=16384)
_sse_active_queues: Dict[str, "queue.Queue[str]"] = {}
_sse_active_queues_lock = threading.Lock()
_sse_stop_events: Dict[str, threading.Event] = {}

# global broadcast queue: Receives all events to be published
_broadcast_queue: "queue.Queue[SSEEvent]" = queue.Queue(maxsize=16384)

# background worker thread
_worker_thread: Optional[threading.Thread] = None
_shutdown_event = threading.Event()
_STOP_SIGNAL = "STOP"

def _is_terminal_event(sse_event: SSEEvent) -> bool:
    if sse_event.event_data == _STOP_SIGNAL:
        return True
    try:
        return json.loads(sse_event.event_data).get("event") == "on_trace_end"
    except (TypeError, ValueError, AttributeError):
        return False


def _put_terminal_event(
    target_queue: queue.Queue,
    item: Any,
    *,
    track_tasks: bool = False,
) -> None:
    while True:
        try:
            target_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                target_queue.get_nowait()
                if track_tasks:
                    target_queue.task_done()
            except queue.Empty:
                continue


def _deliver_to_session_queue(
    target_queue: "queue.Queue[str]", sse_event: SSEEvent
) -> None:
    if sse_event.event_data == _STOP_SIGNAL:
        stop_event = _sse_stop_events.get(sse_event.session_id)
        if stop_event is not None:
            stop_event.set()
        return
    if _is_terminal_event(sse_event):
        _put_terminal_event(target_queue, sse_event.event_data)
    else:
        target_queue.put_nowait(sse_event.event_data)

def register_session(session_id: str, max_queue_size: int = 1000):
    with _sse_active_queues_lock:
        if session_id not in _sse_active_queues:
            _sse_active_queues[session_id] = queue.Queue(maxsize=max_queue_size)
            _sse_stop_events[session_id] = threading.Event()

def unregister_session(session_id: str):
    with _sse_active_queues_lock:
        _sse_active_queues.pop(session_id, None)
        _sse_stop_events.pop(session_id, None)
    TraceSummaryStats.cleanup_session(session_id)

def send_sse_stop_signal(session_id: str):
    """Order STOP after pending broadcasts without occupying the data queue."""
    with _sse_active_queues_lock:
        if session_id not in _sse_active_queues:
            return
    _ensure_worker_started()
    _broadcast_queue.put(
        SSEEvent(
            session_id=session_id,
            trace_id=session_id,
            event_data=_STOP_SIGNAL,
            is_volatile=True,
        )
    )

def _ensure_worker_started():
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return
    with threading.Lock():
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_broadcast_worker, daemon=True)
            _worker_thread.start()


async def event_generator(session_id: str) -> AsyncGenerator[str, None]:
    """SSE 异步生成器，供 FastAPI 使用"""
    if session_id not in _sse_active_queues:
        raise ValueError(f"Session {session_id} not registered")
    local_queue = _sse_active_queues[session_id]
    stop_event = _sse_stop_events[session_id]
    task = asyncio.current_task()
    def on_cancel() -> None:
        unregister_session(session_id)
    try:
        # 注册取消回调（当客户端断开时触发）
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(lambda t: on_cancel())
        while True:
            try:
                # 尝试从队列获取数据（非阻塞）
                item = local_queue.get_nowait()
                if item == _STOP_SIGNAL:
                    print(f"⏹️ SSE stream stopped for {session_id}")
                    break
                yield f"data: {json.dumps(item)}\n\n"
            except queue.Empty:
                if stop_event.is_set():
                    print(f"⏹️ SSE stream stopped for {session_id}")
                    break
                # 等待新消息，允许被取消
                try:
                    await asyncio.wait_for(asyncio.sleep(0.1), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
            except Exception as e:
                logger.error("event_generator: ",e)
                raise
    finally:
        unregister_session(session_id)

def _broadcast_worker():
    """background thread: uniformly handles event broadcasts"""
    while not _shutdown_event.is_set():
        try:
            sse_event: SSEEvent = _broadcast_queue.get(timeout=0.5)
            session_id = sse_event.session_id
            log_payload = {
                "msg": "Broadcast queue overflow! Dropping streaming event.",
                "trace_id": getattr(sse_event, "trace_id", "unknown"),
                "session_id": getattr(sse_event, "session_id", "unknown"),
                "is_volatile": getattr(sse_event, "is_volatile", False),
                "current_queue_size": _broadcast_queue.qsize(),
            }
            # SSE for span events and token streams
            with _sse_active_queues_lock:
                q = _sse_active_queues.get(sse_event.session_id)
                if q is not None:
                    try:
                        _deliver_to_session_queue(q, sse_event)
                    except queue.Full:
                        logger.error(f"Queue Full Detail: {log_payload}")


            # SSE for on_status_update span event
            if not getattr(sse_event, "is_volatile", False):
                try:
                    # update, use trace_id
                    event_data_obj = json.loads(sse_event.event_data)
                    TraceSummaryStats.parsing_session_stats(event_data_obj, sse_event.trace_id)

                    status_list = [p.to_dict() for p in TraceSummaryStats.get_all_instances_by_creation_order(sse_event.trace_id)]
                    update_msg = json.dumps({"event":"on_status_update", "data": status_list, "trace_id": sse_event.trace_id}, ensure_ascii=False)

                    # stats sse
                    with _sse_active_queues_lock:
                        q = _sse_active_queues.get(session_id)
                        if q is not None:
                            try:
                                q.put_nowait(update_msg)
                            #     print(f"SSEBatchTracingProcessor output : {sse_event.event_data}")
                            except queue.Full:
                                logger.error(f"Queue Full Detail: {log_payload}")
                except Exception as stats_err:
                    logger.error(f"Stats Error: {stats_err}")

            _broadcast_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[SSE Worker] Unexpected error: {e}")

    # clean up remaining events (optional)
    while not _broadcast_queue.empty():
        try:
            _broadcast_queue.get_nowait()
            _broadcast_queue.task_done()
        except queue.Empty:
            break


class SSEBatchTracingProcessor(TracingProcessor):
    """
    To batch forward the trace/span events of the Agent to the broadcast queue
    """

    def __init__(
            self,
            default_session_id: str = "default",
            max_queue_size: int = 8192,
            max_batch_publish_delay: float = 1.0,
    ):
        self.default_session_id = default_session_id
        self.max_queue_size = max_queue_size
        self.max_batch_publish_delay = max_batch_publish_delay
        self._next_flush_time = time.time() + max_batch_publish_delay
        self._shutdown_event = threading.Event()
        self._started = False
        self._start_lock = threading.Lock()
        self._trace_to_session: Dict[str, str] = {}
        self._binding_lock = threading.Lock()

    def bind_trace_to_session(self, trace_id: str, session_id: str) -> None:
        """Explicitly bind a trace to the specified session_id."""
        if not trace_id or not session_id:
            raise ValueError("trace_id and session_id must not be empty")

        with self._binding_lock:
            self._trace_to_session[trace_id] = session_id


    def _ensure_started(self):
        """start the background broadcast thread"""
        if self._started:
            return
        with self._start_lock:
            if not self._started:
                _ensure_worker_started()
                self._started = True

    def _extract_event_from_span(self, span: Span[Any]) -> Dict[str, Any]:
        span_data = span.span_data
        span_type = span_data.type
        # common foundation
        content = {
            "span_id": span.span_id,
            "parent_id": span.parent_id,
            "trace_id": span.trace_id,
            "span_type": span_type,
        }
        if span.started_at:
            content["start_time"] = int(datetime.fromisoformat(span.started_at).timestamp() * 1000)
        if span.ended_at:
            content["end_time"] = int(datetime.fromisoformat(span.ended_at).timestamp() * 1000)
            duration_s = int((content["end_time"] - content["start_time"]))
            content["duration_s"] = float(duration_s / 1000)
        # error process
        if hasattr(span, "error") and span.error:
            content["error"] = str(span.error)
        # get span data according to span_type
        if span_type == "mcp_tools": # span data is MCPListToolsSpanData
            content.update({
                "data": {
                    "server": span_data.server,
                    "available_tools": span_data.result or [],
                }
            })
        elif span_type == "function": # FunctionSpanData
            content.update({
                "data": {
                    "name": span_data.name,
                    "input": span_data.input,
                    "output": str(span_data.output) if span_data.output is not None else None,
                    "mcp_data": span_data.mcp_data,
                }
            })
        elif span_type == "generation":
            # GenerationSpanData
            content.update({
                "data": {
                    # "model": span_data.model,
                    "model_config": span_data.model_config,
                    "usage": span_data.usage,
                    "messages": span_data.input,
                    "response": span_data.output,
                }
            })
        elif span_type == "agent": # AgentSpanData
            content.update({
                "data": {
                    "agent_name": span_data.name,
                    "handoffs": span_data.handoffs or [],
                    "tools": span_data.tools or [],
                    "output_type": span_data.output_type,
                }
            })
        elif span_type == "custom":
            if span_data.name == "llm_call_metric":
                content.update({
                    "data":{
                        "name": span_data.name,
                        "model": span_data.data.get("model"),
                        "model_config": span_data.data.get("model_config"),
                        "metrics":  span_data.data.get('metrics'),
                    }
                })
            if span_data.name == "generation_resum_compress":
                content.update({
                    "data": {
                        **span_data.data,
                        "name": span_data.name
                    }
                })
            if span_data.name == "e2e_call_metric":
                content.update({
                    "data": {
                        "name": span_data.name,
                        "time": span_data.data.get('metrics')
                    }
                })

        return content

    def on_trace_start(self, trace: Trace) -> None:
        session_id = self.get_session_id(trace.trace_id)
        self._ensure_started()

        event = {
            "event": "on_trace_start",
            "trace_id": trace.trace_id,
            "workflow": trace.name,
            "timestamp": int(time.time() * 1000),
            "session_id": session_id
        }
        try:
            _broadcast_queue.put_nowait(SSEEvent(session_id=session_id, trace_id=trace.trace_id, event_data=json.dumps(event)))
        except queue.Full:
            pass

    def on_trace_end(self, trace: Trace) -> None:
        session_id = self.get_session_id(trace.trace_id)
        event = {
            "event": "on_trace_end",
            "trace_id": trace.trace_id,
            "workflow": trace.name,
            "timestamp": int(time.time() * 1000),
            "session_id": session_id
        }
        _put_terminal_event(
            _broadcast_queue,
            SSEEvent(
                session_id=session_id,
                trace_id=trace.trace_id,
                event_data=json.dumps(event),
            ),
            track_tasks=True,
        )

    def on_span_start(self, span: Span[Any]) -> None:
        try:
            session_id = self.get_session_id(span.trace_id)
            span_type = span.span_data.type
            event_data = self._extract_event_from_span(span)
            event_data.update({
                "event": "on_span_start",
                "session_id": session_id
            })
            try:
                payload = json.dumps(event_data, ensure_ascii=False, default=str)
                _broadcast_queue.put_nowait(
                    SSEEvent(session_id=session_id, trace_id=span.trace_id, event_data=payload))
            except queue.Full:
                pass
        except Exception as e:
            raise
    def on_span_end(self, span: Span[Any]) -> None:
        try:

            session_id = self.get_session_id(span.trace_id)
            span_type = span.span_data.type
            event_data = self._extract_event_from_span(span)

            event_data.update({
                "event": "on_span_end",
                "session_id": session_id
            })

            try:
                payload = json.dumps(event_data, ensure_ascii=False, default=str)
                _broadcast_queue.put_nowait(
                    SSEEvent(session_id=session_id, trace_id=span.trace_id, event_data=payload))
            except queue.Full:
                pass
        except Exception as e:
            raise

    def on_span_stream(self, span: Span, chunk_type: str, delta: Any):
        session_id = self.get_session_id(span.trace_id)

        event_data = {
            "span_id": span.span_id,
            "parent_id": span.parent_id,
            "trace_id": span.trace_id,
            "span_type": "trace.span.chunk",
            "chunk_type":chunk_type,
            "delta": delta,
        }
        try:
            payload = json.dumps(event_data, ensure_ascii=False, default=str)
            _broadcast_queue.put_nowait(
                SSEEvent(session_id=session_id, trace_id=span.trace_id, event_data=payload, is_volatile=True))
        except queue.Full:
            pass

    def get_session_id(self, trace_id) -> str:
        if self._trace_to_session.__len__() == 0:
            return trace_id
        if self._trace_to_session.get(trace_id) is None:
            return trace_id
        return self._trace_to_session[trace_id]

    def shutdown(self) -> None:
        self._shutdown_event.set()

    def force_flush(self):
        pass

SSEProfessor = SSEBatchTracingProcessor()
add_trace_processor(SSEProfessor)


