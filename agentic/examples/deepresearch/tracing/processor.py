import asyncio
import json
import os
import queue
import threading
import time
from typing import Dict, Optional, AsyncGenerator
import logging

from examples.deepresearch.config.deepresearch_config import DR_WORKSPACE_PATH

os.makedirs(DR_WORKSPACE_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{DR_WORKSPACE_PATH}/DeepResearchLogs.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

# Per-session message queues
_sse_active_queues: Dict[str, "queue.Queue[str]"] = {}
_sse_active_queues_lock = threading.Lock()

_STOP_SIGNAL = "STOP"


def register_session(session_id: str, max_queue_size: int = 1000):
    with _sse_active_queues_lock:
        if session_id not in _sse_active_queues:
            _sse_active_queues[session_id] = queue.Queue(maxsize=max_queue_size)


def unregister_session(session_id: str):
    with _sse_active_queues_lock:
        _sse_active_queues.pop(session_id, None)


def send_sse_stop_signal(session_id: str):
    with _sse_active_queues_lock:
        if session_id in _sse_active_queues:
            q = _sse_active_queues.get(session_id)
            try:
                q.put_nowait(_STOP_SIGNAL)
            except queue.Full:
                pass


def emit_react_event(session_id: str, event_type: str, event_data: dict):
    """Push an event directly to the session's SSE queue."""
    with _sse_active_queues_lock:
        q = _sse_active_queues.get(session_id)
        if q is not None:
            event = {
                "event": event_type,
                "data": event_data,
                "timestamp": int(time.time() * 1000),
                "session_id": session_id
            }
            try:
                q.put_nowait(json.dumps(event, ensure_ascii=False))
            except queue.Full:
                logger.warning(f"SSE queue full for session {session_id}, dropping event")


async def event_generator(session_id: str) -> AsyncGenerator[str, None]:
    """SSE async generator for FastAPI EventSourceResponse."""
    if session_id not in _sse_active_queues:
        raise ValueError(f"Session {session_id} not registered")
    local_queue = _sse_active_queues[session_id]
    task = asyncio.current_task()

    def on_cancel():
        unregister_session(session_id)

    try:
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(lambda t: on_cancel())
        while True:
            try:
                item = local_queue.get_nowait()
                if item == _STOP_SIGNAL:
                    break
                yield f"data: {item}\n\n"
            except queue.Empty:
                try:
                    await asyncio.wait_for(asyncio.sleep(0.1), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
            except Exception as e:
                logger.error(f"event_generator error: {e}")
                raise
    finally:
        unregister_session(session_id)