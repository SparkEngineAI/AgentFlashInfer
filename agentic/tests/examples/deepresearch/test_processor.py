"""Tests for SSE tracing processor."""

import asyncio
import json
import queue
from unittest.mock import patch

import pytest

from examples.deepresearch.tracing.processor import (
    register_session,
    unregister_session,
    send_sse_stop_signal,
    emit_react_event,
    event_generator,
    _sse_active_queues,
    _sse_active_queues_lock,
    _STOP_SIGNAL,
)


@pytest.fixture(autouse=True)
def clean_queues():
    """Clear all queues before each test."""
    with _sse_active_queues_lock:
        _sse_active_queues.clear()
    yield
    with _sse_active_queues_lock:
        _sse_active_queues.clear()


class TestRegisterSession:
    def test_creates_queue(self):
        register_session("sess_1")
        with _sse_active_queues_lock:
            assert "sess_1" in _sse_active_queues

    def test_idempotent(self):
        register_session("sess_1")
        q1 = _sse_active_queues["sess_1"]
        register_session("sess_1")
        q2 = _sse_active_queues["sess_1"]
        assert q1 is q2  # same queue, not overwritten

    def test_custom_queue_size(self):
        register_session("sess_1", max_queue_size=10)
        q = _sse_active_queues["sess_1"]
        assert q.maxsize == 10


class TestUnregisterSession:
    def test_removes_queue(self):
        register_session("sess_1")
        unregister_session("sess_1")
        with _sse_active_queues_lock:
            assert "sess_1" not in _sse_active_queues

    def test_nonexistent_session_no_error(self):
        unregister_session("nonexistent")  # should not raise


class TestSendSseStopSignal:
    def test_puts_stop_in_queue(self):
        register_session("sess_1")
        send_sse_stop_signal("sess_1")
        q = _sse_active_queues["sess_1"]
        assert q.get_nowait() == _STOP_SIGNAL

    def test_full_queue_silent_drop(self):
        register_session("sess_1", max_queue_size=1)
        q = _sse_active_queues["sess_1"]
        q.put("filler")  # fill the queue
        send_sse_stop_signal("sess_1")  # should not raise

    def test_unregistered_session_no_error(self):
        send_sse_stop_signal("nonexistent")  # should not raise


class TestEmitReactEvent:
    def test_puts_json_in_queue(self):
        register_session("sess_1")
        emit_react_event("sess_1", "on_react_step", {"type": "thinking", "round": 1})
        q = _sse_active_queues["sess_1"]
        data = json.loads(q.get_nowait())
        assert data["event"] == "on_react_step"
        assert data["data"]["type"] == "thinking"
        assert "timestamp" in data

    def test_includes_session_id(self):
        register_session("sess_1")
        emit_react_event("sess_1", "on_react_step", {"type": "answer"})
        data = json.loads(_sse_active_queues["sess_1"].get_nowait())
        assert data["session_id"] == "sess_1"

    def test_unregistered_session_no_error(self):
        emit_react_event("nonexistent", "on_react_step", {"type": "thinking"})

    def test_full_queue_silent_drop(self):
        register_session("sess_1", max_queue_size=1)
        _sse_active_queues["sess_1"].put("filler")
        emit_react_event("sess_1", "on_react_step", {"type": "thinking"})


@pytest.mark.asyncio
class TestEventGenerator:
    async def test_yields_events(self):
        register_session("sess_1")
        emit_react_event("sess_1", "on_react_step", {"type": "thinking", "round": 1})
        send_sse_stop_signal("sess_1")

        events = []
        async for event in event_generator("sess_1"):
            events.append(event)

        assert len(events) == 1
        assert events[0].startswith("data: ")
        data = json.loads(events[0][6:].strip())
        assert data["data"]["type"] == "thinking"

    async def test_unregistered_session_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            async for _ in event_generator("nonexistent"):
                pass
