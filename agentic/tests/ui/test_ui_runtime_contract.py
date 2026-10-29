import asyncio
from pathlib import Path
import json
import queue
import unittest

import config.config as config
from examples.deepdiver_v2.tracing import processor as trace_processor


AGENTIC_ROOT = Path(__file__).resolve().parents[2]


class UiRuntimeContractTest(unittest.TestCase):
    def test_backend_exposes_ui_tool_predict_default(self):
        self.assertIsInstance(config.tool_predict_switch, bool)

    def test_sse_uses_ui_proxy_and_backend_alias(self):
        hook = (
            AGENTIC_ROOT
            / "agent_infer/ui/ui/src/hook/useSSEStreamByTrace.js"
        ).read_text()
        backend = (
            AGENTIC_ROOT
            / "examples/deepdiver_v2/server/app_ddv2.py"
        ).read_text()

        self.assertNotIn("http://localhost:8289/tracestream", hook)
        self.assertIn("`/api/tracestream/${sessionId}`", hook)
        self.assertIn('@app_ddv2.get("/api/tracestream/{session_id}")', backend)

    def test_sse_trace_processor_is_registered_and_broadcasts_spans(self):
        processor = (
            AGENTIC_ROOT
            / "examples/deepdiver_v2/tracing/processor.py"
        ).read_text()
        active_broadcasts = [
            line
            for line in processor.splitlines()
            if line.lstrip().startswith("_broadcast_queue.put_nowait(")
        ]

        self.assertIn("\nadd_trace_processor(SSEProfessor)\n", processor)
        self.assertGreaterEqual(len(active_broadcasts), 4)
        self.assertIn("_put_terminal_event(", processor)
        self.assertIn("if item == _STOP_SIGNAL:", processor)

    def test_trace_end_replaces_old_data_when_session_queue_is_full(self):
        session_id = "ui-terminal-backpressure-test"
        trace_processor.register_session(session_id, max_queue_size=1)
        try:
            session_queue = trace_processor._sse_active_queues[session_id]
            session_queue.put_nowait("old")
            trace_end = trace_processor.SSEEvent(
                session_id=session_id,
                trace_id=session_id,
                event_data='{"event": "on_trace_end"}',
            )
            trace_processor._deliver_to_session_queue(session_queue, trace_end)
            delivered = json.loads(session_queue.get_nowait())
            self.assertEqual(delivered["event"], "on_trace_end")
        finally:
            trace_processor.unregister_session(session_id)


class UiSseBackpressureTest(unittest.IsolatedAsyncioTestCase):
    async def test_trace_end_is_delivered_before_stop_when_queue_is_full(self):
        session_id = "ui-trace-end-before-stop-test"
        trace_processor.register_session(session_id, max_queue_size=1)
        try:
            session_queue = trace_processor._sse_active_queues[session_id]
            session_queue.put_nowait("old")
            trace_processor._deliver_to_session_queue(
                session_queue,
                trace_processor.SSEEvent(
                    session_id=session_id,
                    trace_id=session_id,
                    event_data='{"event": "on_trace_end"}',
                    is_volatile=True,
                ),
            )
            trace_processor._deliver_to_session_queue(
                session_queue,
                trace_processor.SSEEvent(
                    session_id=session_id,
                    trace_id=session_id,
                    event_data=trace_processor._STOP_SIGNAL,
                    is_volatile=True,
                ),
            )

            async def collect_events():
                return [
                    event
                    async for event in trace_processor.event_generator(session_id)
                ]

            events = await asyncio.wait_for(collect_events(), timeout=0.5)
            self.assertEqual(len(events), 1)
            delivered = json.loads(json.loads(events[0][len("data: ") :]))
            self.assertEqual(delivered["event"], "on_trace_end")
        finally:
            trace_processor.unregister_session(session_id)

    async def test_global_trace_end_is_forwarded_before_stop(self):
        session_id = "ui-global-trace-end-before-stop-test"
        original_broadcast_queue = trace_processor._broadcast_queue
        test_broadcast_queue = queue.Queue(maxsize=1)
        trace_processor._broadcast_queue = test_broadcast_queue
        trace_processor.register_session(session_id, max_queue_size=1)
        try:
            trace_processor._broadcast_queue.put_nowait(
                trace_processor.SSEEvent(
                    session_id=session_id,
                    trace_id=session_id,
                    event_data='{"event": "on_trace_end"}',
                    is_volatile=True,
                )
            )
            trace_processor.send_sse_stop_signal(session_id)

            async def collect_events():
                return [
                    event
                    async for event in trace_processor.event_generator(session_id)
                ]

            events = await asyncio.wait_for(collect_events(), timeout=2.0)
            self.assertEqual(len(events), 1)
            delivered = json.loads(json.loads(events[0][len("data: ") :]))
            self.assertEqual(delivered["event"], "on_trace_end")
        finally:
            trace_processor._broadcast_queue = original_broadcast_queue
            while not test_broadcast_queue.empty():
                test_broadcast_queue.get_nowait()
                test_broadcast_queue.task_done()
            trace_processor.unregister_session(session_id)


if __name__ == "__main__":
    unittest.main()
