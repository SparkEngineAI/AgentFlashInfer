"""Tests for Session and SessionManager."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from examples.deepresearch.server.session import Session, SessionManager


@pytest.fixture
def mock_stop_signal():
    with patch("examples.deepresearch.server.session.send_sse_stop_signal") as m:
        yield m


@pytest.fixture
def session(mock_stop_signal):
    return Session(session_id="sess_test", user_token="user1", concurrency=1)


@pytest.fixture
def manager():
    return SessionManager()


# ── Session ──


class TestSessionCreateTask:
    @pytest.mark.asyncio
    async def test_create_task_stores_task(self, session):
        async def dummy_work():
            return "done"

        session.create_task(dummy_work, "request", 0, "task_1")
        assert "task_1" in session.tasks
        assert session.tasks["task_1"] is not None

    @pytest.mark.asyncio
    async def test_create_task_stores_meta(self, session):
        async def dummy_work():
            return "done"

        session.create_task(dummy_work, "request", 0, "task_1")
        meta = session.task_meta["task_1"]
        assert meta["session_id"] == "sess_test"
        assert meta["user_token"] == "user1"
        assert meta["index"] == 0
        assert "start_time" in meta


class TestSessionEndTask:
    @pytest.mark.asyncio
    async def test_end_task_decrements_running(self, session):
        async def dummy_work():
            return "done"

        session.create_task(dummy_work, "request", 0, "task_1")
        initial = session.running_tasks
        session.end_task("task_1")
        assert session.running_tasks < initial

    @pytest.mark.asyncio
    async def test_end_task_sets_end_time(self, session):
        async def dummy_work():
            return "done"

        session.create_task(dummy_work, "request", 0, "task_1")
        session.end_task("task_1")
        assert "end_time" in session.task_meta.get("task_1", {})

    @pytest.mark.asyncio
    async def test_end_task_sends_stop_signal_when_no_running(self, session, mock_stop_signal):
        async def dummy_work():
            return "done"

        session.create_task(dummy_work, "request", 0, "task_1")
        session.end_task("task_1")
        # When running_tasks hits 0, stop signal should be sent
        mock_stop_signal.assert_called_with("sess_test")


class TestSessionStopSession:
    def test_stop_session_sets_completed_at(self, session):
        session.stop_session()
        assert session.completed_at is not None

    @pytest.mark.asyncio
    async def test_stop_session_sends_stop_signal(self, session, mock_stop_signal):
        async def dummy_work():
            return "done"

        session.create_task(dummy_work, "request", 0, "task_1")
        session.stop_session()
        mock_stop_signal.assert_called()


class TestSessionStopTask:
    @pytest.mark.asyncio
    async def test_stop_task_removes_from_dicts(self, session):
        async def dummy_work():
            return "done"

        session.create_task(dummy_work, "request", 0, "task_1")
        session._stop_task("task_1")
        assert "task_1" not in session.tasks
        assert "task_1" not in session.task_meta


# ── SessionManager ──


class TestSessionManagerCreateSession:
    def test_creates_session_with_id(self, manager):
        session = manager.create_session("user1")
        assert session.session_id.startswith("sess_")
        assert session.user_token == "user1"

    def test_registers_user_mapping(self, manager):
        session = manager.create_session("user1")
        assert session.session_id in manager.user_sessions["user1"]

    def test_stores_session(self, manager):
        session = manager.create_session("user1")
        assert manager.get_session_by_session_id(session.session_id) is session


class TestSessionManagerStopSession:
    def test_removes_session(self, manager, mock_stop_signal):
        session = manager.create_session("user1")
        manager.stop_session(session.session_id)
        assert manager.get_session_by_session_id(session.session_id) is None


class TestSessionManagerCleanup:
    def test_cleanup_removes_all_user_sessions(self, manager, mock_stop_signal):
        s1 = manager.create_session("user1")
        s2 = manager.create_session("user1")
        manager.cleanup_user_sessions("user1")
        assert manager.get_session_by_session_id(s1.session_id) is None
        assert manager.get_session_by_session_id(s2.session_id) is None
        assert "user1" not in manager.user_sessions

    def test_cleanup_nonexistent_user(self, manager):
        manager.cleanup_user_sessions("nobody")  # should not raise


class TestSessionManagerStats:
    def test_stats_empty(self, manager):
        stats = manager.get_system_stats()
        assert stats["total_running_tasks"] == 0
        assert stats["warning"] is False

    def test_stats_with_sessions(self, manager):
        manager.create_session("user1", concurrency=2)
        stats = manager.get_system_stats()
        assert stats["total_running_tasks"] == 2
        assert stats["tasks_per_user"]["user1"] == 2
