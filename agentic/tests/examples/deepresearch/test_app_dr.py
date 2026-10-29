"""Integration tests for FastAPI endpoints."""

import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from examples.deepresearch.server.app_dr import app_dr
from examples.deepresearch.server.session import SessionManager


@pytest.fixture
def fresh_manager():
    return SessionManager()


@pytest.fixture(autouse=True)
def patch_session_manager(fresh_manager):
    with patch("examples.deepresearch.server.app_dr.sessionManager", fresh_manager):
        yield fresh_manager


@pytest.fixture(autouse=True)
def patch_agent():
    mock_agent = MagicMock()
    mock_agent.async_run = AsyncMock(return_value={
        "prediction": "test answer",
        "termination": "answer",
    })
    with patch("examples.deepresearch.server.app_dr.agent", mock_agent):
        yield mock_agent


@pytest.fixture(autouse=True)
def patch_sse():
    with patch("examples.deepresearch.server.app_dr.register_session"), \
         patch("examples.deepresearch.server.app_dr.emit_react_event"), \
         patch("examples.deepresearch.server.app_dr.unregister_session"):
        yield


@pytest.fixture
def client():
    return TestClient(app_dr)


class TestSubmitEndpoint:
    def test_submit_returns_session_id(self, client):
        response = client.post("/api/submit", json={
            "user_input": "test question",
            "user_token": "user1",
        })
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "task_id" in data
        assert data["status"] == "started"

    def test_submit_with_empty_input(self, client):
        response = client.post("/api/submit", json={
            "user_input": "",
            "user_token": "user1",
        })
        assert response.status_code == 200
        data = response.json()
        assert "error" in data


class TestStatsEndpoint:
    def test_stats_returns_structure(self, client):
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_running_tasks" in data
        assert "tasks_per_user" in data
        assert "warning" in data


class TestStopEndpoint:
    def test_stop_nonexistent_session(self, client):
        response = client.get("/api/stop/nonexistent")
        assert response.status_code == 200

    def test_stop_existing_session(self, client, patch_session_manager):
        with patch("examples.deepresearch.server.session.send_sse_stop_signal"):
            session = patch_session_manager.create_session("user1")
        response = client.get(f"/api/stop/{session.session_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "stopped"


class TestHistoryEndpoint:
    def test_empty_history(self, client):
        with patch("examples.deepresearch.server.app_dr.CONVERSATION_HISTORY", []):
            response = client.get("/history")
        assert response.status_code == 200
        assert response.json() == []

    def test_history_with_task_id_filter(self, client):
        response = client.get("/history?task_id=nonexistent")
        assert response.status_code == 200
        assert response.json() == []
