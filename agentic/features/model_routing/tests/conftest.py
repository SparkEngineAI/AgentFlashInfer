"""Shared fixtures for model_routing tests."""

from unittest.mock import MagicMock

import pytest

from features.model_routing.config import ModelRoutingConfig


# ── Config fixtures ──


@pytest.fixture
def default_config() -> ModelRoutingConfig:
    return ModelRoutingConfig()


@pytest.fixture
def enabled_config() -> ModelRoutingConfig:
    return ModelRoutingConfig(
        enabled=True,
        small_model_url="http://7B:7000/v1",
        large_model_url="http://38B:7000/v1",
        small_model_name="pangu_small",
        large_model_name="pangu_large",
        small_model_api_key="sk-small",
        large_model_api_key="sk-large",
    )


# ── Mock Model / ModelSettings ──


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.model = "pangu_small"
    return model


@pytest.fixture
def mock_model_settings() -> MagicMock:
    settings = MagicMock()
    settings.temperature = 0.6
    settings.max_tokens = 8192
    settings.timeout = 180
    return settings


# ── Conversation history fixtures ──
# should_escalate checks conversation_history[-2] for trigger_tool_name,
# so histories must end with [..., assistant_with_trigger, user].


@pytest.fixture
def short_history() -> list[dict]:
    """3-item history — below default progress_check_min_iter=3."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]


@pytest.fixture
def long_history_with_think() -> list[dict]:
    """9-item history ending with [assistant_with_think, user] — triggers progress check."""
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(4):
        history.append({"role": "user", "content": f"Query {i + 1}"})
        history.append({"role": "assistant", "content": f'Called "think" — step {i + 1}'})
    history.append({"role": "user", "content": "Query 5"})
    return history


@pytest.fixture
def long_history_without_think() -> list[dict]:
    """9-item history ending with [assistant_without_think, user] — does NOT trigger progress check."""
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(4):
        history.append({"role": "user", "content": f"Query {i + 1}"})
        history.append({"role": "assistant", "content": f"Regular response {i + 1}"})
    history.append({"role": "user", "content": "Query 5"})
    return history