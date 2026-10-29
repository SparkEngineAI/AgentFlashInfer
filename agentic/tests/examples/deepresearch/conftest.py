"""Shared fixtures for DeepResearch tests."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ── Environment fixtures ──


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Set minimal env vars so config imports don't fail."""
    env = {
        "VLLM_MODEL_NAME": "test-model",
        "VLLM_MODEL_PATH": "/tmp/test-model",
        "VLLM_PORTS": "7001,7002",
        "SERPER_KEY": "test-serper-key",
        "JINA_API_KEYS": "test-jina-key",
        "API_KEY": "test-api-key",
        "API_BASE": "https://api.test.com/v1",
        "SUMMARY_MODEL_NAME": "test-summary-model",
        "SANDBOX_FUSION_ENDPOINT": "http://sandbox:8080",
        "DR_WORKSPACE_PATH": "/tmp/test_dr_workspace",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)


# ── Sample data fixtures ──

# XML tokens used by the agent for tool calls and observations
THINK_START = "θα"
THINK_END = "κψ"
OBS_START = "παρα"
OBS_END = "κρψ"


@pytest.fixture
def sample_tool_call_search():
    """LLM output containing a search tool call."""
    return (
        f"{THINK_START}I need to search for this{THINK_END}\n"
        '\n{"name": "search", "arguments": {"query": ["quantum computing"]}}\n'
    )


@pytest.fixture
def sample_tool_call_python():
    """LLM output containing a python tool call."""
    return (
        f"{THINK_START}Let me run some code{THINK_END}\n"
        '\n{"name": "python", "arguments": {"code": ""}}\n'
        "<code>print('hello')\n</code>\n"
    )


@pytest.fixture
def sample_answer():
    """LLM output containing a final answer."""
    return (
        f"{THINK_START}Based on my research{THINK_END}\n"
        "<answer>The answer is 42</answer>"
    )


@pytest.fixture
def sample_serp_response():
    """Sample Serper API search response."""
    return {
        "organic": [
            {
                "title": "Quantum Computing 101",
                "link": "https://example.com/qc",
                "snippet": "An introduction to quantum computing.",
                "date": "2024-01-15",
                "source": "example.com",
            },
            {
                "title": "Advanced Quantum Algorithms",
                "link": "https://example.com/aqa",
                "snippet": "Deep dive into quantum algorithms.",
            },
        ]
    }


@pytest.fixture
def sample_scholar_response():
    """Sample Serper API scholar response."""
    return {
        "organic": [
            {
                "title": "Quantum Error Correction Survey",
                "link": "https://scholar.example.com/qec",
                "snippet": "A comprehensive survey of QEC methods.",
                "year": 2023,
                "publicationInfo": "Journal of Quantum Info",
                "citedBy": 42,
                "pdfUrl": "https://scholar.example.com/qec.pdf",
            },
        ]
    }


@pytest.fixture
def mock_tokenizer():
    """Mock AutoTokenizer that supports apply_chat_template and __call__."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<mocked prompt>"
    mock_input_ids = MagicMock()
    mock_input_ids.__len__ = MagicMock(return_value=100)
    tokenizer.return_value = {"input_ids": [mock_input_ids]}
    return tokenizer
