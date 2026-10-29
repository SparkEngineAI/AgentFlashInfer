import os
import subprocess
import sys
from pathlib import Path


AGENTIC_DIR = Path(__file__).resolve().parents[1]


def test_config_imports_without_proxy_and_uses_runtime_endpoints():
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_BASE_URL": "http://127.0.0.1:8103/v1",
            "OPENAI_API_KEY": "EMPTY",
            "EXAMPLE_MODEL_NAME": "pangu_auto",
            "SEARCH_URL": "https://search.example.test",
            "SEARCH_API_KEY": "search-key",
            "URL_CRAWLER_API_KEYS": "crawler-key",
            "SEARCH_COMPRESS_URL": "http://127.0.0.1:8103/v1/chat/completions",
            "SEARCH_COMPRESS_MODEL_NAME": "pangu_auto",
        }
    )
    env.pop("PROXY_CACHE", None)

    code = """
from config import config
assert config.PROXY_CACHE is None
assert config.SEARCH_COMPRESS_URL == 'http://127.0.0.1:8103/v1/chat/completions'
assert config.SEARCH_COMPRESS_MODEL_NAME == 'pangu_auto'
"""
    result = subprocess.run(
        [os.environ.get("AGENT_TEST_PYTHON", sys.executable), "-c", code],
        cwd=AGENTIC_DIR,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_routing_mode_requires_only_the_two_model_endpoint_pairs(tmp_path):
    """Routing mode must not require legacy default-model environment keys."""
    env = os.environ.copy()
    env.update(
        {
            "MODEL_ROUTING_SWITCH": "true",
            "SMALL_MODEL_URL": "http://127.0.0.1:8103/v1",
            "SMALL_MODEL_NAME": "pangu_small",
            "LARGE_MODEL_URL": "http://127.0.0.1:8104/v1",
            "LARGE_MODEL_NAME": "pangu_large",
            "SEARCH_URL": "https://search.example.test",
            "SEARCH_API_KEY": "search-key",
            "URL_CRAWLER_API_KEYS": "crawler-key",
        }
    )
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "EXAMPLE_MODEL_NAME",
        "ROUTING_MODEL_NAME",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = f"{AGENTIC_DIR}:{env.get('PYTHONPATH', '')}"

    code = """
from config import config
assert config.DEFAULT_MODEL_URL == 'http://127.0.0.1:8103/v1'
assert config.DEFAULT_MODEL_NAME == 'pangu_small'
assert config.SMALL_MODEL_NAME == 'pangu_small'
assert config.LARGE_MODEL_NAME == 'pangu_large'
assert config.SEARCH_COMPRESS_URL == 'http://127.0.0.1:8103/v1/chat/completions'
assert config.SEARCH_COMPRESS_MODEL_NAME == 'pangu_small'
from agent_infer.agent_specs import base_agent
from agent_infer.mcp_factory.tools import tool_register
assert base_agent.BASE_URL == 'http://127.0.0.1:8103/v1'
assert base_agent.MODEL_NAME == 'pangu_small'
assert tool_register.BASE_URL == 'http://127.0.0.1:8103/v1'
assert tool_register.MODEL_NAME == 'pangu_small'
"""
    result = subprocess.run(
        [os.environ.get("AGENT_TEST_PYTHON", sys.executable), "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
