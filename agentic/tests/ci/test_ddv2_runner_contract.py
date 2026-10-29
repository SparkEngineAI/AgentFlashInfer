"""Regression checks for DDv2 runner integration contracts."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "deepdiver_v2"
    / "deepdiver_v2_agent_runner.py"
)


def test_function_tool_conversion_is_qualified_through_ddv2_converter() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))

    unqualified_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "function_tool_to_openai_tool"
    ]

    assert not unqualified_calls, (
        "DDv2 runner must call "
        "Ddv2ChatConverter.function_tool_to_openai_tool(...)"
    )


def test_compression_receives_history_including_current_step_items() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    compression_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "on_turn_end"
    ]

    assert len(compression_calls) == 1
    generated_items_kw = next(
        keyword
        for keyword in compression_calls[0].keywords
        if keyword.arg == "generated_items"
    )
    value = generated_items_kw.value
    assert (
        isinstance(value, ast.Attribute)
        and value.attr == "generated_items"
        and isinstance(value.value, ast.Name)
        and value.value.id == "tool_result"
    ), "compression must include the current tool call/output in its boundary"


def test_early_stop_guidance_is_persisted_in_current_step_result() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    injected_appends = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "injected_item"
    ]

    assert len(injected_appends) == 1
    target = injected_appends[0].func.value
    assert (
        isinstance(target, ast.Attribute)
        and target.attr == "new_step_items"
        and isinstance(target.value, ast.Name)
        and target.value.id == "tool_result"
    ), "early-stop guidance must be visible in the next turn's generated history"


def test_model_router_is_run_local_instead_of_class_shared() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Ddv2AgentRunner"
    )

    class_router_state = [
        node
        for node in runner_class.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            any(
                isinstance(target, ast.Name) and target.id == "_router"
                for target in node.targets
            )
            if isinstance(node, ast.Assign)
            else isinstance(node.target, ast.Name) and node.target.id == "_router"
        )
    ]

    assert not class_router_state, "parallel DDv2 runs must not share router state"
    assert 'context_wrapper.context["model_router"] = ModelRouter(' in source


def test_direct_answer_is_not_treated_as_a_function_tool_call() -> None:
    required_env = {
        "OPENAI_BASE_URL": "http://127.0.0.1:8103/v1",
        "OPENAI_API_KEY": "EMPTY",
        "EXAMPLE_MODEL_NAME": "pangu_auto",
        "SEARCH_URL": "https://search.example.test",
        "SEARCH_API_KEY": "search-key",
        "URL_CRAWLER_API_KEYS": "crawler-key",
    }
    with patch.dict(os.environ, required_env, clear=False):
        from examples.deepdiver_v2.deepdiver_v2_agent_runner import Ddv2AgentRunner

    prior_tool_call = SimpleNamespace(
        raw_item=SimpleNamespace(name="think", arguments="old")
    )
    direct_answer = SimpleNamespace(
        raw_item={"role": "assistant", "content": "final answer"}
    )
    tool_result = SimpleNamespace(
        generated_items=[prior_tool_call, direct_answer],
        new_step_items=[direct_answer],
    )

    assert (
        Ddv2AgentRunner._find_latest_function_call(tool_result.new_step_items) is None
    )
    assert "_find_latest_function_call(tool_result.new_step_items)" in (
        RUNNER_PATH.read_text(encoding="utf-8")
    )
