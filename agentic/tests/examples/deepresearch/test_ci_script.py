"""Contract and behavior tests for the standalone DeepResearch CI script."""

import importlib
import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace


CI_SCRIPT = Path(__file__).resolve().parents[3] / "run_tongyi.sh"


def test_ci_uses_external_vllm_and_owns_only_deepresearch_process() -> None:
    source = CI_SCRIPT.read_text(encoding="utf-8")

    assert 'ENV_FILE="${DEEPRESEARCH_ENV_FILE:-$SCRIPT_DIR/.env}"' in source
    assert 'AGENT_ENV_NAME="${AGENT_ENV_NAME:-agentflashinfer}"' in source
    assert "vllm serve" not in source
    assert "examples/deepresearch/run.sh" not in source
    assert "setsid conda run --no-capture-output" in source
    assert 'python -m uvicorn examples.deepresearch.server.app_dr:app_dr' in source
    assert 'kill -TERM -- "-$DR_PGID"' in source
    assert "pkill" not in source
    assert "killall" not in source


def test_ci_requires_strict_runtime_evidence_and_persists_artifacts() -> None:
    source = CI_SCRIPT.read_text(encoding="utf-8")

    assert 'VLLM_MODEL_PATH' in source
    assert '/v1/models' in source
    assert '/v1/chat/completions' in source
    assert 'on_trace_end' in source
    assert 'termination' in source
    assert 'prediction' in source
    assert 'tool_name") == "search"' in source
    assert '/history?task_id={session_id}' in source
    assert 'client_summary.json' in source
    assert 'summary.txt' in source
    assert 'DR_WORKSPACE_PATH="$RUN_DIR/workspace"' in source
    assert 'DR_SERVER_PORT="${DR_SERVER_PORT:-8390}"' in source
    assert "time.monotonic() - started_at > args.timeout" in source
    assert 'export PYTHONUNBUFFERED=1' in source
    assert 'payload_text.startswith("data:")' in source
    assert "prompt injected: reason=(repeat|min_iter)" in source
    assert "target=new_step_items" in source
    assert "prompt observed in next turn: reason=(repeat|min_iter)" in source
    assert "COMPRESSION_STALENESS_THRESHOLD" in source



def _embedded_client_namespace() -> dict:
    source = CI_SCRIPT.read_text(encoding="utf-8")
    marker = 'cat > "$HTTP_CLIENT" <<\'PYEOF\'\n'
    embedded = source.split(marker, 1)[1].split("\nPYEOF\n", 1)[0]
    namespace = {"__name__": "deepresearch_ci_test_helper"}
    original_path = sys.path[:]
    for module_name in list(sys.modules):
        if module_name in {"examples", "features"} or module_name.startswith(
            ("examples.", "features.")
        ):
            sys.modules.pop(module_name)
    sys.path.insert(0, str(CI_SCRIPT.parent))
    try:
        exec(compile(embedded, "http_client.py", "exec"), namespace)
        importlib.import_module("examples.deepresearch.features.model_router")
        importlib.import_module("features.model_routing.config")
    finally:
        sys.path[:] = original_path
    return namespace


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _base_ci_env(tmp_path: Path, fake_bin: Path) -> dict[str, str]:
    env_file = tmp_path / "empty.env"
    env_file.write_text("", encoding="utf-8")
    model_path = tmp_path / "model"
    model_path.mkdir()
    return {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DEEPRESEARCH_CI_INNER": "1",
        "DEEPRESEARCH_ENV_FILE": str(env_file),
        "DEEPRESEARCH_CI_LOG_DIR": str(tmp_path / "logs"),
        "VLLM_MODEL_NAME": "test-model",
        "VLLM_MODEL_PATH": str(model_path),
        "VLLM_BASE_URL": "http://127.0.0.1:9999/v1",
        "DR_SERVER_PORT": str(_free_port()),
        "DEEPRESEARCH_CI_REQUIRE_SEARCH": "false",
        "DEEPRESEARCH_CI_REQUIRE_MODEL_ROUTING": "false",
        "DEEPRESEARCH_CI_REQUIRE_COMPRESSION": "false",
        "DEEPRESEARCH_CI_REQUIRE_EARLY_STOP": "false",
    }


def test_disabled_feature_checks_explicitly_disable_agent_features(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_conda = fake_bin / "conda"
    fake_conda.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            case "$*" in
                *"python -m uvicorn"*)
                    printf '%s %s %s\n' \
                        "$MODEL_ROUTING_SWITCH" "$COMPRESSION_ENABLED" "$EARLY_STOP_ENABLED" \
                        > "$FAKE_ENV_CAPTURE"
                    trap 'exit 0' TERM INT
                    while true; do sleep 0.1; done
                    ;;
                *"probe-vllm"*|*"agent-health"*|*"run-agent"*) exit 0 ;;
                *) exit 1 ;;
            esac
            """
        ),
        encoding="utf-8",
    )
    fake_conda.chmod(0o755)
    capture = tmp_path / "feature_env.txt"
    env = _base_ci_env(tmp_path, fake_bin)
    env["MODEL_ROUTING_SWITCH"] = "false"
    env.pop("DEEPRESEARCH_CI_REQUIRE_MODEL_ROUTING")
    env["FAKE_ENV_CAPTURE"] = str(capture)

    result = subprocess.run(
        ["bash", str(CI_SCRIPT)], env=env, text=True, capture_output=True, timeout=10
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert capture.read_text(encoding="utf-8").strip() == "false false false"


def test_query_mode_submits_the_supplied_query(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_conda = fake_bin / "conda"
    fake_conda.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            case "$*" in
                *"python -m uvicorn"*)
                    trap 'exit 0' TERM INT
                    while true; do sleep 0.1; done
                    ;;
                *"probe-vllm"*|*"agent-health"*) exit 0 ;;
                *"run-agent"*) printf '%s\\n' "$*" > "$FAKE_RUN_AGENT_ARGS"; exit 0 ;;
                *) exit 1 ;;
            esac
            """
        ),
        encoding="utf-8",
    )
    fake_conda.chmod(0o755)
    env = _base_ci_env(tmp_path, fake_bin)
    captured_args = tmp_path / "run_agent_args.txt"
    env["FAKE_RUN_AGENT_ARGS"] = str(captured_args)

    result = subprocess.run(
        ["bash", str(CI_SCRIPT), "--query", "what is the answer?"],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--query what is the answer?" in captured_args.read_text(encoding="utf-8")


def test_interactive_mode_reads_query_from_standard_input(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_conda = fake_bin / "conda"
    fake_conda.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            case "$*" in
                *"python -m uvicorn"*)
                    trap 'exit 0' TERM INT
                    while true; do sleep 0.1; done
                    ;;
                *"probe-vllm"*|*"agent-health"*) exit 0 ;;
                *"run-agent"*) printf '%s\\n' "$*" > "$FAKE_RUN_AGENT_ARGS"; exit 0 ;;
                *) exit 1 ;;
            esac
            """
        ),
        encoding="utf-8",
    )
    fake_conda.chmod(0o755)
    env = _base_ci_env(tmp_path, fake_bin)
    captured_args = tmp_path / "run_agent_args.txt"
    env["FAKE_RUN_AGENT_ARGS"] = str(captured_args)

    result = subprocess.run(
        ["bash", str(CI_SCRIPT), "--interactive"],
        env=env,
        input="interactive question\n",
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--query interactive question" in captured_args.read_text(encoding="utf-8")


def test_agent_client_polls_history_until_completed_answer(tmp_path: Path) -> None:
    namespace = _embedded_client_namespace()

    class FakeResponse:
        def __init__(self, payload=None, lines=None):
            self.status_code = 200
            self._payload = payload
            self._lines = lines or []

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            return iter(self._lines)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeSession:
        def __init__(self):
            self.history_calls = 0

        def post(self, *_args, **_kwargs):
            return FakeResponse(
                {"session_id": "sess_1", "task_id": "task_1", "status": "started"}
            )

        def get(self, url, **_kwargs):
            if "/tracestream/" in url:
                event = {
                    "event": "on_trace_end",
                    "data": {"termination": "answer", "prediction": "final answer"},
                }
                # EventSourceResponse wraps the app's preformatted SSE line,
                # so the wire payload contains a nested data: prefix.
                return FakeResponse(lines=["data: data: " + json.dumps(event)])
            if "/history?" in url:
                self.history_calls += 1
                if self.history_calls == 1:
                    return FakeResponse([])
                return FakeResponse([
                    {
                        "task_id": "sess_1",
                        "completed": True,
                        "agent_response": "final answer",
                    }
                ])
            raise AssertionError(url)

    fake_session = FakeSession()
    namespace["new_session"] = lambda: fake_session
    args = SimpleNamespace(
        base_url="http://agent",
        query="question",
        timeout=5,
        output_dir=str(tmp_path),
        require_search=False,
    )

    assert namespace["run_agent"](args) == 0
    assert fake_session.history_calls == 2
    assert json.loads(
        (tmp_path / "client_summary.json").read_text(encoding="utf-8")
    )["success"] is True


def test_router_probe_covers_real_endpoint_mapping_and_cycle(tmp_path: Path) -> None:
    namespace = _embedded_client_namespace()
    args = SimpleNamespace(
        small_base_url="http://small:6002/v1",
        small_model="Qwen3.5-4B",
        large_base_url="http://large:6001/v1",
        large_model="Qwen3.5-27B",
        output=str(tmp_path / "routing_probe.json"),
    )

    assert namespace["probe_router_cycle"](args) == 0
    payload = json.loads((tmp_path / "routing_probe.json").read_text())
    assert [step["model_type"] for step in payload["steps"]] == [
        "small_model",
        "large_model",
        "small_model",
    ]
    assert payload["steps"][0]["model_name"] == "Qwen3.5-4B"
    assert payload["steps"][1]["model_name"] == "Qwen3.5-27B"
    assert payload["success"] is True


def test_total_timeout_covers_vllm_probe(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_conda = fake_bin / "conda"
    fake_conda.write_text(
        "#!/usr/bin/env bash\nsleep 30\n",
        encoding="utf-8",
    )
    fake_conda.chmod(0o755)
    env = _base_ci_env(tmp_path, fake_bin)
    env.pop("DEEPRESEARCH_CI_INNER")
    env["DEEPRESEARCH_CI_TOTAL_TIMEOUT"] = "1"

    result = subprocess.run(
        ["bash", str(CI_SCRIPT)], env=env, text=True, capture_output=True, timeout=6
    )

    assert result.returncode == 124

def test_ci_script_is_executable_and_has_valid_bash_syntax() -> None:
    assert os.access(CI_SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(CI_SCRIPT)], check=True)


def test_ci_inherits_model_routing_requirement_from_the_environment() -> None:
    source = CI_SCRIPT.read_text(encoding="utf-8")

    assert (
        'REQUIRE_MODEL_ROUTING="${DEEPRESEARCH_CI_REQUIRE_MODEL_ROUTING:-${MODEL_ROUTING_SWITCH:-false}}"'
        in source
    )
