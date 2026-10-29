#!/usr/bin/env bash
# Tongyi DeepResearch runner. The vLLM services must already be running.

set -uo pipefail
DEEPRESEARCH_CI_TOTAL_TIMEOUT="${DEEPRESEARCH_CI_TOTAL_TIMEOUT:-1800}"
if ! [[ "$DEEPRESEARCH_CI_TOTAL_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    printf '[FAIL] DEEPRESEARCH_CI_TOTAL_TIMEOUT must be a positive integer\n' >&2
    exit 2
fi
if [ "${DEEPRESEARCH_CI_INNER:-0}" != "1" ]; then
    if ! command -v timeout >/dev/null 2>&1; then
        printf '[FAIL] GNU timeout is required\n' >&2
        exit 2
    fi
    export DEEPRESEARCH_CI_INNER=1
    exec timeout --signal=TERM --kill-after=20s "${DEEPRESEARCH_CI_TOTAL_TIMEOUT}s" bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DEEPRESEARCH_ENV_FILE:-$SCRIPT_DIR/.env}"
AGENT_ENV_NAME="${AGENT_ENV_NAME:-agentflashinfer}"
LOG_ROOT="${DEEPRESEARCH_CI_LOG_DIR:-$SCRIPT_DIR/ci_logs/deepresearch}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="$LOG_ROOT/$RUN_ID"
AGENT_LOG="$RUN_DIR/agent_server.log"
HTTP_CLIENT="$RUN_DIR/http_client.py"
SUMMARY_FILE="$RUN_DIR/summary.txt"

DEFAULT_TEST_QUERY="${DEEPRESEARCH_TEST_QUERY:-一位画家，父亲心脏病去世，有一个姐姐，与妻子育有五个子女，后婚姻破裂，后经历三段感情史。后有一部文学作品基于此人撰写，这部作品叫什么？}"
INTERACTIVE_MODE=false
SELECTED_QUERY="$DEFAULT_TEST_QUERY"
SHOW_HELP=false
DR_PID=""
DR_PGID=""

mkdir -p "$RUN_DIR"
cd "$SCRIPT_DIR"

log() {
    printf '%s\n' "$1" | tee -a "$SUMMARY_FILE"
}

fail() {
    log "[FAIL] $1"
    return 1
}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

print_usage() {
    cat <<'EOF'
Usage:
  bash run_tongyi.sh                                  # interactive in a terminal; strict verification otherwise
  bash run_tongyi.sh --interactive                    # prompt for a query and print the final answer
  bash run_tongyi.sh --query "your query"             # run one query and print the final answer
  DEEPRESEARCH_QUERY="your query" bash run_tongyi.sh # non-interactive query mode
  bash run_tongyi.sh --verify                         # force strict verification mode

The Qwen vLLM service or services must already be running. This script starts
and cleans up only the Tongyi DeepResearch Agent server.
EOF
}

configure_run_mode() {
    local cli_query=""
    local prompt_for_query=false
    local force_ci=false

    INTERACTIVE_MODE=false
    SELECTED_QUERY="$DEFAULT_TEST_QUERY"
    SHOW_HELP=false

    case "${1:-}" in
        --query)
            if [ "$#" -ne 2 ]; then
                fail "--query requires exactly one query argument"
                return 2
            fi
            cli_query="$2"
            ;;
        --interactive)
            if [ "$#" -ne 1 ]; then
                fail "--interactive does not accept additional arguments"
                return 2
            fi
            prompt_for_query=true
            ;;
        --verify|--ci)
            if [ "$#" -ne 1 ]; then
                fail "--verify does not accept additional arguments"
                return 2
            fi
            force_ci=true
            ;;
        -h|--help)
            SHOW_HELP=true
            return 0
            ;;
        "")
            if [ "$#" -ne 0 ]; then
                fail "Unknown argument: $1"
                return 2
            fi
            [ -t 0 ] && prompt_for_query=true
            ;;
        *)
            fail "Unknown argument: $1"
            return 2
            ;;
    esac

    if $force_ci; then
        SELECTED_QUERY="$DEFAULT_TEST_QUERY"
    elif [ -n "$cli_query" ]; then
        SELECTED_QUERY="$cli_query"
        INTERACTIVE_MODE=true
    elif $prompt_for_query; then
        printf 'Enter query: '
        if ! IFS= read -r SELECTED_QUERY; then
            fail "Unable to read query from standard input"
            return 2
        fi
        INTERACTIVE_MODE=true
    elif [ -n "${DEEPRESEARCH_QUERY:-}" ]; then
        SELECTED_QUERY="$DEEPRESEARCH_QUERY"
        INTERACTIVE_MODE=true
    fi

    if $INTERACTIVE_MODE && [[ "$SELECTED_QUERY" =~ ^[[:space:]]*$ ]]; then
        fail "Query must not be blank"
        return 2
    fi
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM

    if [ -n "$DR_PGID" ] && kill -0 -- "-$DR_PGID" 2>/dev/null; then
        log "[INFO] Stopping DeepResearch process group $DR_PGID"
        kill -TERM -- "-$DR_PGID" 2>/dev/null || true
        local attempt=0
        while kill -0 -- "-$DR_PGID" 2>/dev/null && [ "$attempt" -lt 20 ]; do
            sleep 0.5
            attempt=$((attempt + 1))
        done
        if kill -0 -- "-$DR_PGID" 2>/dev/null; then
            kill -KILL -- "-$DR_PGID" 2>/dev/null || true
        fi
    fi

    exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

load_config() {
    if [ ! -f "$ENV_FILE" ]; then
        fail "Environment file not found: $ENV_FILE"
        return 1
    fi

    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a

    MODEL_WEIGHTS_ROOT="${MODEL_WEIGHTS_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)/weights}"
    if is_true "${MODEL_ROUTING_SWITCH:-false}"; then
        VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-${LARGE_MODEL_NAME:-}}"
    else
        VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-${EXAMPLE_MODEL_NAME:-}}"
    fi
    VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$MODEL_WEIGHTS_ROOT/$VLLM_MODEL_NAME}"
    VLLM_PORTS="${VLLM_PORTS:-${VLLM_PORT:-6001}}"
    local first_vllm_port="${VLLM_PORTS%%,*}"
    VLLM_BASE_URL="${VLLM_BASE_URL:-${OPENAI_BASE_URL:-http://127.0.0.1:${first_vllm_port}/v1}}"
    VLLM_API_KEY="${VLLM_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"

    DR_HOST="${DR_HOST:-127.0.0.1}"
    DR_SERVER_PORT="${DR_SERVER_PORT:-8390}"
    DR_BASE_URL="http://$DR_HOST:$DR_SERVER_PORT"
    DR_WORKSPACE_PATH="$RUN_DIR/workspace"
    mkdir -p "$DR_WORKSPACE_PATH"
    DR_START_TIMEOUT="${DR_START_TIMEOUT:-120}"
    TEST_QUERY_TIMEOUT="${TEST_QUERY_TIMEOUT:-900}"
    if $INTERACTIVE_MODE; then
        TEST_QUERY="$SELECTED_QUERY"
    else
        TEST_QUERY="${DEEPRESEARCH_TEST_QUERY:-$SELECTED_QUERY}"
    fi

    REQUIRE_SEARCH="${DEEPRESEARCH_CI_REQUIRE_SEARCH:-true}"
    REQUIRE_MODEL_ROUTING="${DEEPRESEARCH_CI_REQUIRE_MODEL_ROUTING:-${MODEL_ROUTING_SWITCH:-false}}"
    REQUIRE_COMPRESSION="${DEEPRESEARCH_CI_REQUIRE_COMPRESSION:-true}"
    REQUIRE_EARLY_STOP="${DEEPRESEARCH_CI_REQUIRE_EARLY_STOP:-true}"


    MODEL_ROUTING_SWITCH=false
    SMALL_MODEL_URL="${SMALL_MODEL_URL:-$VLLM_BASE_URL}"
    LARGE_MODEL_URL="${LARGE_MODEL_URL:-$VLLM_BASE_URL}"
    SMALL_MODEL_NAME="${SMALL_MODEL_NAME:-$VLLM_MODEL_NAME}"
    LARGE_MODEL_NAME="${LARGE_MODEL_NAME:-$VLLM_MODEL_NAME}"
    SMALL_MODEL_API_KEY="${SMALL_MODEL_API_KEY:-$VLLM_API_KEY}"
    LARGE_MODEL_API_KEY="${LARGE_MODEL_API_KEY:-$VLLM_API_KEY}"
    PROGRESS_CHECK_MIN_ITER="${PROGRESS_CHECK_MIN_ITER:-1}"
    LARGE_ROUNDS="${LARGE_ROUNDS:-1}"
    if is_true "$REQUIRE_MODEL_ROUTING"; then
        MODEL_ROUTING_SWITCH=true
    fi

    COMPRESSION_ENABLED=false
    COMPRESSION_TOKEN_THRESHOLD="${COMPRESSION_TOKEN_THRESHOLD:-1}"
    COMPRESSION_TRIGGER_TOOLS="${COMPRESSION_TRIGGER_TOOLS:-search}"
    COMPRESSION_LLM_URL="${COMPRESSION_LLM_URL:-${VLLM_BASE_URL%/}/chat/completions}"
    COMPRESSION_LLM_API_KEY="${COMPRESSION_LLM_API_KEY:-$VLLM_API_KEY}"
    COMPRESSION_LLM_MODEL_NAME="${COMPRESSION_LLM_MODEL_NAME:-$VLLM_MODEL_NAME}"
    COMPRESSION_STALENESS_THRESHOLD="${COMPRESSION_STALENESS_THRESHOLD:-100}"
    if is_true "$REQUIRE_COMPRESSION"; then
        COMPRESSION_ENABLED=true
    fi

    EARLY_STOP_ENABLED=false
    EARLY_STOP_MIN_ITER="${EARLY_STOP_MIN_ITER:-1}"
    if is_true "$REQUIRE_EARLY_STOP"; then
        EARLY_STOP_ENABLED=true
    fi

    export PYTHONNOUSERSITE=1
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
    export no_proxy="$NO_PROXY"
    export VLLM_MODEL_NAME VLLM_MODEL_PATH VLLM_PORTS
    export MODEL_ROUTING_SWITCH SMALL_MODEL_URL LARGE_MODEL_URL
    export SMALL_MODEL_NAME LARGE_MODEL_NAME SMALL_MODEL_API_KEY LARGE_MODEL_API_KEY
    export PROGRESS_CHECK_MIN_ITER LARGE_ROUNDS
    export COMPRESSION_ENABLED COMPRESSION_TOKEN_THRESHOLD COMPRESSION_TRIGGER_TOOLS
    export COMPRESSION_LLM_URL COMPRESSION_LLM_API_KEY COMPRESSION_LLM_MODEL_NAME
    export COMPRESSION_STALENESS_THRESHOLD
    export EARLY_STOP_ENABLED EARLY_STOP_MIN_ITER
    export SEARCH_API_KEY URL_CRAWLER_API_KEYS DR_SERVER_PORT DR_WORKSPACE_PATH

    if [ -z "$VLLM_MODEL_NAME" ]; then
        fail "VLLM_MODEL_NAME is required"
        return 1
    fi
    if [ -z "$VLLM_MODEL_PATH" ] || [ ! -d "$VLLM_MODEL_PATH" ] || [ ! -r "$VLLM_MODEL_PATH" ]; then
        fail "VLLM_MODEL_PATH must point to a readable local tokenizer/model directory"
        return 1
    fi
    if is_true "$REQUIRE_SEARCH" && [ -z "${SEARCH_API_KEY:-}" ]; then
        fail "SEARCH_API_KEY is required when search validation is enabled"
        return 1
    fi

    {
        printf 'env_file=%s\n' "$ENV_FILE"
        printf 'agent_env=%s\n' "$AGENT_ENV_NAME"
        printf 'vllm_base_url=%s\n' "$VLLM_BASE_URL"
        printf 'vllm_model_name=%s\n' "$VLLM_MODEL_NAME"
        printf 'vllm_model_path=%s\n' "$VLLM_MODEL_PATH"
        printf 'dr_base_url=%s\n' "$DR_BASE_URL"
        printf 'require_search=%s\n' "$REQUIRE_SEARCH"
        printf 'require_model_routing=%s\n' "$REQUIRE_MODEL_ROUTING"
        printf 'require_compression=%s\n' "$REQUIRE_COMPRESSION"
        printf 'require_early_stop=%s\n' "$REQUIRE_EARLY_STOP"
    } > "$RUN_DIR/config.txt"
}

create_http_client() {
    cat > "$HTTP_CLIENT" <<'PYEOF'
#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def api_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("CI_HTTP_API_KEY", "")
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def v1_url(base_url: str, resource: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/{resource}"
    return f"{base}/v1/{resource}"


def new_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def probe_vllm(args) -> int:
    # Strictly probe /v1/models and /v1/chat/completions.
    output_dir = Path(args.output_dir)
    session = new_session()

    try:
        models_response = session.get(
            v1_url(args.base_url, "models"), headers=api_headers(), timeout=20
        )
        models_payload = models_response.json()
    except Exception as exc:
        models_payload = {"error": str(exc)}
        write_json(output_dir / f"vllm_{args.label}_models.json", models_payload)
        print(f"vLLM {args.label} model-list request failed: {exc}", file=sys.stderr)
        return 1

    write_json(output_dir / f"vllm_{args.label}_models.json", models_payload)
    model_ids = [item.get("id") for item in models_payload.get("data", [])]
    if models_response.status_code != 200 or args.model not in model_ids:
        print(
            f"vLLM {args.label} does not serve expected model {args.model!r}; "
            f"available={model_ids}",
            file=sys.stderr,
        )
        return 1

    try:
        chat_response = session.post(
            v1_url(args.base_url, "chat/completions"),
            headers=api_headers(),
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=90,
        )
        chat_payload = chat_response.json()
    except Exception as exc:
        chat_payload = {"error": str(exc)}
        write_json(output_dir / f"vllm_{args.label}_chat.json", chat_payload)
        print(f"vLLM {args.label} generation request failed: {exc}", file=sys.stderr)
        return 1

    write_json(output_dir / f"vllm_{args.label}_chat.json", chat_payload)
    choices = chat_payload.get("choices", [])
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    if chat_response.status_code != 200 or not str(content).strip():
        print(f"vLLM {args.label} returned no generated content", file=sys.stderr)
        return 1

    return 0


def probe_router_cycle(args) -> int:
    from examples.deepresearch.features.model_router import (
        DRModelRouter,
        DRProgressCheckStrategy,
    )
    from features.model_routing.config import ModelRoutingConfig

    config = ModelRoutingConfig(
        enabled=True,
        small_model_url=args.small_base_url,
        large_model_url=args.large_base_url,
        small_model_name=args.small_model,
        large_model_name=args.large_model,
    )
    strategy = DRProgressCheckStrategy(config=config, large_rounds=1)
    decisions = iter((False, True))
    strategy.should_escalate_sync = lambda *_args: next(decisions)
    router = DRModelRouter(config=config, strategy=strategy)
    endpoints = [router.resolve("DeepResearch", []) for _ in range(3)]
    steps = [
        {
            "model_type": endpoint.model_type,
            "model_name": endpoint.model_name,
            "base_url": endpoint.base_url,
        }
        for endpoint in endpoints
    ]
    expected = [
        {
            "model_type": "small_model",
            "model_name": args.small_model,
            "base_url": args.small_base_url,
        },
        {
            "model_type": "large_model",
            "model_name": args.large_model,
            "base_url": args.large_base_url,
        },
        {
            "model_type": "small_model",
            "model_name": args.small_model,
            "base_url": args.small_base_url,
        },
    ]
    payload = {"success": steps == expected, "steps": steps}
    write_json(Path(args.output), payload)
    return 0 if payload["success"] else 1


def agent_health(args) -> int:
    output = Path(args.output)
    try:
        response = new_session().get(f"{args.base_url.rstrip('/')}/api/stats", timeout=10)
        payload = response.json()
    except Exception as exc:
        payload = {"error": str(exc)}
        write_json(output, payload)
        return 1

    write_json(output, payload)
    return 0 if response.status_code == 200 and "total_running_tasks" in payload else 1


def run_agent(args) -> int:
    output_dir = Path(args.output_dir)
    session = new_session()
    base_url = args.base_url.rstrip("/")
    summary = {
        "success": False,
        "session_id": "",
        "task_id": "",
        "search_call": False,
        "search_result": False,
        "termination": "",
        "prediction": "",
        "history_verified": False,
        "errors": [],
    }

    try:
        response = session.post(
            f"{base_url}/api/submit",
            json={"user_input": args.query, "user_token": "deepresearch_ci", "concurrency": 1},
            timeout=30,
        )
        submit_payload = response.json()
        write_json(output_dir / "submit.json", submit_payload)
        if response.status_code != 200 or submit_payload.get("status") != "started":
            raise RuntimeError(f"submit failed: HTTP {response.status_code}: {submit_payload}")

        session_id = submit_payload.get("session_id", "")
        task_id = submit_payload.get("task_id", "")
        if not session_id or not task_id:
            raise RuntimeError("submit response is missing session_id or task_id")
        summary["session_id"] = session_id
        summary["task_id"] = task_id

        started_at = time.monotonic()
        trace_end = None
        with session.get(
            f"{base_url}/tracestream/{session_id}",
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=(10, args.timeout + 30),
        ) as stream_response:
            stream_response.raise_for_status()
            with (output_dir / "sse.jsonl").open("w", encoding="utf-8") as stream_file:
                for raw_line in stream_response.iter_lines(decode_unicode=True):
                    if time.monotonic() - started_at > args.timeout:
                        raise TimeoutError(f"DeepResearch exceeded {args.timeout}s")
                    if not raw_line or not raw_line.startswith("data:"):
                        continue
                    payload_text = raw_line[5:].strip()
                    if payload_text.startswith("data:"):
                        payload_text = payload_text[5:].strip()
                    if not payload_text:
                        continue
                    event = json.loads(payload_text)
                    stream_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                    stream_file.flush()

                    event_data = event.get("data", {})
                    if event.get("event") == "on_react_step":
                        if event_data.get("type") == "tool_call" and event_data.get("tool_name") == "search":
                            summary["search_call"] = True
                        if event_data.get("type") == "tool_result" and event_data.get("tool_name") == "search":
                            content = str(event_data.get("content", ""))
                            if re.search(r"found [1-9][0-9]* results", content):
                                summary["search_result"] = True
                    if event.get("event") == "on_trace_end":
                        trace_end = event_data
                        break

        if trace_end is None:
            summary["errors"].append("SSE ended without on_trace_end")
        else:
            summary["termination"] = str(trace_end.get("termination", ""))
            summary["prediction"] = str(trace_end.get("prediction", ""))
            if summary["termination"] != "answer":
                summary["errors"].append(
                    f"termination is {summary['termination']!r}, expected 'answer'"
                )
            if not summary["prediction"].strip():
                summary["errors"].append("prediction is empty")

        history_payload = []
        history_deadline = started_at + args.timeout
        last_history_error = ""
        while time.monotonic() < history_deadline:
            try:
                history_response = session.get(
                    f"{base_url}/history?task_id={session_id}", timeout=20
                )
                if history_response.status_code == 200:
                    candidate = history_response.json()
                    if isinstance(candidate, list):
                        history_payload = candidate
                        completed_item = next(
                            (
                                item
                                for item in candidate
                                if item.get("task_id") == session_id
                                and item.get("completed") is True
                                and str(item.get("agent_response", "")).strip()
                            ),
                            None,
                        )
                        if completed_item is not None:
                            history_answer = str(completed_item["agent_response"]).strip()
                            if history_answer == summary["prediction"].strip():
                                summary["history_verified"] = True
                            else:
                                summary["errors"].append(
                                    "history answer does not match SSE prediction"
                                )
                            break
                else:
                    last_history_error = f"history HTTP {history_response.status_code}"
            except Exception as exc:
                last_history_error = str(exc)
            time.sleep(0.25)

        write_json(output_dir / "history.json", history_payload)
        if not summary["history_verified"] and not any(
            error.startswith("history answer") for error in summary["errors"]
        ):
            detail = f": {last_history_error}" if last_history_error else ""
            summary["errors"].append(
                f"completed non-empty answer not found in history{detail}"
            )

        if args.require_search:
            if not summary["search_call"]:
                summary["errors"].append("search tool_call event was not observed")
            if not summary["search_result"]:
                summary["errors"].append("successful non-empty search result was not observed")

    except Exception as exc:
        summary["errors"].append(str(exc))
        if summary["session_id"]:
            try:
                session.get(f"{base_url}/api/stop/{summary['session_id']}", timeout=10)
            except Exception:
                pass

    summary["success"] = not summary["errors"]
    write_json(output_dir / "client_summary.json", summary)
    return 0 if summary["success"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe-vllm")
    probe.add_argument("--base-url", required=True)
    probe.add_argument("--model", required=True)
    probe.add_argument("--label", required=True)
    probe.add_argument("--output-dir", required=True)

    health = subparsers.add_parser("agent-health")
    health.add_argument("--base-url", required=True)
    health.add_argument("--output", required=True)

    router = subparsers.add_parser("probe-router-cycle")
    router.add_argument("--small-base-url", required=True)
    router.add_argument("--small-model", required=True)
    router.add_argument("--large-base-url", required=True)
    router.add_argument("--large-model", required=True)
    router.add_argument("--output", required=True)

    agent = subparsers.add_parser("run-agent")
    agent.add_argument("--base-url", required=True)
    agent.add_argument("--query", required=True)
    agent.add_argument("--timeout", type=int, required=True)
    agent.add_argument("--output-dir", required=True)
    agent.add_argument("--require-search", action="store_true")

    args = parser.parse_args()
    if args.command == "probe-vllm":
        return probe_vllm(args)
    if args.command == "agent-health":
        return agent_health(args)
    if args.command == "probe-router-cycle":
        return probe_router_cycle(args)
    return run_agent(args)


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF
}

run_python() {
    conda run --no-capture-output -n "$AGENT_ENV_NAME" python "$HTTP_CLIENT" "$@"
}

probe_vllm() {
    local base_url="$1"
    local model_name="$2"
    local api_key="$3"
    local label="$4"
    log "[INFO] Probing vLLM $label at $base_url (model=$model_name)"
    CI_HTTP_API_KEY="$api_key" run_python probe-vllm \
        --base-url "$base_url" \
        --model "$model_name" \
        --label "$label" \
        --output-dir "$RUN_DIR"
}

start_deepresearch() {
    if ! python3 - "$DR_HOST" "$DR_SERVER_PORT" <<'PYEOF'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind((sys.argv[1], int(sys.argv[2])))
PYEOF
    then
        fail "DeepResearch port $DR_HOST:$DR_SERVER_PORT is already occupied"
        return 1
    fi

    log "[INFO] Starting DeepResearch from conda env $AGENT_ENV_NAME"
    setsid conda run --no-capture-output -n "$AGENT_ENV_NAME" python -m uvicorn examples.deepresearch.server.app_dr:app_dr \
        --host "$DR_HOST" --port "$DR_SERVER_PORT" > "$AGENT_LOG" 2>&1 &
    DR_PID=$!
    DR_PGID=$DR_PID

    local elapsed=0
    while [ "$elapsed" -lt "$DR_START_TIMEOUT" ]; do
        if ! kill -0 "$DR_PID" 2>/dev/null; then
            tail -n 80 "$AGENT_LOG" | tee -a "$SUMMARY_FILE"
            fail "DeepResearch exited before becoming ready"
            return 1
        fi
        if run_python agent-health --base-url "$DR_BASE_URL" --output "$RUN_DIR/dr_health.json"; then
            log "[PASS] DeepResearch is ready at $DR_BASE_URL"
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done

    tail -n 80 "$AGENT_LOG" | tee -a "$SUMMARY_FILE"
    fail "DeepResearch startup timed out after ${DR_START_TIMEOUT}s"
    return 1
}

check_feature_evidence() {
    local failed=0

    if is_true "$REQUIRE_MODEL_ROUTING"; then
        if grep -q 'PROGRESS CHECK RAW' "$AGENT_LOG" \
            && grep -q '"success": true' "$RUN_DIR/routing_probe.json"; then
            log "[PASS] Model routing: runtime check plus small -> large -> small probe"
        else
            log "[FAIL] Strict model-routing evidence is incomplete"
            failed=1
        fi
    else
        log "[SKIP] Model routing was not required"
    fi

    if is_true "$REQUIRE_COMPRESSION"; then
        if grep -q '\[async_compress\] task submitted' "$AGENT_LOG" \
            && grep -qE '\[async_compress\] task completed: .*excluded_items=[1-9][0-9]*' "$AGENT_LOG"; then
            log "[PASS] Async compression was submitted and applied"
        else
            log "[FAIL] Strict async-compression evidence is incomplete"
            failed=1
        fi
    else
        log "[SKIP] Async compression was not required"
    fi

    if is_true "$REQUIRE_EARLY_STOP"; then
        if grep -qE '\[early_stop\] (repeat output detected|min_iter reached)' "$AGENT_LOG" \
            && grep -qE '\[early_stop\] prompt injected: reason=(repeat|min_iter), iter=[0-9]+, target=new_step_items' "$AGENT_LOG" \
            && grep -qE '\[early_stop\] prompt observed in next turn: reason=(repeat|min_iter), injected_iter=[0-9]+, current_iter=[0-9]+' "$AGENT_LOG"; then
            log "[PASS] Early-stop guidance was triggered"
        else
            log "[FAIL] Strict early-stop evidence is incomplete"
            failed=1
        fi
    else
        log "[SKIP] Early stop was not required"
    fi

    return "$failed"
}

print_final_answer() {
    local summary_path="$RUN_DIR/client_summary.json"
    if [ ! -f "$summary_path" ]; then
        log "[WARN] Final-answer artifact was not written: $summary_path"
        return 0
    fi

    {
        printf '\n==========================================\n'
        printf 'TONGYI DEEPRESEARCH FINAL ANSWER\n'
        printf '==========================================\n'
        python3 - "$summary_path" <<'PYEOF'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(str(summary.get("prediction", "")).strip())
PYEOF
        printf '==========================================\n'
        printf 'Result artifact: %s\n' "$summary_path"
    } | tee -a "$SUMMARY_FILE"
}

main() {
    configure_run_mode "$@" || return $?
    if $SHOW_HELP; then
        print_usage
        return 0
    fi

    log "Tongyi DeepResearch run"
    log "Artifacts: $RUN_DIR"

    load_config || return 1
    create_http_client

    probe_vllm "$VLLM_BASE_URL" "$VLLM_MODEL_NAME" "$VLLM_API_KEY" primary \
        || return 1
    log "[PASS] Primary vLLM model and minimal generation are healthy"

    if is_true "$REQUIRE_MODEL_ROUTING"; then
        probe_vllm "$SMALL_MODEL_URL" "$SMALL_MODEL_NAME" "$SMALL_MODEL_API_KEY" small \
            || return 1
        probe_vllm "$LARGE_MODEL_URL" "$LARGE_MODEL_NAME" "$LARGE_MODEL_API_KEY" large \
            || return 1
        log "[PASS] Small and large routing endpoints are healthy"
        run_python probe-router-cycle \
            --small-base-url "$SMALL_MODEL_URL" \
            --small-model "$SMALL_MODEL_NAME" \
            --large-base-url "$LARGE_MODEL_URL" \
            --large-model "$LARGE_MODEL_NAME" \
            --output "$RUN_DIR/routing_probe.json" || return 1
        log "[PASS] Router state machine maps small -> large -> small"
    fi

    start_deepresearch || return 1

    local search_arg=()
    if is_true "$REQUIRE_SEARCH"; then
        search_arg=(--require-search)
    fi

    if $INTERACTIVE_MODE; then
        log "[INFO] Mode: run one user query to completion"
        log "[INFO] Query: $TEST_QUERY"
        search_arg=()
    else
        log "[INFO] Mode: strict non-interactive verification"
        log "[INFO] Submitting DeepResearch query"
    fi
    local client_status=0
    run_python run-agent \
        --base-url "$DR_BASE_URL" \
        --query "$TEST_QUERY" \
        --timeout "$TEST_QUERY_TIMEOUT" \
        --output-dir "$RUN_DIR" \
        "${search_arg[@]}" || client_status=$?

    if [ "$client_status" -eq 0 ]; then
        if $INTERACTIVE_MODE; then
            log "[PASS] Final answer, SSE termination and history are valid"
            print_final_answer
            return 0
        fi
        log "[PASS] Search, final answer, SSE termination and history are valid"
    else
        if $INTERACTIVE_MODE; then
            log "[FAIL] DeepResearch request did not produce a completed answer"
            tail -n 100 "$AGENT_LOG" > "$RUN_DIR/agent_server_tail.log"
            return 1
        fi
        log "[FAIL] DeepResearch request did not produce strict successful evidence"
    fi

    local feature_status=0
    check_feature_evidence || feature_status=$?

    if [ "$client_status" -eq 0 ] && [ "$feature_status" -eq 0 ]; then
        log "[PASS] Tongyi DeepResearch run completed"
        return 0
    fi

    log "[FAIL] Tongyi DeepResearch run failed"
    tail -n 100 "$AGENT_LOG" > "$RUN_DIR/agent_server_tail.log"
    return 1
}

main "$@"
