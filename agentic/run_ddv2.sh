#!/bin/bash
# ==============================================================================
# DeepDiver end-to-end runner
# ==============================================================================
# This script performs comprehensive testing of the DeepDiver agent system,
# including vLLM service health check, MCP server startup, agent service test,
# and model routing verification.
#
#   1. Configure agentic/.env:
#      - MODEL_ROUTING_SWITCH=true: SMALL_MODEL_URL/NAME and LARGE_MODEL_URL/NAME
#      - MODEL_ROUTING_SWITCH=false: OPENAI_BASE_URL and EXAMPLE_MODEL_NAME
#      - SEARCH_API_KEY: Serper API key for web search
#      - URL_CRAWLER_API_KEYS: Jina API key for URL crawling
#
#   2. Start the configured vLLM service or services before running this script
#   2. Start vLLM inference service before running this script
#
#   3. Run: bash run_ddv2.sh
#
# ==============================================================================
# Test Steps:
#   Step 1: Validate Configuration
#      - Execute python config/config.py to verify all required configs
#
#   Step 2: Check vLLM Service Health
#      - Test vLLM service connectivity via /v1/chat/completions endpoint
#
#   Step 3: Test vLLM Inference
#      - Send a simple prompt to verify model inference works
#
#   Step 4: Start MCP Server
#      - Start MCP server in background (agent_infer/mcp_factory/mcp_server.py)
#      - Wait for service to be ready at http://localhost:6279/mcp
#
#   Step 5: Start Agent Server
#      - Start Agent server in background (examples/deepdiver_v2/server/app_ddv2.py)
#      - Wait for service to be ready at http://localhost:8289
#
#   Step 6: Test Agent with Search Tool
#      - Send a complex query requiring web search (in background)
#      - Monitor logs for 4 detection items in order:
#        1. Search tool call (batch_web_search etc.) [agent_log + mcp_log]
#        2. Model routing (PROGRESS CHECK RAW / [switch] / [large_model] / [small_model]) [agent_log ONLY]
#        3. Async compress ([async_compress] print markers) [agent_log ONLY]
#        4. Early stop ([early_stop] print markers) [agent_log ONLY]
#      - Display agent runtime progress: [large_model]/[small_model] responses,
#        PROGRESS CHECK, [switch], etc. from agent_log in real-time
#      - Early termination: once ALL 4 items are detected, terminate the
#        agent request to save test time (no need to wait for full completion)
#      - If timeout is reached before all items are detected, also terminate
#
#   Step 7: Check Model Routing, Async Compress & Early Stop
#      - All detections via agent_server log (print output):
#        - Model routing: PROGRESS CHECK RAW, [switch], [large_model], [small_model]
#        - Async compress: [async_compress] task submitted / task completed
#        - Early stop: [early_stop] repeat output / min_iter reached
#      - Serves as final confirmation if Step 6 already detected items
#
# ==============================================================================
# Log Files:
#   All logs are stored in: app/agentic/ci_logs/
#
#   - e2e_test_YYYYMMDD_HHMMSS.log    : Main test execution log
#   - config_YYYYMMDD_HHMMSS.log      : Configuration validation log
#   - mcp_server_YYYYMMDD_HHMMSS.log  : MCP server log
#   - agent_server_YYYYMMDD_HHMMSS.log: Agent server log
#
# ==============================================================================
# Useful Commands:
#
#   View main test log:
#     cat ci_logs/e2e_test_*.log
#
#   View Agent server log (real-time):
#     tail -f ci_logs/agent_server_*.log
#
#   View MCP server log:
#     cat ci_logs/mcp_server_*.log
#
#   Search for specific events in logs:
#     grep "PROGRESS CHECK" ci_logs/agent_server_*.log
#     grep "batch_web_search" ci_logs/*.log
#     grep -E "error|Error|ERROR" ci_logs/*.log
#
#   Check if services are running:
#     ps aux | grep -E "mcp_server|app_ddv2.py"
#     netstat -tlnp | grep -E "6279|8289"
#
#   Manually test Agent API:
#     curl -X POST http://localhost:8289/api/submit \
#       -H "Content-Type: application/json" \
#       -d '{"user_input": "你好", "agent_name": "assistant", "stream": false, "user_token": "test"}'
#
# ==============================================================================

# NOTE: no 'set -e' — each step returns 0/1 and main() counts failed_tests;
# aborting on first error would prevent later detection steps from running.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep the runner pinned to this checkout and its conda environment.
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Distributed workspace upload is unrelated to E2E feature validation and can
# start a large local Ray cluster after the request has already completed.
export DISABLE_DISTRIBUTED_STORAGE=true

cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/ci_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_LOG="$LOG_DIR/e2e_test_${TIMESTAMP}.log"

# Default timeouts (in seconds)
VLLM_TIMEOUT=30
MCP_TIMEOUT=60
AGENT_TIMEOUT=120
TEST_QUERY_TIMEOUT="${TEST_QUERY_TIMEOUT:-600}"
MCP_PORT="${MCP_PORT:-6279}"
AGENT_PORT="${AGENT_PORT:-8289}"
DEFAULT_TEST_QUERY="一位画家，父亲心脏病去世，有一个姐姐，与妻子育有五个子女，后婚姻破裂，后经历三段感情史。后有一部文学作品基于此人撰写，这部作品叫什么？"
INTERACTIVE_MODE=false
SELECTED_QUERY="$DEFAULT_TEST_QUERY"
SHOW_HELP=false
ROUTING_MODE=false

# PIDs for cleanup
MCP_PID=""
AGENT_PID=""

# ==============================================================================
# Utility Functions
# ==============================================================================

log() {
    echo -e "$1" | tee -a "$TEST_LOG"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[PASS]${NC} $1"
}

log_error() {
    log "${RED}[FAIL]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

print_usage() {
    cat <<'EOF'
Usage:
  bash run_ddv2.sh                         # interactive in a terminal; strict verification otherwise
  bash run_ddv2.sh --interactive           # prompt for a query and print the final answer
  bash run_ddv2.sh --query "your query"    # run one query and print the final answer
  DDV2_QUERY="your query" bash run_ddv2.sh # non-interactive query mode
  bash run_ddv2.sh --verify                # force strict verification mode

The vLLM service must already be running. This script starts and cleans up only
the MCP and DDv2 Agent servers.
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
                log_error "--query requires exactly one query argument"
                return 2
            fi
            cli_query="$2"
            ;;
        --interactive)
            if [ "$#" -ne 1 ]; then
                log_error "--interactive does not accept additional arguments"
                return 2
            fi
            prompt_for_query=true
            ;;
        --verify|--ci)
            if [ "$#" -ne 1 ]; then
                log_error "--verify does not accept additional arguments"
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
                log_error "Unknown argument: $1"
                return 2
            fi
            [ -t 0 ] && prompt_for_query=true
            ;;
        *)
            log_error "Unknown argument: $1"
            return 2
            ;;
    esac

    if $force_ci; then
        SELECTED_QUERY="$DEFAULT_TEST_QUERY"
        INTERACTIVE_MODE=false
    elif [ -n "$cli_query" ]; then
        SELECTED_QUERY="$cli_query"
        INTERACTIVE_MODE=true
    elif $prompt_for_query; then
        printf 'Enter query: '
        if ! IFS= read -r SELECTED_QUERY; then
            log_error "Unable to read query from standard input"
            return 2
        fi
        INTERACTIVE_MODE=true
    elif [ -n "${DDV2_QUERY:-}" ]; then
        SELECTED_QUERY="$DDV2_QUERY"
        INTERACTIVE_MODE=true
    else
        SELECTED_QUERY="$DEFAULT_TEST_QUERY"
        INTERACTIVE_MODE=false
    fi

    if $INTERACTIVE_MODE && [[ "$SELECTED_QUERY" =~ ^[[:space:]]*$ ]]; then
        log_error "Query must not be blank"
        return 2
    fi
}

port_is_available() {
    python3 - "$1" "$2" <<'PYEOF'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        raise SystemExit(1)
PYEOF
}

search_evidence_found() {
    local agent_log="$1"
    local mcp_log="$2"
    grep -qE '\[search\] batch_web_search completed: successful_queries=[1-9][0-9]*, result_items=[1-9][0-9]*' "$agent_log" "$mcp_log" 2>/dev/null
}

routing_evidence_found() {
    local agent_log="$1"
    grep -q 'PROGRESS CHECK RAW' "$agent_log" 2>/dev/null &&
        grep -qE '<value>[[:space:]]*FALSE[[:space:]]*</value>' "$agent_log" 2>/dev/null &&
        grep -qE '\[switch\].*switch to large_model' "$agent_log" 2>/dev/null &&
        awk '
            /\[model_routing\] selected=large_model reason=escalated / {
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^route_id=/) {
                        split($i, value, "=")
                        escalated[value[2]] = 1
                    }
                }
            }
            /\[model_routing\] selected=small_model reason=deescalated / {
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^route_id=/) {
                        split($i, value, "=")
                        if (escalated[value[2]]) {
                            found = 1
                        }
                    }
                }
            }
            END { exit(found ? 0 : 1) }
        ' "$agent_log"
}

compression_evidence_found() {
    local agent_log="$1"
    grep -q '\[async_compress\] task submitted' "$agent_log" 2>/dev/null &&
        grep -q 'LLM request succeeded' "$agent_log" 2>/dev/null &&
        grep -qE '\[async_compress\] task completed: .*excluded_items=[1-9][0-9]*' "$agent_log" 2>/dev/null
}

early_stop_evidence_found() {
    local agent_log="$1"
    grep -qE '\[early_stop\] (repeat output detected|min_iter reached)' "$agent_log" 2>/dev/null &&
        grep -qE '\[early_stop\] prompt injected: reason=(repeat|min_iter), iter=[0-9]+, target=new_step_items' "$agent_log" 2>/dev/null &&
        grep -qE '\[early_stop\] prompt observed in next turn: reason=(repeat|min_iter), injected_iter=[0-9]+, current_iter=[0-9]+' "$agent_log" 2>/dev/null
}

agent_failure_found() {
    local agent_log="$1"
    local app_log="$2"
    local app_log_start="${3:-0}"

    if grep -qE '\[agent_task\] failed:' "$agent_log" 2>/dev/null; then
        return 0
    fi
    [ -f "$app_log" ] || return 1

    local app_log_size
    app_log_size=$(wc -c < "$app_log" 2>/dev/null || echo 0)
    [ "$app_log_size" -gt "$app_log_start" ] || return 1
    tail -c +$((app_log_start + 1)) "$app_log" 2>/dev/null |
        grep -qE 'Exception occurred in function start_agent|Traceback \(most recent call last\)'
}

cleanup() {
    log_info "Cleaning up processes..."

    # Kill MCP server with timeout
    if [ -n "$MCP_PID" ] && kill -0 "$MCP_PID" 2>/dev/null; then
        log_info "Stopping MCP server (PID: $MCP_PID)..."
        kill -TERM "$MCP_PID" 2>/dev/null || true
        # Wait with timeout (5 seconds)
        local wait_count=0
        while kill -0 "$MCP_PID" 2>/dev/null && [ $wait_count -lt 10 ]; do
            sleep 0.5
            wait_count=$((wait_count + 1))
        done
        # Force kill if still running
        if kill -0 "$MCP_PID" 2>/dev/null; then
            kill -9 "$MCP_PID" 2>/dev/null || true
        fi
    fi

    # Kill Agent server with timeout
    if [ -n "$AGENT_PID" ] && kill -0 "$AGENT_PID" 2>/dev/null; then
        log_info "Stopping Agent server (PID: $AGENT_PID)..."
        kill -TERM "$AGENT_PID" 2>/dev/null || true
        # Wait with timeout (5 seconds)
        local wait_count=0
        while kill -0 "$AGENT_PID" 2>/dev/null && [ $wait_count -lt 10 ]; do
            sleep 0.5
            wait_count=$((wait_count + 1))
        done
        # Force kill if still running
        if kill -0 "$AGENT_PID" 2>/dev/null; then
            kill -9 "$AGENT_PID" 2>/dev/null || true
        fi
    fi

    log_info "Cleanup completed."
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# ==============================================================================
# Load Environment Variables
# ==============================================================================

load_env() {
    if [ ! -f "$SCRIPT_DIR/.env" ]; then
        log_error ".env file not found at $SCRIPT_DIR/.env"
        exit 1
    fi

    log_info "Loading environment variables from .env..."

    # Parse .env file manually to handle spaces around '=' and inline comments
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments (lines starting with #)
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

        # Extract key and value, handling spaces around '='
        if [[ "$line" =~ ^([a-zA-Z_][a-zA-Z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"

            # Remove surrounding quotes if present
            if [[ "$value" =~ ^\"(.*)\"$ ]]; then
                value="${BASH_REMATCH[1]}"
            elif [[ "$value" =~ ^\'(.*)\'$ ]]; then
                value="${BASH_REMATCH[1]}"
            fi

            # Remove inline comments (# not inside quotes)
            # If value contains #, take only the part before #
            if [[ "$value" == *#* ]]; then
                value="${value%%#*}"
            fi

            # Trim trailing whitespace
            value="${value%"${value##*[![:space:]]}"}"

            # Export the variable
            export "$key=$value"
        fi
    done < "$SCRIPT_DIR/.env"

    case "${MODEL_ROUTING_SWITCH:-true}" in
        1|true|TRUE|yes|YES|on|ON)
            ROUTING_MODE=true
            for key in SMALL_MODEL_URL SMALL_MODEL_NAME LARGE_MODEL_URL LARGE_MODEL_NAME; do
                if [ -z "${!key:-}" ]; then
                    log_error "$key not found in .env while MODEL_ROUTING_SWITCH=true"
                    exit 1
                fi
            done
            # Retained for the existing one-endpoint logging helpers; route
            # mode preflights both endpoints explicitly below.
            VLLM_BASE_URL="${SMALL_MODEL_URL%/v1}"
            VLLM_MODEL_NAME="$SMALL_MODEL_NAME"
            ;;
        *)
            ROUTING_MODE=false
            if [ -z "$OPENAI_BASE_URL" ] || [ -z "$EXAMPLE_MODEL_NAME" ]; then
                log_error "OPENAI_BASE_URL and EXAMPLE_MODEL_NAME are required when MODEL_ROUTING_SWITCH=false"
                exit 1
            fi
            VLLM_BASE_URL="${OPENAI_BASE_URL%/v1}"
            VLLM_MODEL_NAME="$EXAMPLE_MODEL_NAME"
            ;;
    esac

    # Disable mcp_hub_switch for this single-server run (no load balancing)
    export mcp_hub_switch=False
    export MCP_HUB_SWITCH=False

    if $ROUTING_MODE; then
        log_info "Small model: $SMALL_MODEL_URL ($SMALL_MODEL_NAME)"
        log_info "Large model: $LARGE_MODEL_URL ($LARGE_MODEL_NAME)"
    else
        log_info "Default model: $VLLM_BASE_URL ($VLLM_MODEL_NAME)"
    fi
    log_info "mcp_hub_switch: $mcp_hub_switch (disabled for this single-server run)"
}

# ==============================================================================
# Python Helper for HTTP Requests (no curl needed)
# ==============================================================================

# Create a Python script for HTTP testing
create_http_tester() {
    cat > "$LOG_DIR/http_tester.py" << 'PYEOF'
#!/usr/bin/env python3
import sys
import json
import time
import requests

def test_vllm_health(base_url, model_name, timeout=15):
    """Test vLLM service health via chat/completions"""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            if "choices" in data:
                return {"success": True, "message": "vLLM service is healthy"}
        return {"success": False, "message": f"Status: {resp.status_code}, Response: {resp.text[:200]}"}
    except Exception as e:
        return {"success": False, "message": str(e)}

def test_vllm_inference(base_url, model_name, timeout=60):
    """Test vLLM inference"""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello, who are you?"}],
        "max_tokens": 50
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"success": True, "message": f"Response: {content[:100]}..."}
        return {"success": False, "message": f"Status: {resp.status_code}"}
    except Exception as e:
        return {"success": False, "message": str(e)}

def test_mcp_health(host, port, timeout=5):
    """Test MCP server health"""
    url = f"http://{host}:{port}/mcp"
    try:
        resp = requests.get(url, timeout=timeout)
        return {"success": True, "message": f"MCP server reachable at {url}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "Connection refused"}
    except Exception as e:
        return {"success": False, "message": str(e)}

def test_agent_health(port, timeout=5):
    """Test Agent server health"""
    url = f"http://localhost:{port}/docs"
    try:
        resp = requests.get(url, timeout=timeout)
        return {"success": True, "message": f"Agent server reachable at port {port}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "Connection refused"}
    except Exception as e:
        return {"success": False, "message": str(e)}

def test_agent_api(port, query, timeout=3600):
    """Test Agent API (async mode via /api/submit)"""
    url = f"http://localhost:{port}/api/submit"
    payload = {
        "user_input": query,
        "agent_name": "assistant",
        "stream": False,
        "user_token": "ci_test_token",
        "concurrency": 1
    }
    try:
        # app_ddv2 uses /api/submit which returns immediately with session_id.
        # The shell script monitors logs for detection items and handles early
        # termination via /api/stop/{session_id}.
        resp = requests.post(url, json=payload, timeout=(10, 30))
        if resp.status_code == 200:
            data = resp.json()
            session_id = data.get("session_id", "")
            trace_ids = data.get("trace_ids", [])
            status = data.get("status", "")
            if not session_id or status != "started":
                return {
                    "success": False,
                    "message": f"Invalid submit response: {data}",
                }
            return {
                "success": True,
                "session_id": session_id,
                "trace_ids": trace_ids,
                "content": f"Session started: {session_id}",
                "message": f"Session ID: {session_id}, Status: {status}"
            }
        return {"success": False, "message": f"Status: {resp.status_code}, Response: {resp.text[:200]}"}
    except Exception as e:
        return {"success": False, "message": str(e)}

def wait_agent_result(port, session_id, timeout=3600):
    """Poll completed history for an asynchronously submitted session."""
    url = f"http://localhost:{port}/history"
    deadline = time.monotonic() + timeout
    last_message = ""
    while time.monotonic() < deadline:
        try:
            resp = requests.get(
                url,
                params={"task_id": session_id},
                timeout=(10, 30),
            )
            if resp.status_code == 200:
                data = resp.json()
                answers = [
                    str(item.get("agent_response", "")).strip()
                    for item in data
                    if item.get("task_id") == session_id
                    and item.get("completed") is True
                    and str(item.get("agent_response", "")).strip()
                ]
                if answers:
                    return {
                        "success": True,
                        "session_id": session_id,
                        "answers": answers,
                    }
            else:
                last_message = f"History HTTP status {resp.status_code}"
        except Exception as e:
            last_message = str(e)
        time.sleep(1)
    return {
        "success": False,
        "session_id": session_id,
        "message": last_message or f"Timed out after {timeout}s",
    }

def stop_agent_session(port, session_id, timeout=10):
    """Stop an agent session via /api/stop/{session_id}"""
    url = f"http://localhost:{port}/api/stop/{session_id}"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return {"success": True, "message": f"Session {session_id} stopped"}
        return {"success": False, "message": f"Status: {resp.status_code}"}
    except Exception as e:
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "message": "No command specified"}))
        sys.exit(1)

    command = sys.argv[1]

    if command == "vllm_health":
        base_url = sys.argv[2]
        model_name = sys.argv[3]
        result = test_vllm_health(base_url, model_name)
    elif command == "vllm_inference":
        base_url = sys.argv[2]
        model_name = sys.argv[3]
        result = test_vllm_inference(base_url, model_name)
    elif command == "mcp_health":
        host = sys.argv[2]
        port = int(sys.argv[3])
        result = test_mcp_health(host, port)
    elif command == "agent_health":
        port = int(sys.argv[2])
        result = test_agent_health(port)
    elif command == "agent_api":
        port = int(sys.argv[2])
        query = sys.argv[3]
        result = test_agent_api(port, query)
    elif command == "agent_result":
        port = int(sys.argv[2])
        session_id = sys.argv[3]
        timeout = int(sys.argv[4])
        result = wait_agent_result(port, session_id, timeout)
    elif command == "agent_stop":
        port = int(sys.argv[2])
        session_id = sys.argv[3]
        result = stop_agent_session(port, session_id)
    else:
        result = {"success": False, "message": f"Unknown command: {command}"}

    print(json.dumps(result))
PYEOF
}

# ==============================================================================
# Test Functions
# ==============================================================================

# Step 1: Validate Config
validate_config() {
    log_info "=========================================="
    log_info "Step 1: Validating Configuration"
    log_info "=========================================="

    log_info "Running python config/config.py to validate configuration..."

    if python config/config.py > "$LOG_DIR/config_${TIMESTAMP}.log" 2>&1; then
        log_success "Configuration validation passed"
        return 0
    else
        log_error "Configuration validation failed"
        log_error "Config log:"
        cat "$LOG_DIR/config_${TIMESTAMP}.log" | tee -a "$TEST_LOG"
        return 1
    fi
}

# Step 2: Check vLLM Model Service Health
test_vllm_service() {
    log_info "=========================================="
    log_info "Step 2: Checking vLLM Model Service Health"
    log_info "=========================================="

    if $ROUTING_MODE; then
        (
            ROUTING_MODE=false
            VLLM_BASE_URL="${SMALL_MODEL_URL%/v1}"
            VLLM_MODEL_NAME="$SMALL_MODEL_NAME"
            test_vllm_service
        ) || return 1
        (
            ROUTING_MODE=false
            VLLM_BASE_URL="${LARGE_MODEL_URL%/v1}"
            VLLM_MODEL_NAME="$LARGE_MODEL_NAME"
            test_vllm_service
        ) || return 1
        return 0
    fi

    local max_attempts=$((VLLM_TIMEOUT / 5))
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        log_info "Attempt $attempt/$max_attempts: Checking $VLLM_BASE_URL/v1/chat/completions..."

        local result
        result=$(python "$LOG_DIR/http_tester.py" vllm_health "$VLLM_BASE_URL" "$VLLM_MODEL_NAME" 2>&1)

        local success
        success=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "false")

        if [ "$success" = "True" ]; then
            log_success "vLLM service is reachable at $VLLM_BASE_URL"
            return 0
        else
            local message
            message=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('message', 'Unknown error'))" 2>/dev/null || echo "$result")
            log_info "Failed: $message, retrying..."
        fi

        attempt=$((attempt + 1))
        sleep 5
    done

    log_error "vLLM service is not reachable after ${VLLM_TIMEOUT}s"
    log_error "Please check if vLLM service is running at $VLLM_BASE_URL"
    return 1
}

# Step 3: Test vLLM Inference
test_vllm_inference() {
    log_info "=========================================="
    log_info "Step 3: Testing vLLM Inference"
    log_info "=========================================="

    if $ROUTING_MODE; then
        (
            ROUTING_MODE=false
            VLLM_BASE_URL="${SMALL_MODEL_URL%/v1}"
            VLLM_MODEL_NAME="$SMALL_MODEL_NAME"
            test_vllm_inference
        ) || return 1
        (
            ROUTING_MODE=false
            VLLM_BASE_URL="${LARGE_MODEL_URL%/v1}"
            VLLM_MODEL_NAME="$LARGE_MODEL_NAME"
            test_vllm_inference
        ) || return 1
        return 0
    fi

    log_info "Sending test prompt: 'Hello, who are you?'"

    local result
    result=$(python "$LOG_DIR/http_tester.py" vllm_inference "$VLLM_BASE_URL" "$VLLM_MODEL_NAME" 2>&1)

    local success
    success=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "false")

    if [ "$success" = "True" ]; then
        local message
        message=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('message', ''))" 2>/dev/null || echo "")
        log_success "vLLM inference test passed"
        log_info "$message"
        return 0
    else
        local message
        message=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('message', 'Unknown error'))" 2>/dev/null || echo "$result")
        log_error "vLLM inference test failed: $message"
        return 1
    fi
}

# Step 4: Start MCP Server
start_mcp_server() {
    log_info "=========================================="
    log_info "Step 4: Starting MCP Server"
    log_info "=========================================="

    local mcp_port="$MCP_PORT"
    local mcp_host="${MCP_HOST:-localhost}"

    if ! port_is_available "$mcp_host" "$mcp_port"; then
        log_error "MCP port $mcp_host:$mcp_port is already occupied; refusing to terminate its owner"
        return 1
    fi

    log_info "Starting MCP server on $mcp_host:$mcp_port..."

    # Start MCP server in background
    python agent_infer/mcp_factory/mcp_server.py > "$LOG_DIR/mcp_server_${TIMESTAMP}.log" 2>&1 &
    MCP_PID=$!

    log_info "MCP server PID: $MCP_PID"

    # Wait for MCP server to be ready
    local max_attempts=$((MCP_TIMEOUT / 3))
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        log_info "Waiting for MCP server... ($attempt/$max_attempts s)"

        local result
        result=$(python "$LOG_DIR/http_tester.py" mcp_health "$mcp_host" "$mcp_port" 2>&1)

        local success
        success=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "false")

        if [ "$success" = "True" ]; then
            log_success "MCP server is ready at http://$mcp_host:$mcp_port/mcp"
            return 0
        fi

        # Check if process is still running
        if ! kill -0 "$MCP_PID" 2>/dev/null; then
            log_error "MCP server process died unexpectedly"
            log_error "MCP log tail:"
            tail -50 "$LOG_DIR/mcp_server_${TIMESTAMP}.log" | tee -a "$TEST_LOG"
            return 1
        fi

        attempt=$((attempt + 1))
        sleep 3
    done

    log_error "MCP server failed to start within ${MCP_TIMEOUT}s"
    log_error "MCP log tail:"
    tail -50 "$LOG_DIR/mcp_server_${TIMESTAMP}.log" | tee -a "$TEST_LOG"
    return 1
}

# Step 5: Start Agent Server
start_agent_server() {
    log_info "=========================================="
    log_info "Step 5: Starting Agent Server"
    log_info "=========================================="

    local agent_port="$AGENT_PORT"

    if ! port_is_available "127.0.0.1" "$agent_port"; then
        log_error "Agent port 127.0.0.1:$agent_port is already occupied; refusing to terminate its owner"
        return 1
    fi

    log_info "Starting Agent server on port $agent_port..."

    # Start Agent server in background (app_ddv2.py)
    python examples/deepdiver_v2/server/app_ddv2.py > "$LOG_DIR/agent_server_${TIMESTAMP}.log" 2>&1 &
    AGENT_PID=$!

    log_info "Agent server PID: $AGENT_PID"

    # Wait for Agent server to be ready
    local max_attempts=$((AGENT_TIMEOUT / 3))
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        log_info "Waiting for Agent server... ($attempt/$max_attempts s)"

        # Check for startup failure in agent log first
        if [ -f "$LOG_DIR/agent_server_${TIMESTAMP}.log" ] && grep -q "Application startup failed\|Address already in use" "$LOG_DIR/agent_server_${TIMESTAMP}.log" 2>/dev/null; then
            log_error "Agent server startup failed (port conflict or MCP connection error)"
            log_error "Agent log tail:"
            tail -50 "$LOG_DIR/agent_server_${TIMESTAMP}.log" | tee -a "$TEST_LOG"
            return 1
        fi

        local result
        result=$(python "$LOG_DIR/http_tester.py" agent_health "$agent_port" 2>&1)

        local success
        success=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "false")

        if [ "$success" = "True" ]; then
            log_success "Agent server is ready at http://localhost:$agent_port"
            return 0
        fi

        # Check if process is still running
        if ! kill -0 "$AGENT_PID" 2>/dev/null; then
            log_error "Agent server process died unexpectedly"
            log_error "Agent log tail:"
            tail -50 "$LOG_DIR/agent_server_${TIMESTAMP}.log" | tee -a "$TEST_LOG"
            return 1
        fi

        attempt=$((attempt + 1))
        sleep 3
    done

    log_error "Agent server failed to start within ${AGENT_TIMEOUT}s"
    log_error "Agent log tail:"
    tail -50 "$LOG_DIR/agent_server_${TIMESTAMP}.log" | tee -a "$TEST_LOG"
    return 1
}

run_interactive_query() {
    local query="$1"
    local agent_port="$AGENT_PORT"
    local submit_file="$LOG_DIR/agent_api_result_${TIMESTAMP}.json"
    local result_file="$LOG_DIR/agent_final_result_${TIMESTAMP}.json"

    log_info "=========================================="
    log_info "Running DDv2 query to completion"
    log_info "=========================================="
    log_info "Query: $query"

    python "$LOG_DIR/http_tester.py" agent_api "$agent_port" "$query" \
        > "$submit_file" 2>&1

    local submit_success
    local session_id
    submit_success=$(python3 -c \
        "import json,sys; print(json.load(open(sys.argv[1])).get('success', False))" \
        "$submit_file" 2>/dev/null || echo "False")
    session_id=$(python3 -c \
        "import json,sys; print(json.load(open(sys.argv[1])).get('session_id', ''))" \
        "$submit_file" 2>/dev/null || echo "")
    if [ "$submit_success" != "True" ] || [ -z "$session_id" ]; then
        local submit_message
        submit_message=$(python3 -c \
            "import json,sys; print(json.load(open(sys.argv[1])).get('message', 'Invalid submit response'))" \
            "$submit_file" 2>/dev/null || cat "$submit_file")
        log_error "Agent API request failed: $submit_message"
        return 1
    fi

    log_info "Agent session started: session_id=$session_id"
    log_info "Waiting up to ${TEST_QUERY_TIMEOUT}s for the completed answer..."
    python "$LOG_DIR/http_tester.py" agent_result \
        "$agent_port" "$session_id" "$TEST_QUERY_TIMEOUT" \
        > "$result_file" 2>&1

    local result_success
    result_success=$(python3 -c \
        "import json,sys; print(json.load(open(sys.argv[1])).get('success', False))" \
        "$result_file" 2>/dev/null || echo "False")
    if [ "$result_success" != "True" ]; then
        local result_message
        result_message=$(python3 -c \
            "import json,sys; print(json.load(open(sys.argv[1])).get('message', 'No completed answer'))" \
            "$result_file" 2>/dev/null || cat "$result_file")
        log_error "Agent did not produce a completed answer: $result_message"
        python "$LOG_DIR/http_tester.py" agent_stop \
            "$agent_port" "$session_id" >/dev/null 2>&1 || true
        return 1
    fi

    {
        printf '\n==========================================\n'
        printf 'DDv2 FINAL ANSWER\n'
        printf '==========================================\n'
        python3 - "$result_file" <<'PYEOF'
import json
import sys

answers = json.load(open(sys.argv[1], encoding="utf-8"))["answers"]
for index, answer in enumerate(answers, 1):
    if len(answers) > 1:
        print(f"[Trace {index}]")
    print(answer)
PYEOF
        printf '==========================================\n'
        printf 'Result artifact: %s\n' "$result_file"
    } | tee -a "$TEST_LOG"
    return 0
}

# Step 6: Test Agent API with Search Tool
# Early termination: agent request runs in background; once ALL 4 detection
# items are confirmed in logs, the request is terminated early to save time.
# Detection order: search tool -> model routing -> async compress -> early stop
#
# Log source mapping (all detections use agent_server log via print markers):
#   - Model routing: print() -> agent_server log ONLY
#   - Async compress: print("[async_compress] ...") -> agent_server log ONLY
#   - Early stop: print("[early_stop] ...") -> agent_server log ONLY
#   - Search tool: can appear in both agent_server log and MCP server log
test_agent_with_search_tool() {
    log_info "=========================================="
    log_info "Step 6: Testing Agent with Search Tool"
    log_info "=========================================="

    local agent_port="$AGENT_PORT"
    # Use the default benchmark query to trigger model routing.
    local test_query="$DEFAULT_TEST_QUERY"

    log_info "Sending test query to Agent API (async /api/submit)..."
    log_info "Query: '$test_query'"

    local agent_log="$LOG_DIR/agent_server_${TIMESTAMP}.log"
    local mcp_log="$LOG_DIR/mcp_server_${TIMESTAMP}.log"
    local app_log="$SCRIPT_DIR/workspace/AgenticAppLogs.log"
    local app_log_start=0
    if [ -f "$app_log" ]; then
        app_log_start=$(wc -c < "$app_log" 2>/dev/null || echo 0)
    fi

    # Send agent API request — /api/submit returns immediately with session_id
    local agent_api_result_file="$LOG_DIR/agent_api_result_${TIMESTAMP}.json"
    python "$LOG_DIR/http_tester.py" agent_api "$agent_port" "$test_query" > "$agent_api_result_file" 2>&1
    local api_exit_code=$?

    # Parse session_id from the response
    local session_id=""
    local api_success=false
    if [ -f "$agent_api_result_file" ] && [ -s "$agent_api_result_file" ]; then
        local result
        result=$(cat "$agent_api_result_file" 2>/dev/null)
        local success_val
        success_val=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "false")
        if [ "$success_val" = "True" ]; then
            api_success=true
            session_id=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('session_id', ''))" 2>/dev/null || echo "")
            log_info "Agent session started: session_id=$session_id"
        else
            local message
            message=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('message', 'Unknown error'))" 2>/dev/null || echo "$result")
            log_error "Agent API request failed: $message"
        fi
    else
        log_error "Agent API returned no response"
    fi

    if ! $api_success; then
        log_error "Step 6 failed: Agent API request did not succeed"
        return 1
    fi

    local required_feature_count=4
    if ! $ROUTING_MODE; then
        required_feature_count=3
        log_info "Model routing is disabled; its runtime check will be skipped"
    fi

    log_info "Monitoring logs for $required_feature_count required detection items..."

    # 4 detection items (in priority order):
    #   1. search_tool: batch_web_search / web_search (agent_log + mcp_log)
    #   2. model_routing: PROGRESS CHECK RAW / [switch] / [large_model] / [small_model] (agent_log ONLY)
    #   3. async_compress: [async_compress] task submitted / task completed (agent_log ONLY)
    #   4. early_stop: [early_stop] repeat output / min_iter reached (agent_log ONLY)
    local max_wait=$TEST_QUERY_TIMEOUT
    local wait_interval=3
    local elapsed=0
    local search_tool_called=false
    local routing_detected=false
    local compress_detected=false
    local early_stop_detected=false
    local agent_failed=false

    # Track last read position for each log to display incremental content
    local agent_log_pos=0
    local mcp_log_pos=0

    while [ $elapsed -lt $max_wait ]; do
        if agent_failure_found "$agent_log" "$app_log" "$app_log_start"; then
            agent_failed=true
            log_error "Agent background task failed; stopping feature polling"
            if [ -f "$app_log" ]; then
                tail -c +$((app_log_start + 1)) "$app_log" 2>/dev/null > \
                    "$LOG_DIR/agent_app_error_${TIMESTAMP}.log"
                log_error "New Agent application log: $LOG_DIR/agent_app_error_${TIMESTAMP}.log"
                tail -50 "$LOG_DIR/agent_app_error_${TIMESTAMP}.log" |
                    tee -a "$TEST_LOG"
            else
                tail -50 "$agent_log" | tee -a "$TEST_LOG"
            fi
            break
        fi

        if ! $search_tool_called && search_evidence_found "$agent_log" "$mcp_log"; then
            search_tool_called=true
            log_success "[1/4] Search returned successful non-empty results"
        fi

        if ! $routing_detected && routing_evidence_found "$agent_log"; then
            routing_detected=true
            log_success "[2/4] Model routing completed small -> large -> small"
        fi

        if ! $compress_detected && compression_evidence_found "$agent_log"; then
            compress_detected=true
            log_success "[3/4] Async compression was submitted and applied"
        fi

        if ! $early_stop_detected && early_stop_evidence_found "$agent_log"; then
            early_stop_detected=true
            log_success "[4/4] Early-stop guidance was injected"
        fi

        # Early termination: all required items detected, no need to wait for full completion.
        if $search_tool_called && $compress_detected && $early_stop_detected && \
            { ! $ROUTING_MODE || $routing_detected; }; then
            log_info "All $required_feature_count required detection items confirmed, stopping agent session early to save time..."
            if [ -n "$session_id" ]; then
                python "$LOG_DIR/http_tester.py" agent_stop "$agent_port" "$session_id" 2>/dev/null || true
                log_info "Agent session stopped via /api/stop/$session_id"
            fi
            break
        fi

        # --- Display agent runtime progress from agent_log ---
        # Show: PROGRESS CHECK, [switch], [large_model], [small_model], model_routing
        if [ -f "$agent_log" ]; then
            local current_size
            current_size=$(wc -c < "$agent_log" 2>/dev/null || true)
            if [ "$current_size" -gt "$agent_log_pos" ]; then
                tail -c +$((agent_log_pos + 1)) "$agent_log" 2>/dev/null | while IFS= read -r line; do
                    # Filter and display model routing and key agent activity
                    if echo "$line" | grep -qE "PROGRESS CHECK|\\[switch\\]|\\[large_model\\]|\\[small_model\\]|\\[async_compress\\]|\\[early_stop\\]|model_routing|batch_web_search|url_crawler|task_id|session_id"; then
                        # Strip ANSI color codes for display
                        local clean_line
                        clean_line=$(echo "$line" | sed 's/\x1b\[[0-9;]*m//g' 2>/dev/null || echo "$line")
                        log_info "[Agent] $clean_line"
                    fi
                done
                agent_log_pos=$current_size
            fi
        fi

        # --- Display MCP runtime progress from mcp_log ---
        if [ -f "$mcp_log" ]; then
            local current_mcp_size
            current_mcp_size=$(wc -c < "$mcp_log" 2>/dev/null || true)
            if [ "$current_mcp_size" -gt "$mcp_log_pos" ]; then
                tail -c +$((mcp_log_pos + 1)) "$mcp_log" 2>/dev/null | while IFS= read -r line; do
                    if echo "$line" | grep -qE "batch_web_search|web_search|search_config"; then
                        log_info "[MCP] $line"
                    fi
                done
                mcp_log_pos=$current_mcp_size
            fi
        fi

        # Print detection progress
        log_info "Detection: search=$search_tool_called, routing=$routing_detected, compress=$compress_detected, early_stop=$early_stop_detected (${elapsed}s/${max_wait}s)"

        sleep $wait_interval
        elapsed=$((elapsed + wait_interval))
    done

    if $agent_failed; then
        if [ -n "$session_id" ]; then
            python "$LOG_DIR/http_tester.py" agent_stop "$agent_port" "$session_id" 2>/dev/null || true
        fi
        log_error "Step 6 failed - Agent background task raised an exception"
        return 1
    fi

    # If timeout reached, stop the session via API
    if [ $elapsed -ge $max_wait ] && [ -n "$session_id" ]; then
        log_warn "Agent request timed out after ${max_wait}s, stopping session..."
        python "$LOG_DIR/http_tester.py" agent_stop "$agent_port" "$session_id" 2>/dev/null || true
        log_info "Agent session stopped via /api/stop/$session_id"
    fi

    # Wait a moment for logs to be fully written
    sleep 2

    # Final check: verify all enabled items (catch any that were missed during polling)
    # Each item only checks the correct log source
    search_evidence_found "$agent_log" "$mcp_log" && search_tool_called=true
    if $ROUTING_MODE; then
        routing_evidence_found "$agent_log" && routing_detected=true
    fi
    compression_evidence_found "$agent_log" && compress_detected=true
    early_stop_evidence_found "$agent_log" && early_stop_detected=true

    # Summary
    local detected_count=0
    $search_tool_called && detected_count=$((detected_count + 1))
    if $ROUTING_MODE && $routing_detected; then
        detected_count=$((detected_count + 1))
    fi
    $compress_detected && detected_count=$((detected_count + 1))
    $early_stop_detected && detected_count=$((detected_count + 1))

    log_info "Detection results: $detected_count/$required_feature_count required items detected"
    $search_tool_called && log_success "  [1/4] Search tool: PASS" || log_warn "  [1/4] Search tool: NOT DETECTED"
    if $ROUTING_MODE; then
        $routing_detected && log_success "  [2/4] Model routing: PASS" || log_warn "  [2/4] Model routing: NOT DETECTED"
    else
        log_info "  [2/4] Model routing: SKIPPED (MODEL_ROUTING_SWITCH=false)"
    fi
    $compress_detected && log_success "  [3/4] Async compress: PASS" || log_warn "  [3/4] Async compress: NOT DETECTED"
    $early_stop_detected && log_success "  [4/4] Early stop: PASS" || log_warn "  [4/4] Early stop: NOT DETECTED"

    if $search_tool_called && $compress_detected && $early_stop_detected && \
        { ! $ROUTING_MODE || $routing_detected; }; then
        log_success "Step 6 passed - all $required_feature_count enabled runtime features have strict evidence"
        return 0
    fi

    log_error "Step 6 failed - only $detected_count/$required_feature_count enabled runtime features have strict evidence"
    return 1
}

# Step 7: Check Model Routing, Async Compress & Early Stop
# Note: If Step 6 already detected these via early termination,
# this step serves as a final confirmation and detailed report.
#
# All detections use agent_server log via print() markers:
#   - Model routing: print() -> agent_server log ONLY
#   - Async compress: print("[async_compress] ...") -> agent_server log ONLY
#   - Early stop: print("[early_stop] ...") -> agent_server log ONLY
test_model_routing() {
    if ! $ROUTING_MODE; then
        log_info "Step 7: Model routing disabled; routing check skipped"
        return 0
    fi
    log_info "=========================================="
    log_info "Step 7: Checking Model Routing, Async Compress & Early Stop"
    log_info "=========================================="

    local agent_log="$LOG_DIR/agent_server_${TIMESTAMP}.log"

    # ---- Model Routing Check (agent_log ONLY) ----
    log_info "Checking model routing via agent_server log (print output)..."

    local routing_found=false
    local progress_check_count=0
    local switch_to_large_count=0
    local switch_back_count=0
    local large_model_response_count=0
    local small_model_response_count=0

    if [ -f "$agent_log" ]; then
        progress_check_count=$(grep -c "PROGRESS CHECK RAW" "$agent_log" 2>/dev/null || true)
        switch_to_large_count=$(grep -c "\\[model_routing\\] selected=large_model reason=escalated" "$agent_log" 2>/dev/null || true)
        switch_back_count=$(grep -c "\\[model_routing\\] selected=small_model reason=deescalated" "$agent_log" 2>/dev/null || true)
        large_model_response_count=$(grep -c "\\[model_routing\\] selected=large_model" "$agent_log" 2>/dev/null || true)
        small_model_response_count=$(grep -c "\\[model_routing\\] selected=small_model" "$agent_log" 2>/dev/null || true)

        if routing_evidence_found "$agent_log"; then
            routing_found=true
            log_info "PROGRESS CHECK RAW=$progress_check_count, switch_to_large=$switch_to_large_count, switch_back_to_small=$switch_back_count"
            log_info "Model responses: large_model=$large_model_response_count, small_model=$small_model_response_count"
            log_info "Recent model routing entries:"
            grep -E "PROGRESS CHECK RAW|\\[switch\\].*switch to large_model|\\[switch\\].*switch back to small_model" "$agent_log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -5 | while read -r line; do
                log_info "  $line"
            done
            # Show recent model response entries
            if [ "$large_model_response_count" -gt 0 ] || [ "$small_model_response_count" -gt 0 ]; then
                log_info "Recent model response entries:"
                grep -E "\\[large_model\\]|\\[small_model\\]" "$agent_log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -5 | while read -r line; do
                    log_info "  $line"
                done
            fi
        fi
    else
        log_warn "Agent server log not found"
    fi

    # ---- Async Compress Check (agent_log ONLY, print("[async_compress] ...")) ----
    log_info "Checking async compress (resum_compress_async) via agent_server log..."

    local compress_found=false
    local compress_submit_count=0
    local compress_complete_count=0

    if [ -f "$agent_log" ]; then
        compress_submit_count=$(grep -c "\[async_compress\] task submitted" "$agent_log" 2>/dev/null || true)
        compress_complete_count=$(grep -c "\[async_compress\] task completed" "$agent_log" 2>/dev/null || true)

        if compression_evidence_found "$agent_log"; then
            compress_found=true
            log_info "Async compress: submitted=$compress_submit_count, completed=$compress_complete_count"
            log_info "Recent async compress entries:"
            grep "\[async_compress\]" "$agent_log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -5 | while read -r line; do
                log_info "  $line"
            done
        fi
    else
        log_warn "Agent server log not found"
    fi

    # ---- Early Stop Check (agent_log ONLY, print("[early_stop] ...")) ----
    log_info "Checking early stop (early_stop_enabled) via agent_server log..."

    local early_stop_found=false
    local early_stop_repeat_count=0
    local early_stop_analyze_count=0

    if [ -f "$agent_log" ]; then
        early_stop_repeat_count=$(grep -c "\[early_stop\] repeat output detected" "$agent_log" 2>/dev/null || true)
        early_stop_analyze_count=$(grep -c "\[early_stop\] min_iter reached" "$agent_log" 2>/dev/null || true)

        if early_stop_evidence_found "$agent_log"; then
            early_stop_found=true
            log_info "Early stop: repeat_detected=$early_stop_repeat_count, min_iter_reached=$early_stop_analyze_count"
            log_info "Recent early stop entries:"
            grep "\[early_stop\]" "$agent_log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -5 | while read -r line; do
                log_info "  $line"
            done
        fi
    else
        log_warn "Agent server log not found"
    fi

    # ---- Display Configuration ----
    log_info "Model routing config:"
    log_info "  Large Model URL: $LARGE_MODEL_URL"
    log_info "  Small Model URL: $SMALL_MODEL_URL"
    log_info "  Large Model Name: $LARGE_MODEL_NAME"
    log_info "  Small Model Name: $SMALL_MODEL_NAME"

    # ---- Summary ----
    if $routing_found && $compress_found && $early_stop_found; then
        log_success "Model routing completed small -> large -> small"
        log_success "Async compression was submitted and applied"
        log_success "Early-stop guidance was injected"
        return 0
    fi

    ! $routing_found && log_error "Strict model routing evidence is incomplete"
    ! $compress_found && log_error "Strict async compression evidence is incomplete"
    ! $early_stop_found && log_error "Strict early-stop guidance evidence is incomplete"
    return 1
}

# ==============================================================================
# Main Test Runner
# ==============================================================================

main() {
    configure_run_mode "$@"
    local mode_status=$?
    if [ "$mode_status" -ne 0 ]; then
        print_usage
        return "$mode_status"
    fi
    if $SHOW_HELP; then
        print_usage
        return 0
    fi

    log_info "=========================================="
    log_info "DeepDiver end-to-end run"
    log_info "Started at: $(date)"
    log_info "Log file: $TEST_LOG"
    log_info "=========================================="

    local failed_tests=0

    # Create HTTP tester script
    create_http_tester

    # Step 1: Load environment and validate config
    load_env

    if ! validate_config; then
        ((failed_tests++))
        log_error "Step 1 failed: Configuration validation failed"
    fi

    # Step 2 & 3: Check vLLM service
    if ! test_vllm_service; then
        ((failed_tests++))
        log_error "Step 2 failed: vLLM service not available"
    fi

    if ! test_vllm_inference; then
        ((failed_tests++))
        log_error "Step 3 failed: vLLM inference failed"
    fi

    # Step 4: Start MCP server
    if ! start_mcp_server; then
        ((failed_tests++))
        log_error "Step 4 failed: MCP server startup failed"
    fi

    # Step 5: Start Agent server
    if ! start_agent_server; then
        ((failed_tests++))
        log_error "Step 5 failed: Agent server startup failed"
    fi

    if $INTERACTIVE_MODE; then
        log_info "Mode: run one user query to completion"
        if [ "$failed_tests" -eq 0 ]; then
            if ! run_interactive_query "$SELECTED_QUERY"; then
                ((failed_tests++))
                log_error "Interactive query failed"
            fi
        else
            log_error "Skipping query because a prerequisite step failed"
        fi
    else
        log_info "Mode: strict non-interactive verification"
        # Step 6: Test Agent with search tool
        if ! test_agent_with_search_tool; then
            ((failed_tests++))
            log_error "Step 6 failed: Agent search tool test failed"
        fi

        # Step 7: Check model routing
        if ! test_model_routing; then
            ((failed_tests++))
            log_error "Step 7 failed: Model routing, async compress & early stop check failed"
        fi
    fi

    # Summary
    log_info "=========================================="
    log_info "Test Summary"
    log_info "=========================================="
    log_info "Completed at: $(date)"
    log_info "Failed tests: $failed_tests"

    return "$failed_tests"
}

# Run main
main "$@"
