#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============ Default Configuration ============
DEVICE_ID=${DEVICE_ID:-0}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-6001}
VLLM_MODEL_PATH=${VLLM_MODEL_PATH:-""}
VLLM_MODEL_NAME=${VLLM_MODEL_NAME:-""}
VLLM_ENV_NAME=${VLLM_ENV_NAME:-new-vllm}
AGENT_ENV_NAME=${AGENT_ENV_NAME:-agentflashinfer}
DR_SERVER_PORT=${DR_SERVER_PORT:-8290}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}

# ============ Load .env ============
ENV_FILE="${DEEPRESEARCH_ENV_FILE:-$APP_ROOT/.env}"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
fi
MODEL_WEIGHTS_ROOT=${MODEL_WEIGHTS_ROOT:-"$APP_ROOT/../weights"}
if [ "${MODEL_ROUTING_SWITCH:-false}" = "true" ]; then
    VLLM_MODEL_NAME=${VLLM_MODEL_NAME:-${LARGE_MODEL_NAME:-}}
else
    VLLM_MODEL_NAME=${VLLM_MODEL_NAME:-${EXAMPLE_MODEL_NAME:-}}
fi
VLLM_MODEL_PATH=${VLLM_MODEL_PATH:-$MODEL_WEIGHTS_ROOT/$VLLM_MODEL_NAME}
export VLLM_MODEL_NAME VLLM_MODEL_PATH


# ============ Validate ============
if [ -z "$VLLM_MODEL_PATH" ]; then
    echo "Error: VLLM_MODEL_PATH is not set"
    echo "Usage: VLLM_MODEL_PATH=/path/to/model bash run.sh"
    echo "   or: set VLLM_MODEL_PATH in .env"
    exit 1
fi

export VLLM_PORTS="$VLLM_PORT"
if [ -z "$VLLM_MODEL_NAME" ]; then
    export VLLM_MODEL_NAME="$(basename "$VLLM_MODEL_PATH")"
fi

# ============ 1. Start VLLM ============
VLLM_STARTED_BY_US=false

if curl --noproxy '*' -fsS "http://$VLLM_HOST:$VLLM_PORT/v1/models" >/dev/null; then
    echo "vLLM is already running on $VLLM_HOST:$VLLM_PORT; skipping startup."
else
    echo "Starting vLLM from conda env $VLLM_ENV_NAME on GPU $DEVICE_ID ..."
    CUDA_VISIBLE_DEVICES="$DEVICE_ID" \
      conda run --no-capture-output -n "$VLLM_ENV_NAME" \
      vllm serve "$VLLM_MODEL_PATH" \
      --served-model-name "$VLLM_MODEL_NAME" \
      --host "$VLLM_HOST" \
      --port "$VLLM_PORT" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --trust-remote-code \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --disable-log-requests &

    VLLM_PID=$!
    VLLM_STARTED_BY_US=true

    echo "Waiting for vLLM at http://$VLLM_HOST:$VLLM_PORT/v1 ..."
    timeout=600
    elapsed=0
    while ! curl --noproxy '*' -fsS "http://$VLLM_HOST:$VLLM_PORT/v1/models" >/dev/null; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Error: vLLM exited before becoming ready."
            exit 1
        fi
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "Error: vLLM startup timed out after ${timeout}s."
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "vLLM is ready (PID: $VLLM_PID)"
fi

# ============ 2. Start DeepResearch Service ============
echo "Starting DeepResearch from conda env $AGENT_ENV_NAME on port $DR_SERVER_PORT ..."
cd "$APP_ROOT"
PYTHONNOUSERSITE=1 PYTHONPATH="$APP_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  conda run --no-capture-output -n "$AGENT_ENV_NAME" \
  python -m examples.deepresearch.main &
DR_PID=$!

# Trap exit: kill both processes
cleanup() {
    echo ""
    echo "Shutting down ..."
    kill $DR_PID 2>/dev/null
    if [ "$VLLM_STARTED_BY_US" = true ]; then
        kill $VLLM_PID 2>/dev/null
        wait $VLLM_PID 2>/dev/null
    fi
    wait $DR_PID 2>/dev/null
    echo "Done"
}
trap cleanup EXIT INT TERM

echo "============================================"
echo " DeepResearch service is running"
echo " API docs:  http://127.0.0.1:$DR_SERVER_PORT/docs"
echo " VLLM:      http://$VLLM_HOST:$VLLM_PORT/v1"
echo "============================================"

wait "$DR_PID"
