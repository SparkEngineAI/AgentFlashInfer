"""Contract tests for the DeepResearch launcher."""

import subprocess
from pathlib import Path


RUN_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "deepresearch"
    / "run.sh"
)


def test_run_script_uses_project_env_and_separate_conda_environments() -> None:
    source = RUN_SCRIPT.read_text(encoding="utf-8")

    assert 'ENV_FILE="${DEEPRESEARCH_ENV_FILE:-$APP_ROOT/.env}"' in source
    assert 'VLLM_ENV_NAME=${VLLM_ENV_NAME:-new-vllm}' in source
    assert 'AGENT_ENV_NAME=${AGENT_ENV_NAME:-agentflashinfer}' in source
    assert 'CUDA_VISIBLE_DEVICES="$DEVICE_ID"' in source
    assert 'ASCEND_RT_VISIBLE_DEVICES' not in source
    assert 'conda run --no-capture-output -n "$VLLM_ENV_NAME"' in source
    assert '--served-model-name "$VLLM_MODEL_NAME"' in source
    assert 'conda run --no-capture-output -n "$AGENT_ENV_NAME"' in source


def test_run_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUN_SCRIPT)], check=True)
