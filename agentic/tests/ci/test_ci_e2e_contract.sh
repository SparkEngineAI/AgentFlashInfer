#!/usr/bin/env bash

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTIC_DIR="$(cd "$TEST_DIR/../.." && pwd)"
AGENT_APP="$AGENTIC_DIR/examples/deepdiver_v2/server/app_ddv2.py"
CI_SCRIPT="$AGENTIC_DIR/run_ddv2.sh"
TMP_DIR="$(mktemp -d /tmp/ddv2-ci-contract.XXXXXX)"

# Source the functions without executing main. The copied script intentionally
# uses /tmp as SCRIPT_DIR so this test never touches the real CI artifacts.
sed '/^main "\$@"$/d' "$CI_SCRIPT" > "$TMP_DIR/ci_lib.sh"
# shellcheck disable=SC1090
source "$TMP_DIR/ci_lib.sh"
trap - EXIT
trap 'rm -rf -- "$TMP_DIR"' EXIT

declare -F search_evidence_found >/dev/null || fail "missing strict search detector"
declare -F routing_evidence_found >/dev/null || fail "missing strict routing detector"
declare -F compression_evidence_found >/dev/null || fail "missing strict compression detector"
declare -F early_stop_evidence_found >/dev/null || fail "missing strict early-stop detector"
declare -F agent_failure_found >/dev/null || fail "missing agent failure detector"
declare -F configure_run_mode >/dev/null || fail "missing CLI query mode parser"
declare -F run_interactive_query >/dev/null || fail "missing interactive completion runner"

create_http_tester
python3 -m py_compile "$LOG_DIR/http_tester.py" ||
    fail "generated HTTP helper has invalid Python syntax"
if ! python3 - "$LOG_DIR/http_tester.py" <<'PYEOF'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("ddv2_http_tester", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

module.requests.post = lambda *_args, **_kwargs: Response({"error": "Invalid input"})
assert module.test_agent_api(8289, "question")["success"] is False

responses = iter([
    Response([]),
    Response([{
        "task_id": "sess_test",
        "completed": True,
        "agent_response": "completed answer",
    }]),
])
module.requests.get = lambda *_args, **_kwargs: next(responses)
module.time.sleep = lambda _seconds: None
result = module.wait_agent_result(8289, "sess_test", timeout=5)
assert result["success"] is True
assert result["answers"] == ["completed answer"]
PYEOF
then
    fail "generated HTTP helper failed submit/result behavior checks"
fi

DDV2_QUERY="environment query"
configure_run_mode --query "CLI query" || fail "--query mode was rejected"
[ "$INTERACTIVE_MODE" = true ] || fail "--query did not enable completion mode"
[ "$SELECTED_QUERY" = "CLI query" ] || fail "--query did not override DDV2_QUERY"

DDV2_QUERY="environment query"
configure_run_mode || fail "DDV2_QUERY mode was rejected"
[ "$INTERACTIVE_MODE" = true ] || fail "DDV2_QUERY did not enable completion mode"
[ "$SELECTED_QUERY" = "environment query" ] || fail "DDV2_QUERY was not selected"

configure_run_mode --query "   " >/dev/null 2>&1 &&
    fail "blank --query was accepted"

DDV2_QUERY="environment query"
configure_run_mode --ci || fail "explicit strict CI mode was rejected"
[ "$INTERACTIVE_MODE" = false ] || fail "--ci did not select strict CI mode"
[ "$SELECTED_QUERY" = "$DEFAULT_TEST_QUERY" ] ||
    fail "--ci did not restore the default benchmark query"

DDV2_QUERY="environment query"
configure_run_mode --interactive <<<"prompted query" ||
    fail "interactive prompt mode was rejected"
[ "$SELECTED_QUERY" = "prompted query" ] ||
    fail "explicit --interactive did not override DDV2_QUERY"

python() {
    case "$*" in
        *" agent_api "*)
            printf '%s\n' '{"success": true, "session_id": "sess_test", "message": "started"}'
            ;;
        *" agent_result "*)
            printf '%s\n' '{"success": true, "session_id": "sess_test", "answers": ["completed answer"]}'
            ;;
        *)
            command python "$@"
            ;;
    esac
}
interactive_output=$(run_interactive_query "question" 2>&1) ||
    fail "interactive runner rejected a completed answer: $interactive_output"
printf '%s' "$interactive_output" | grep -q "completed answer" ||
    fail "interactive runner did not print the completed answer"
unset -f python
unset DDV2_QUERY

agent_log="$TMP_DIR/agent.log"
mcp_log="$TMP_DIR/mcp.log"

# Registration and generic debug markers must not count as feature success.
cat > "$mcp_log" <<'EOF'
Tool batch_web_search registered successfully
EOF
cat > "$agent_log" <<'EOF'
[large_model] startup
[async_compress] submit check: tool_name=think
[early_stop] usage debug
EOF

search_evidence_found "$agent_log" "$mcp_log" && fail "tool registration counted as search success"
routing_evidence_found "$agent_log" && fail "generic model marker counted as routing success"
compression_evidence_found "$agent_log" && fail "submit check counted as compression success"
early_stop_evidence_found "$agent_log" && fail "generic marker counted as early-stop success"

cat > "$agent_log" <<'EOF'
[early_stop] min_iter reached, analyze progress, iter=2
[early_stop] prompt injected: reason=min_iter, iter=2, target=new_step_items
EOF
early_stop_evidence_found "$agent_log" &&
    fail "unobserved early-stop injection counted as success"

cat > "$agent_log" <<'EOF'
======================================== PROGRESS CHECK RAW ========================================
<value>FALSE</value>
[switch] PROGRESS=FALSE, switch to large_model for 2 THINK rounds.
[model_routing] selected=large_model reason=escalated agent=ObjectiveInformationSeekerAgent endpoint=http://127.0.0.1:8103/v1 route_id=seeker-0
[model_routing] selected=small_model reason=deescalated agent=ObjectiveInformationSeekerAgent endpoint=http://127.0.0.1:8103/v1 route_id=seeker-1
EOF
routing_evidence_found "$agent_log" &&
    fail "routing transitions from different runs counted as one cycle"

app_log="$TMP_DIR/app.log"
cat > "$app_log" <<'EOF'
old failure before this CI request
Exception occurred in function start_agent
EOF
app_log_start=$(wc -c < "$app_log")
agent_failure_found "$agent_log" "$app_log" "$app_log_start" &&
    fail "old application failure counted against the current request"
cat >> "$app_log" <<'EOF'
Exception occurred in function start_agent
EOF
agent_failure_found "$agent_log" "$app_log" "$app_log_start" ||
    fail "new application failure was not detected"

cat > "$agent_log" <<'EOF'
[agent_task] failed: task_id=task_1 type=NameError
EOF
agent_failure_found "$agent_log" "$app_log" "$(wc -c < "$app_log")" ||
    fail "stdout background failure was not detected"

cat > "$mcp_log" <<'EOF'
[search] batch_web_search completed: successful_queries=2, result_items=11
EOF
cat > "$agent_log" <<'EOF'
======================================== PROGRESS CHECK RAW ========================================
<value>FALSE</value>
[switch] PROGRESS=FALSE, switch to large_model for 2 THINK rounds.
[model_routing] selected=large_model reason=escalated agent=ObjectiveInformationSeekerAgent endpoint=http://127.0.0.1:8103/v1 route_id=seeker-0
[model_routing] selected=small_model reason=deescalated agent=ObjectiveInformationSeekerAgent endpoint=http://127.0.0.1:8103/v1 route_id=seeker-0
[async_compress] task submitted: comp_1, token_num_before=100, turn=2
LLM request succeeded, usage={'total_tokens': 50}
[async_compress] task completed: comp_1, turn=3, excluded_items=4
[early_stop] min_iter reached, analyze progress, iter=2
[early_stop] prompt injected: reason=min_iter, iter=2, target=new_step_items
[early_stop] prompt observed in next turn: reason=min_iter, injected_iter=2, current_iter=3
EOF

search_evidence_found "$agent_log" "$mcp_log" || fail "valid search evidence rejected"
routing_evidence_found "$agent_log" || fail "valid routing transition rejected"
compression_evidence_found "$agent_log" || fail "valid applied compression rejected"
early_stop_evidence_found "$agent_log" || fail "valid early-stop injection rejected"

if grep -Eq 'pkill|xargs[^#]*kill' "$CI_SCRIPT"; then
    fail "CI contains broad process-kill cleanup"
fi

grep -q 'port=int(os.getenv("AGENT_PORT", "8289"))' "$AGENT_APP" ||
    fail "agent server does not honor AGENT_PORT"
if grep -q 'reload=True' "$AGENT_APP"; then
    fail "agent server enables a reloader that escapes CI PID cleanup"
fi


grep -q 'export PYTHONNOUSERSITE=1' "$CI_SCRIPT" ||
    fail "CI allows user-site dependency pollution"
grep -q 'export PYTHONPATH="$SCRIPT_DIR' "$CI_SCRIPT" ||
    fail "CI does not prioritize the current repository"
grep -q 'export DISABLE_DISTRIBUTED_STORAGE=true' "$CI_SCRIPT" ||
    fail "CI does not disable unrelated distributed workspace uploads"

cat > "$TMP_DIR/.env" <<'EOF'
MODEL_ROUTING_SWITCH=true
SMALL_MODEL_URL=http://127.0.0.1:8103/v1
SMALL_MODEL_NAME=pangu_small
LARGE_MODEL_URL=http://127.0.0.1:8104/v1
LARGE_MODEL_NAME=pangu_large
EOF

(
    unset OPENAI_BASE_URL OPENAI_API_KEY EXAMPLE_MODEL_NAME ROUTING_MODEL_NAME
    load_env
    [ "$ROUTING_MODE" = true ] || exit 1
    [ "$SMALL_MODEL_URL" = "http://127.0.0.1:8103/v1" ] || exit 1
    [ "$SMALL_MODEL_NAME" = "pangu_small" ] || exit 1
    [ "$LARGE_MODEL_URL" = "http://127.0.0.1:8104/v1" ] || exit 1
    [ "$LARGE_MODEL_NAME" = "pangu_large" ] || exit 1
) || fail "routing-only .env was rejected by the DDv2 CI"

grep -q 'Model routing: SKIPPED (MODEL_ROUTING_SWITCH=false)' "$CI_SCRIPT" ||
    fail "routing-disabled runs are not reported as skipped"
grep -q 'all \$required_feature_count enabled runtime features have strict evidence' "$CI_SCRIPT" ||
    fail "runtime feature summary is not mode-aware"

# A failing step must make main return non-zero.
create_http_tester() { :; }
load_env() { :; }
validate_config() { return 1; }
test_vllm_service() { return 0; }
test_vllm_inference() { return 0; }
start_mcp_server() { return 0; }
start_agent_server() { return 0; }
test_agent_with_search_tool() { return 0; }
test_model_routing() { return 0; }
log_info() { :; }
log_error() { :; }

main
main_status=$?
[ "$main_status" -eq 1 ] || fail "main returned $main_status for one failed step"

echo "PASS: DDv2 CI contract"
