
# 获取脚本所在目录（支持被 symlink 调用）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 获取Project目录
PROJECT_DIR=$(pwd | sed 's|\(.*Pangu_infer\).*|\1|')
PROJECT_DIR="${PROJECT_DIR}/agentic/inference"

echo "Project dir: ${PROJECT_DIR}"

cd $PROJECT_DIR

source $PROJECT_DIR/serving/base.sh

# Write additional config in a more friendly way
read -r -d '' RAW_CFG <<'JSON'
{
  "enable_hybrid_graph_mode": true,
  "reasoner_sampler": {
    "enabled": true
  }
}
JSON

# compress into valid json string
ADD_CFG="$(json_compact_or_die - <<<"$RAW_CFG")"

# An example of using env to change cfg values
export CFG__additional_config__reasoner_sampler__enabled=false
export CFG__gpu_memory_utilization=0.95

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/pangu_infer:${PYTHONPATH}"


export PRECHECKPOINT_PATH=$LOAD_CKPT_DIR

# 思考截停
export REASONER_EARLY_THINK_STOPPING_ENABLED="1"
export REASONER_EARLY_THINK_STOPPING_ENABLE_COMPRESSION="1"
export REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING_STOP_TOKENS=16000
export REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING_STOP_ROUNDS=0
# 限制最长思考长度
export REASONER_EARLY_THINK_STOPPING_STEP=16000
export REASONER_EARLY_THINK_STOPPING_STRING="[151668]"
export REASONER_EARLY_THINK_STOPPING_TAGS="[151668]"
# 重复输出检测与截断
export REASONER_EARLY_THINK_STOPPING_REPEAT_DETECTION="1"
export REASONER_EARLY_THINK_STOPPING_STRING_FOR_REPEATED_OUTPUT=$'\n\nI think I already thought for a long time, and I should be able to deduce the final answer from the content above.</think>'
# 基于verifier小模型的overthink/underthink检测与压缩
export REASONER_EARLY_THINK_STOPPING_COMPRESSION_OVERTHINKING="0"
export REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING="0"
# verifier小模型的url
export REASONER_EARLY_THINK_STOPPING_COMPRESSION_SERVER_URL="${REASONER_EARLY_THINK_STOPPING_COMPRESSION_SERVER_URL:-}"
export REASONER_EARLY_THINK_STOPPING_USER_PATTERN="<|im_start|>user(.*?)\<|im_end|>"

if [ "$USE_REASONING" = "1" ]; then
    reasoning="--enable-reasoning --reasoning-parser qwen3"
fi

if [ "$USE_TOOL" = "1" ]; then
    tools="--enable-auto-tool-choice --tool-call-parser hermes"
fi
if [[ ! -d ${OUTPUT_TEXT_DIR} ]]; then
    mkdir -p ${OUTPUT_TEXT_DIR}
fi


source ${PROJECT_DIR}/serving/scripts/common/set_think_compression_env.sh

vllm serve  \
  ${PRECHECKPOINT_PATH} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --gpu-memory-utilization 0.95 \
  --block-size ${BLOCK_SIZE:=128} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:=8} \
  --max-num-batched-tokens  ${MAX_NUM_BATCHED_TOKENS:=2048} \
  --max-model-len ${MAX_MODEL_LEN:=16384} \
  --trust-remote-code \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 35678 \
  --port 8000 \
  --additional-config "$ADD_CFG" \
  $reasoning \
  $tools  | tee $OUTPUT_TEXT_DIR/inference.log


  