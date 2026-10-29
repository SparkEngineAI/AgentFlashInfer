
# 获取脚本所在目录（支持被 symlink 调用）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 获取Project目录
PROJECT_DIR=$(pwd | sed 's|\(.*Pangu_infer\).*|\1|')
PROJECT_DIR="${PROJECT_DIR}/agentic/inference"

echo "Project dir: ${PROJECT_DIR}"

cd $PROJECT_DIR

source $PROJECT_DIR/serving/base.sh
# export LOAD_CKPT_DIR='/path/to/deepdiver_v2_38b/huggingface'

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

# logging level of adaptor manager: ERROR, INFO, DEBUG
export ADAPTOR_LOG_LEVEL="ERROR"

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/pangu_infer:${PYTHONPATH}"

if [ -n "$CHAT_TEMPLATE_PATH" ]; then
    chat_template="--chat_template ${CHAT_TEMPLATE_PATH}"
fi

if [[ ! -d ${OUTPUT_TEXT_DIR} ]]; then
    mkdir -p ${OUTPUT_TEXT_DIR}
fi

# 思考截停总开关
export REASONER_EARLY_THINK_STOPPING_ENABLED=${REASONER_EARLY_THINK_STOPPING_ENABLED:="1"}
# 限制最长思考长度
export REASONER_EARLY_THINK_STOPPING_STEP=${REASONER_EARLY_THINK_STOPPING_STEP:=28672}
export REASONER_EARLY_THINK_STOPPING_STRING="[165576]"
export REASONER_EARLY_THINK_STOPPING_TAGS="[165576]"
# 重复输出检测与截断
export REASONER_EARLY_THINK_STOPPING_REPEAT_DETECTION=${REASONER_EARLY_THINK_STOPPING_REPEAT_DETECTION:="1"}
export REASONER_EARLY_THINK_STOPPING_STRING_FOR_REPEATED_OUTPUT=$'\n\nI think I already thought for a long time, and I should be able to deduce the final answer from the content above.[unused17]'
# 基于verifier小模型的overthink/underthink检测与压缩
export REASONER_EARLY_THINK_STOPPING_ENABLE_COMPRESSION="0"

# ================参数解析====================
# 解析模型权重参数
LOAD_CKPT_DIR="${LOAD_CKPT_DIR:-$1}"
if [ -z "$LOAD_CKPT_DIR" ] || [ ! -d "$LOAD_CKPT_DIR" ]; then
  echo "[ERROR] 模型路径参数缺失或无效"
  exit 1
fi

# 解析量化参数
ENABLE_QUANTIZATION=""
for arg in "$@"; do
  if [[ "$arg" == "--quant" ]]; then
    ENABLE_QUANTIZATION="--quantization ascend"
  fi
done
# ============================================

python ${PROJECT_DIR}/serving/launcher.py serve \
  ${LOAD_CKPT_DIR} \
  --gpu-memory-utilization 0.95 \
  --block-size ${BLOCK_SIZE:=128} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:=4} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:=$((32*1024))} \
  --max-model-len ${MAX_MODEL_LEN:=131072} \
  --trust-remote-code \
  --port 8000 \
  --host 0.0.0.0 \
  --distributed-executor-backend mp \
  --max-num-seqs ${MAX_NUM_SEQS:=32} \
  --tokenizer-mode "slow" \
  --dtype bfloat16 \
  --served-model-name ${SERVED_MODEL_NAME:=pangu} \
  $reasoning \
  $tools \
  $chat_template \
  $ENABLE_QUANTIZATION \
  2>&1 | tee ${OUTPUT_TEXT_DIR:-./}/inference.log
