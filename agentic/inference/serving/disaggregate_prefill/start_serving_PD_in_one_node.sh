# 获取脚本所在目录（支持被 symlink 调用）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 上移一层到父目录
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )/../"
export PROJECT_DIR=$PROJECT_DIR
echo "Project dir: ${PROJECT_DIR}"
cd $PROJECT_DIR

source $PROJECT_DIR/serving/base.sh
export LOAD_CKPT_DIR="${LOAD_CKPT_DIR:?Please set LOAD_CKPT_DIR to your model checkpoint directory}"

# 1 A3 node, 1 prefill instance with 8 dies, 1 decode instance with 8 dies
LOCAL_IP=`hostname -I | cut -d' ' -f1`
export PREFILL_NODE_IPS=$LOCAL_IP
export DECODE_NODE_IPS=$LOCAL_IP
export NPU_NUM=8
export P_INSTANCE_NUM=1
export D_INSTANCE_NUM=1
export NPU_NUM_PER_P_INSTANCE=8
export NPU_NUM_PER_D_INSTANCE=8
export P_TP_SIZE=1
export D_TP_SIZE=1
export PREFILL_MAX_MODEL_LEN=${PREFILL_MAX_MODEL_LEN:=$((3*1024+2))}
export DECODE_MAX_MODEL_LEN=${PREFILLDECODE_MAX_MODEL_LEN_MAX_MODEL_LEN:=$((5*1024))}
export PREFILL_MAX_NUM_BATCHED_TOKENS=$((3*1024))
export DECODE_MAX_NUM_BATCHED_TOKENS=256
export PREFILL_GRAPH_BATCH=40
export DECODE_GRAPH_BATCH=40
export PREFILL_GPU_MEMORY_UTILIZATION=0.93
export DECODE_GPU_MEMORY_UTILIZATION=0.95
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:=pangu_72b}
export PROXY_HOST=`echo $DECODE_NODE_IPS | cut -d' ' -f1`
export PROXY_PORT="8000"
export NIC_NAME=$nic_name
export USE_SPEC=1
export USE_TOOL=${USE_TOOL:=1}
export USE_REASONING=${USE_REASONING:=1}
export IS_A3=1
export USE_OPENPANGU_MODELING=1
export PREFILL_ENGINE_PORT=9000
export DECODE_ENGINE_PORT=9100

if [ ! -d ${OUTPUT_TEXT_DIR} ]; then
    export OUTPUT_TEXT_DIR=${PROJECT_DIR}/output/
    mkdir -p $OUTPUT_TEXT_DIR
fi
# rm -rf $OUTPUT_TEXT_DIR/*

# proxy and prefill
export ENGINE_PORT=$PREFILL_ENGINE_PORT
export DP_PORT=13395
export GEN_RANKTABLE_MASTER_PORT=6657
export VLLM_LLMDD_RPC_PORT=5557
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup bash serving/disaggregate_prefill/start_serving_xPyD_template.sh > $OUTPUT_TEXT_DIR/start_serving_prefill.log 2>&1 &

# decode
unset PREFILL_NODE_IPS PROXY_HOST
export ENGINE_PORT=$DECODE_ENGINE_PORT
export DP_PORT=13396
export GEN_RANKTABLE_MASTER_PORT=6658
export VLLM_LLMDD_RPC_PORT=5657
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
nohup bash serving/disaggregate_prefill/start_serving_xPyD_template.sh > $OUTPUT_TEXT_DIR/start_serving_decode.log 2>&1 &

tail -f ${OUTPUT_TEXT_DIR}/start_serving_decode.log
