# 获取脚本所在目录（支持被 symlink 调用）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 上移一层到父目录
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )/../"
export PROJECT_DIR=$PROJECT_DIR

echo "Project dir: ${PROJECT_DIR}"

cd $PROJECT_DIR

source $PROJECT_DIR/serving/base.sh
export LOAD_CKPT_DIR="${LOAD_CKPT_DIR:?Please set LOAD_CKPT_DIR to your model checkpoint directory}"

# 推理池
export PREFILL_NODE_IPS=$(python3 ${PROJECT_DIR}/serving/roma_tools/get_node_ip_list.py --mode=prefill)
export DECODE_NODE_IPS=$(python3 ${PROJECT_DIR}/serving/roma_tools/get_node_ip_list.py --mode=decode)
# 训练池/裸机参考
# Example addresses; replace with your actual node IPs.
# export PREFILL_NODE_IPS="10.0.0.1 10.0.0.2"
# export DECODE_NODE_IPS="10.0.0.5 10.0.0.6 10.0.0.7 10.0.0.8"

export P_INSTANCE_NUM=1
export D_INSTANCE_NUM=1
export NPU_NUM_PER_P_INSTANCE=32
export NPU_NUM_PER_D_INSTANCE=64
export P_TP_SIZE=8
export D_TP_SIZE=8
export PREFILL_MAX_MODEL_LEN=$((24*1024+1))
export DECODE_MAX_MODEL_LEN=$((40*1024))
export PREFILL_MAX_NUM_BATCHED_TOKENS=$((4*1024))
export DECODE_MAX_NUM_BATCHED_TOKENS=256
export PREFILL_GRAPH_BATCH=1
export DECODE_GRAPH_BATCH=8
export PREFILL_GPU_MEMORY_UTILIZATION=0.91
export DECODE_GPU_MEMORY_UTILIZATION=0.95
export SERVED_MODEL_NAME="pangu_ultra_moe"
export PROXY_PORT="8000"
export PROXY_HOST=`echo $DECODE_NODE_IPS | cut -d' ' -f1`
export NPU_NUM=$NPU_NUM
export NIC_NAME=$nic_name
export USE_SPEC=1
export NUM_SPEC=1
export USE_REASONING=1
export ENABLE_PREFILL_OPTIMIZATIONS=false

bash serving/disaggregate_prefill/start_serving_xPyD_template.sh
