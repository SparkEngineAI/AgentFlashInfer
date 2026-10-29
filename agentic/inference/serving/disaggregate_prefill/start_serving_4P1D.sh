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
# export PREFILL_NODE_IPS="10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4"
# export DECODE_NODE_IPS="10.0.0.5 10.0.0.6 10.0.0.7 10.0.0.8"

export P_INSTANCE_NUM=4
export D_INSTANCE_NUM=1
export NPU_NUM_PER_P_INSTANCE=16
export NPU_NUM_PER_D_INSTANCE=64
export P_TP_SIZE=8
export D_TP_SIZE=1

export PREFILL_MAX_MODEL_LEN=${PREFILL_MAX_MODEL_LEN:=17000}
export PREFILL_MAX_NUM_BATCHED_TOKENS=${PREFILL_MAX_NUM_BATCHED_TOKENS:=16384}
export PREFILL_GRAPH_BATCH=${PREFILL_GRAPH_BATCH:=4}
export PREFILL_GPU_MEMORY_UTILIZATION=${PREFILL_GPU_MEMORY_UTILIZATION:=0.9}

export DECODE_MAX_MODEL_LEN=${DECODE_MAX_MODEL_LEN:=32768}
export DECODE_MAX_NUM_BATCHED_TOKENS=${DECODE_MAX_NUM_BATCHED_TOKENS:=256}
export DECODE_GRAPH_BATCH=${DECODE_GRAPH_BATCH:=18}
export DECODE_GPU_MEMORY_UTILIZATION=${DECODE_GPU_MEMORY_UTILIZATION:=0.9}
export DECODE_HCCL_BUFFSIZE=${DECODE_HCCL_BUFFSIZE:=1024}

export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:="pangu_ultra_moe"}
export PROXY_PORT="8000"
export PROXY_HOST=`echo $DECODE_NODE_IPS | cut -d' ' -f1`
export NPU_NUM=$NPU_NUM
export NIC_NAME=$nic_name
export USE_SPEC=1
export USE_TOOL=0
export USE_REASONING=1

bash serving/disaggregate_prefill/start_serving_xPyD_template.sh
