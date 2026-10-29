# 获取脚本所在目录（支持被 symlink 调用）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 上移一层到父目录
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )/../"
export PROJECT_DIR=$PROJECT_DIR

echo "Project dir: ${PROJECT_DIR}"

cd $PROJECT_DIR

source $PROJECT_DIR/serving/base.sh
source /usr/local/Ascend/atb/set_env.sh 

export P_INSTANCE_NUM=${P_INSTANCE_NUM:=1}
export D_INSTANCE_NUM=${D_INSTANCE_NUM:=1}
export NPU_NUM_PER_P_INSTANCE=${NPU_NUM_PER_P_INSTANCE:=32}
export NPU_NUM_PER_D_INSTANCE=${NPU_NUM_PER_D_INSTANCE:=32}

if [[ -n $MA_JOB_DIR ]]; then
    export NPU_NUM=$MA_NUM_GPUS
    export OUTPUT_TEXT_DIR=$MA_LOG_DIR
    echo "Try to use getent hosts to get NODE_IP_ARRAY"
    readarray -t NODE_IP_ARRAY < <(
        for i in $(seq 0 $VC_WORKER_NUM); do
            if [[ ${NPU_NUM} -eq 8 ]]; then
                getent hosts "$MA_VJ_NAME-$MA_TASK_NAME-$i.$MA_VJ_NAME" 2>/dev/null | awk '{print $1}' | head -n1
            elif [[ ${NPU_NUM} -eq 16 ]]; then
                getent hosts "$MA_VJ_NAME-$MA_TASK_NAME-$i" 2>/dev/null | awk '{print $1}' | head -n1
            fi
        done
    )
    if [[ ${#NODE_IP_ARRAY[@]} -ne $VC_WORKER_NUM ]]; then
        echo "Try to use txt to get NODE_IP_ARRAY"
        LOCAL_IP=$(hostname -I | cut -d' ' -f1)
        # 从$HOSTNAME提取节点号
        NODE_RANK=$(echo "$HOSTNAME" | awk -F'-' '{print $NF}')
        # 构建输出目录
        NODE_IP_DIR="$(dirname "$OUTPUT_PATH")/${MA_HJ_NAME}/node_ips/"
        mkdir -p "$NODE_IP_DIR"
        # 写入当前节点的 ip 信息
        rm -f ${NODE_IP_DIR}/node${NODE_RANK}_ip.txt
        echo "${NODE_RANK} ${LOCAL_IP}" > "${NODE_IP_DIR}/node${NODE_RANK}_ip.txt"
        echo "Node ${NODE_RANK} wrote IP: ${LOCAL_IP} to ${NODE_IP_DIR}/node${NODE_RANK}_ip.txt"
        # 等待所有节点IP文件写完
        while [ "$(ls -1 ${NODE_IP_DIR}/node*_ip.txt 2>/dev/null | wc -l)" -lt "$VC_WORKER_NUM" ]; do
            echo "Waiting for all $VC_WORKER_NUM nodes to write IPs..."
            sleep 5
        done
        # 从节点IP文件读所有节点IP
        echo "All $VC_WORKER_NUM node ip files detected."
        NODE_IP_ARRAY=()
        for ((i=0; i<VC_WORKER_NUM; i++)); do
            ip=$(awk '{print $2}' "${NODE_IP_DIR}/node${i}_ip.txt")
            NODE_IP_ARRAY+=("$ip")
        done
    fi
    PREFILL_NODE_COUNT=$(($P_INSTANCE_NUM*$NPU_NUM_PER_P_INSTANCE/NPU_NUM))
    export PREFILL_NODE_IPS="${NODE_IP_ARRAY[@]:0:$PREFILL_NODE_COUNT}"
    export DECODE_NODE_IPS="${NODE_IP_ARRAY[@]:$PREFILL_NODE_COUNT}"
    echo "PREFILL_NODE_IPS=${PREFILL_NODE_IPS}"
    echo "DECODE_NODE_IPS=${DECODE_NODE_IPS}"
fi

export P_TP_SIZE=${P_TP_SIZE:=8}
export D_TP_SIZE=${D_TP_SIZE:=8}
export PREFILL_MAX_MODEL_LEN=${PREFILL_MAX_MODEL_LEN:=$((1024*100+1))}
export DECODE_MAX_MODEL_LEN=${DECODE_MAX_MODEL_LEN:=$((1024*128))}
export PREFILL_MAX_NUM_BATCHED_TOKENS=${PREFILL_MAX_NUM_BATCHED_TOKENS:=$((1024*2))}
export DECODE_MAX_NUM_BATCHED_TOKENS=${DECODE_MAX_NUM_BATCHED_TOKENS:=1}
export PREFILL_GRAPH_BATCH=${PREFILL_GRAPH_BATCH:=1}
export DECODE_GRAPH_BATCH=${DECODE_GRAPH_BATCH:=1}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:="pangu_ultra_moe"}
export PROXY_PORT="8000"
export PROXY_HOST=`echo $DECODE_NODE_IPS | cut -d' ' -f1`
export NPU_NUM=$NPU_NUM
export NIC_NAME=$nic_name
export USE_SPEC=${USE_SPEC:=0}
export SPEC_SAMPLE_GREEDY_COMPARE=${SPEC_SAMPLE_GREEDY_COMPARE:=0}

bash serving/disaggregate_prefill/start_serving_xPyD_template.sh
