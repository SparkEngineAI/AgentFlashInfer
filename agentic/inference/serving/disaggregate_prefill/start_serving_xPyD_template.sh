NODE_IP_LIST=`[[ $PREFILL_NODE_IPS == $DECODE_NODE_IPS ]] && echo $PREFILL_NODE_IPS || echo "$PREFILL_NODE_IPS $DECODE_NODE_IPS"`
P_INSTANCE_NUM=${P_INSTANCE_NUM:=4}
D_INSTANCE_NUM=${D_INSTANCE_NUM:=1}
NPU_NUM_PER_P_INSTANCE=${NPU_NUM_PER_P_INSTANCE:=16}
NPU_NUM_PER_D_INSTANCE=${NPU_NUM_PER_D_INSTANCE:=64}
P_TP_SIZE=${P_TP_SIZE:=8}
D_TP_SIZE=${D_TP_SIZE:=1}
PREFILL_MAX_MODEL_LEN=${PREFILL_MAX_MODEL_LEN:=4096}
DECODE_MAX_MODEL_LEN=${DECODE_MAX_MODEL_LEN:=4096}
DECODE_MAX_NUM_BATCHED_TOKENS=${DECODE_MAX_NUM_BATCHED_TOKENS:=256}
PREFILL_MAX_NUM_BATCHED_TOKENS=${PREFILL_MAX_NUM_BATCHED_TOKENS:=4096}
PREFILL_GRAPH_BATCH=${PREFILL_GRAPH_BATCH:=1}
DECODE_GRAPH_BATCH=${DECODE_GRAPH_BATCH:=28}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:="pangu_ultra_moe"}
PROXY_PORT=${PROXY_PORT:="8080"}
PROXY_HOST=${PROXY_HOST:="0.0.0.0"}
NIC_NAME=${NIC_NAME:="enp23s0f3"}
DP_PORT=${DP_PORT:=13395}
ENGINE_PORT=${ENGINE_PORT:=9000}
OUTPUT_TEXT_DIR=${OUTPUT_TEXT_DIR:=${PROJECT_DIR}/outputs}
USE_SPEC=${USE_SPEC:=0}
USE_TOOL=${USE_TOOL:=0}
ENABLE_PREFILL_OPTIMIZATIONS=${ENABLE_PREFILL_OPTIMIZATIONS:=true}
ENABLE_MULTISTREAM_MOE=${ENABLE_MULTISTREAM_MOE:=true}
ENABLE_SUPER_KERNEL=${ENABLE_SUPER_KERNEL:=true}
D_LMHEAD_TP_SIZE=${D_LMHEAD_TP_SIZE:=$NPU_NUM}
ATTN_PREFETCH_SIZE=${ATTN_PREFETCH_SIZE:=96}
W13_PREFETCH_SIZE=${W13_PREFETCH_SIZE:=50}
W2_PREFETCH_SIZE=${W2_PREFETCH_SIZE:=0}
IS_A3=${IS_A3:=$(($NPU_NUM==16))}
ENABLE_KV_NZ=${ENABLE_KV_NZ:=false}

if [[ ! -d ${OUTPUT_TEXT_DIR} ]]; then
    mkdir -p ${OUTPUT_TEXT_DIR}
fi

echo "PREFILL_NODE_IPS: $PREFILL_NODE_IPS"
echo "DECODE_NODE_IPS $DECODE_NODE_IPS"
echo "NODE_IP_LIST $NODE_IP_LIST"
echo "P_INSTANCE_NUM $P_INSTANCE_NUM"
echo "D_INSTANCE_NUM $D_INSTANCE_NUM"
echo "NPU_NUM_PER_P_INSTANCE $NPU_NUM_PER_P_INSTANCE"
echo "NPU_NUM_PER_D_INSTANCE $NPU_NUM_PER_D_INSTANCE"
echo "P_TP_SIZE $P_TP_SIZE"
echo "D_TP_SIZE $D_TP_SIZE"
echo "PREFILL_MAX_MODEL_LEN $PREFILL_MAX_MODEL_LEN"
echo "DECODE_MAX_MODEL_LEN $DECODE_MAX_MODEL_LEN"
echo "DECODE_MAX_NUM_BATCHED_TOKENS $DECODE_MAX_NUM_BATCHED_TOKENS"
echo "PREFILL_MAX_NUM_BATCHED_TOKENS $PREFILL_MAX_NUM_BATCHED_TOKENS"
echo "PREFILL_GRAPH_BATCH $PREFILL_GRAPH_BATCH"
echo "DECODE_GRAPH_BATCH $DECODE_GRAPH_BATCH"
echo "SERVED_MODEL_NAME $SERVED_MODEL_NAME"
echo "PROXY_PORT $PROXY_PORT"
echo "PROXY_HOST $PROXY_HOST"
echo "NPU_NUM $NPU_NUM"
echo "SPEC_SAMPLE_GREEDY_COMPARE $SPEC_SAMPLE_GREEDY_COMPARE"
echo "ENABLE_KV_NZ $ENABLE_KV_NZ"
NODE_NUM_PER_P_INSTANCE=$(($NPU_NUM_PER_P_INSTANCE/$NPU_NUM))
NODE_NUM_PER_D_INSTANCE=$(($NPU_NUM_PER_D_INSTANCE/$NPU_NUM))
P_NPU_NUM=$(($P_INSTANCE_NUM*$NPU_NUM_PER_P_INSTANCE))
D_NPU_NUM=$(($D_INSTANCE_NUM*$NPU_NUM_PER_D_INSTANCE))

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/pangu_infer:${PYTHONPATH}"
export RANKTABLE_PATH=$PROJECT_DIR/ranktable.json
cd serving/disaggregate_prefill/gen_ranktable/
rm $RANKTABLE_PATH
bash ./gen_ranktable.sh --ips $NODE_IP_LIST --npus-per-node $NPU_NUM --network-card-name $NIC_NAME --prefill-device-cnt $P_NPU_NUM --decode-device-cnt $D_NPU_NUM
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=$RANKTABLE_PATH
cd $PROJECT_DIR
export TORCHAIR_CACHE_HOME=$OUTPUT_TEXT_DIR/torchair_cache/
source /usr/local/Ascend/atb/set_env.sh

LOCAL_IP=`hostname -I | cut -d' ' -f1`
if echo "$PREFILL_NODE_IPS" | grep -w "$LOCAL_IP" ; then
    TP_SIZE=$P_TP_SIZE
    NODE_NUM_PER_INSTANCE=$NODE_NUM_PER_P_INSTANCE
    GLOBAL_NODE_RANK=$(echo "${PREFILL_NODE_IPS%${LOCAL_IP}*}" | tr -cd ' ' | wc -c)
    NODE_IPS=$PREFILL_NODE_IPS
    MAX_MODEL_LEN=$PREFILL_MAX_MODEL_LEN
    MAX_NUM_BATCHED_TOKENS=$PREFILL_MAX_NUM_BATCHED_TOKENS
    GRAPH_BATCH=$PREFILL_GRAPH_BATCH
    if [ $IS_A3 = 0 ]; then
        #for A2
        ADD_CFG='{"ascend_scheduler_config":{"enabled":false},"chunked_prefill_for_mla":false}'
    else
        #for A3
        ADD_CFG='{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":false,"enable_kv_nz":'$ENABLE_KV_NZ'},"enable_weight_nz_layout":true,"enable_prefill_optimizations":'$ENABLE_PREFILL_OPTIMIZATIONS',"chunked_prefill_for_mla":true}'
    fi
    ADD_CFG=${PREFILL_ADD_CFG:=$ADD_CFG}
    KV_ROLE="kv_producer"
    EAGER='--enforce-eager'
    GPU_MEMORY_UTILIZATION=${PREFILL_GPU_MEMORY_UTILIZATION:=0.9}
    export HCCL_BUFFSIZE=${PREFILL_HCCL_BUFFSIZE:=200}
else
    rm -rf $TORCHAIR_CACHE_HOME
    TP_SIZE=$D_TP_SIZE
    GLOBAL_NODE_RANK=$(echo "${DECODE_NODE_IPS%${LOCAL_IP}*}" | tr -cd ' ' | wc -c)
    NODE_NUM_PER_INSTANCE=$NODE_NUM_PER_D_INSTANCE
    NODE_IPS=$DECODE_NODE_IPS
    MAX_MODEL_LEN=$DECODE_MAX_MODEL_LEN
    MAX_NUM_BATCHED_TOKENS=$DECODE_MAX_NUM_BATCHED_TOKENS
    GRAPH_BATCH=$DECODE_GRAPH_BATCH
    if [ $IS_A3 = 0 ]; then
        #for A2
        ADD_CFG='{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":true,"enable_kv_nz":false,"graph_batch_sizes":['$GRAPH_BATCH']},"enable_weight_nz_layout":false,"expert_tensor_parallel_size":1}'
    else
        #for A3
        ADD_CFG='{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":true,"enable_kv_nz":'$ENABLE_KV_NZ',"enable_multistream_mla":true,"enable_multistream_moe":'$ENABLE_MULTISTREAM_MOE',"graph_batch_sizes":['$GRAPH_BATCH'],"enable_super_kernel":'$ENABLE_SUPER_KERNEL',"use_cached_graph":true},"enable_weight_nz_layout":true,"expert_tensor_parallel_size":1,"lmhead_tp_size":'$D_LMHEAD_TP_SIZE',"attn_prefetch_size":'$ATTN_PREFETCH_SIZE',"w13_prefetch_size":'$W13_PREFETCH_SIZE',"w2_prefetch_size":'$W2_PREFETCH_SIZE'}'
    fi
    ADD_CFG=${DECODE_ADD_CFG:=$ADD_CFG}
    KV_ROLE="kv_consumer"
    GPU_MEMORY_UTILIZATION=${DECODE_GPU_MEMORY_UTILIZATION:=0.95}
    export HCCL_BUFFSIZE=${DECODE_HCCL_BUFFSIZE:=200}
fi

if [[ $PROXY_HOST == $LOCAL_IP ]]; then
    # for dp proxy
    prefiller_hosts=""
    prefiller_ports=""
    P_NODE_NUM=$(($P_INSTANCE_NUM*$NODE_NUM_PER_P_INSTANCE))
    D_NODE_NUM=$(($D_INSTANCE_NUM*$NODE_NUM_PER_D_INSTANCE))
    PREFILL_DP_SIZE_LOCAL=$(($NPU_NUM/$P_TP_SIZE))
    DECODE_DP_SIZE_LOCAL=$(($NPU_NUM/$D_TP_SIZE))
    for ((i=0; i<$P_NODE_NUM; i++)); do
        IP_=$(echo "$PREFILL_NODE_IPS" | awk -v i=$(($i+1)) '{print $i}')
        PORTS_=${PREFILL_ENGINE_PORT:=$ENGINE_PORT}
        for ((j=0; j<$PREFILL_DP_SIZE_LOCAL; j++)); do
            prefiller_hosts="${prefiller_hosts} ${IP_}"
            local_ports=$(($PORTS_+$j))
            prefiller_ports="${prefiller_ports} ${local_ports}"
        done
    done
    decoder_hosts=""
    decoder_ports=""
    for ((i=0; i<$D_NODE_NUM; i++)); do
        IP_=$(echo "$DECODE_NODE_IPS" | awk -v i=$(($i+1)) '{print $i}')
        PORTS_=${DECODE_ENGINE_PORT:=$ENGINE_PORT}
        for ((j=0; j<$DECODE_DP_SIZE_LOCAL; j++)); do
            decoder_hosts="${decoder_hosts} ${IP_}"
            local_ports=$(($PORTS_+$j))
            decoder_ports="${decoder_ports} ${local_ports}"
        done
    done
    echo "prefiller_hosts $prefiller_hosts"
    echo "prefiller_ports $prefiller_ports"
    echo "decoder_hosts $decoder_hosts"
    echo "decoder_ports $decoder_ports"
    python serving/disaggregate_prefill/load_balance_proxy_server_example.py \
        --port $PROXY_PORT \
        --host $PROXY_HOST \
        --prefiller-hosts $prefiller_hosts \
        --prefiller-ports $prefiller_ports \
        --decoder-hosts $decoder_hosts \
        --decoder-ports $decoder_ports > $OUTPUT_TEXT_DIR/proxy.log 2>&1 &
fi

INSTANCE_RANK=$(($GLOBAL_NODE_RANK/$NODE_NUM_PER_INSTANCE))
NODE_RANK=$(($GLOBAL_NODE_RANK%$NODE_NUM_PER_INSTANCE))
DP_SIZE_LOCAL=$(($NPU_NUM/$TP_SIZE))
DP_SIZE=$(($DP_SIZE_LOCAL*$NODE_NUM_PER_INSTANCE))
DP_RANK_START=$(($NODE_RANK*$DP_SIZE_LOCAL))
DP_IP=$(echo "$NODE_IPS" | awk -v i=$(($INSTANCE_RANK*$NODE_NUM_PER_INSTANCE+1)) '{print $i}')
DP_LOCAL=$LOCAL_IP

export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export VLLM_DP_SIZE=$DP_SIZE
export VLLM_DP_MASTER_IP=$DP_IP
export VLLM_DP_MASTER_PORT=$DP_PORT
export VLLM_DP_SIZE_LOCAL=$DP_SIZE_LOCAL
export HCCL_IF_IP=$DP_LOCAL
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

export VLLM_USE_V1=1
export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1
export VLLM_WORKER_MULTIPROC_METHOD="fork"

export HCCL_CONNECT_TIMEOUT=7200
export VLLM_ASCEND_ENABLE_TOP_N_SIGMA=1

export VLLM_LLMDD_ABORT_REQUEST_TIMEOUT=${VLLM_LLMDD_ABORT_REQUEST_TIMEOUT:=600}

# 思考截停
export PRECHECKPOINT_PATH=${LOAD_CKPT_DIR}
export REASONER_EARLY_THINK_STOPPING_ENABLED=${REASONER_EARLY_THINK_STOPPING_ENABLED:="0"}
export REASONER_EARLY_THINK_STOPPING_ENABLE_COMPRESSION=${REASONER_EARLY_THINK_STOPPING_ENABLE_COMPRESSION:="0"}
export REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING_STOP_TOKENS=${REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING_STOP_TOKENS:=22000}
export REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING_STOP_ROUNDS=${REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING_STOP_ROUNDS:=20}
# 限制最长思考长度
export REASONER_EARLY_THINK_STOPPING_STEP=${REASONER_EARLY_THINK_STOPPING_STEP:=22000}
export REASONER_EARLY_THINK_STOPPING_STRING=${REASONER_EARLY_THINK_STOPPING_STRING:="[45982]"}
export REASONER_EARLY_THINK_STOPPING_TAGS=${REASONER_EARLY_THINK_STOPPING_TAGS:="[45982]"}
# 重复输出检测与截断
export REASONER_EARLY_THINK_STOPPING_REPEAT_DETECTION=${REASONER_EARLY_THINK_STOPPING_REPEAT_DETECTION:="0"}
export REASONER_EARLY_THINK_STOPPING_STRING_FOR_REPEATED_OUTPUT=$'\n\nI think I already thought for a long time, and I should be able to deduce the final answer from the content above.[unused17]'
# 基于verifier小模型的overthink/underthink检测与压缩
export REASONER_EARLY_THINK_STOPPING_COMPRESSION_OVERTHINKING=${REASONER_EARLY_THINK_STOPPING_COMPRESSION_OVERTHINKING:="0"}
export REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING=${REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING:="0"}
# verifier小模型的url
export REASONER_EARLY_THINK_STOPPING_COMPRESSION_SERVER_URL="${REASONER_EARLY_THINK_STOPPING_COMPRESSION_SERVER_URL:-}"

source scripts/common/set_think_compression_env.sh

if grep -E '^\s*"quantize"\s*:' $LOAD_CKPT_DIR/config.json; then
    quantization="--quantization ascend"
fi

if [[ $USE_SPEC -eq 1 ]]; then
    # for A3
    NUM_SPEC=${NUM_SPEC:=1}
    spec='--speculative-config {"num_speculative_tokens":'$NUM_SPEC',"method":"deepseek_mtp"}'
fi

if [[ -n "$CHAT_TEMPLATE_PATH" ]]; then
    chat_template="--chat_template ${CHAT_TEMPLATE_PATH}"
fi

echo "======== Environment variables: ========"
env -0 | sort -z | tr '\0' '\n'
echo "================"

for ((i=0; i<$DP_SIZE_LOCAL; i++)); do
    export VLLM_DP_RANK=$(($DP_RANK_START+$i))
    export VLLM_DP_RANK_LOCAL=$i
    export ENGINE_PORT_LOCAL=$(($ENGINE_PORT+$i))
    python serving/launcher.py serve $LOAD_CKPT_DIR \
        --host 0.0.0.0 \
        --port $ENGINE_PORT_LOCAL \
        --tensor-parallel-size $TP_SIZE \
        --enable-expert-parallel \
        --served-model-name $SERVED_MODEL_NAME \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --trust-remote-code \
        --max-num-seqs $GRAPH_BATCH \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        $EAGER \
        $quantization \
        $spec \
        $reasoning \
        $tools \
        $chat_template \
        --kv-transfer-config \
        '{
            "kv_connector": "LLMDataDistCMgrConnector",
            "kv_buffer_device": "npu",
            "kv_role": "'$KV_ROLE'",
            "kv_parallel_size": "1",
            "kv_port": "20001",
            "engine_id": "0",
            "kv_connector_module_path": "vllm_ascend.distributed.llmdatadist_c_mgr_connector"
        }' \
        --additional-config $ADD_CFG \
        $EXTRA_SERVE_ARGS \
        > $OUTPUT_TEXT_DIR/"$DP_LOCAL"_"$ENGINE_PORT_LOCAL".log 2>&1 &
done

sleep 2s
tail -f $OUTPUT_TEXT_DIR/"$DP_LOCAL"_"$ENGINE_PORT".log
