
# 获取脚本所在目录（支持被 symlink 调用）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 上移一层到父目录
export PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"
echo "Project dir: ${PROJECT_DIR}"
cd $PROJECT_DIR

current_datetime=$(date +'%Y-%m-%d-%H:%M:%S')
export OUTPUT_TEXT_DIR=$OUTPUT_TEXT_DIR/logs_$current_datetime
mkdir -p $OUTPUT_TEXT_DIR

MODEL_CARD=`python3 -c "
import yaml;
with open('${LOAD_CKPT_DIR}/model_info.yaml') as f:
    data = yaml.safe_load(f)
print(data['deploy_config']['model_name'])
"`

# MODEL_CARD转大写并截取VS以后的部分
MODEL_CARD=${MODEL_CARD^^}
MODEL_CARD=${MODEL_CARD#*VS}

# 使用 A2/A3 PD 启动脚本的模型名单，由部署环境提供，多个名称用空格分隔。
# 名称需与上面规范化后的 MODEL_CARD 一致；未配置时不选择此分支。
read -r -a PD_DEPLOY_LIST <<< "${PD_MODEL_CARDS:-}"
USE_PD_MODEL=0

for item in "${PD_DEPLOY_LIST[@]}"; do
    if [[ "$item" == "$MODEL_CARD" ]]; then
        USE_PD_MODEL=1
        export ENABLE_PD=${ENABLE_PD:="1P1D"}
        export ENABLE_PD=${ENABLE_PD^^}
        if [[ $ENABLE_PD =~ ([0-9]+)P([0-9]+)D ]]; then
            P_INSTANCE_NUM=${BASH_REMATCH[1]}  # 第一个括号匹配的内容
            D_INSTANCE_NUM=${BASH_REMATCH[2]}  # 第二个括号匹配的内容
        fi
        break
    fi
done

export SERVED_MODEL_NAME=pangu_auto
echo "=========================================="
echo "MODEL_CARD is $MODEL_CARD"
echo "ENABLE_PD is $ENABLE_PD"
echo "P_INSTANCE_NUM is $P_INSTANCE_NUM"
echo "D_INSTANCE_NUM is $D_INSTANCE_NUM"
echo "=========================================="

export ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/path/to/allowed/media}"
if [[ "$USE_PD_MODEL" == "1" ]]; then
    bash ${PROJECT_DIR}/serving/disaggregate_prefill/start_serving_A2A3_1P1D_bf16_100k_28k.sh

elif [[ ${MODEL_CARD} == "R_38B_5_0_128K" ]] || [[ ${MODEL_CARD} == "LLM_38B_5_0_128K" ]]; then
    if [[ ${ENABLE_PD} == "1P1D" ]]; then
        export PREFILL_MAX_MODEL_LEN=${PREFILL_MAX_MODEL_LEN:=$((88*1024))}
        export PREFILL_MAX_NUM_BATCHED_TOKENS=${PREFILL_MAX_MODEL_LEN:=$((88*1024))}
        bash ${PROJECT_DIR}/serving/disaggregate_prefill/start_serving_dense.sh
    else
        bash ${PROJECT_DIR}/colocate_prefill/pangu_38b/serving_pangu_38b.sh
    fi

elif [[ ${MODEL_CARD} == "R_7B_5_0_128K" ]] || [[ ${MODEL_CARD} == "LLM_7B_5_0_128K" ]]; then
    if [[ ${ENABLE_PD} == "1P1D" ]]; then
        export PREFILL_MAX_MODEL_LEN=${PREFILL_MAX_MODEL_LEN:=$((32*1024))}
        export PREFILL_MAX_NUM_BATCHED_TOKENS=${PREFILL_MAX_NUM_BATCHED_TOKENS:=$((32*1024))}
        export DECODE_MAX_MODEL_LEN=${DECODE_MAX_MODEL_LEN:=$((32*1024))}
        export DECODE_MAX_NUM_BATCHED_TOKENS=${DECODE_MAX_NUM_BATCHED_TOKENS:=256}
        export PREFILL_GRAPH_BATCH=${PREFILL_GRAPH_BATCH:=256}
        export DECODE_GRAPH_BATCH=${DECODE_GRAPH_BATCH:=256}
        bash ${PROJECT_DIR}/serving/disaggregate_prefill/start_serving_dense.sh
    else
        export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:=8}
        export BLOCK_SIZE=${BLOCK_SIZE:=128}
        export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:=131072}
        export MAX_MODEL_LEN=${MAX_MODEL_LEN:=131072}
        export USE_REASONING=${USE_REASONING:=1}
        export =${USE_TOOL:=1}
        bash ${PROJECT_DIR}/colocate_prefill/pangu_7b/serving_pangu_7b.sh
    fi

else
    echo "Please check MODEL_CARD !!!"   
fi
