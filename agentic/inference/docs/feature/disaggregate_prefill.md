# PD分离特性

## 启动脚本

参考`Pangu_infer/serving/disaggregate_prefill/`

## 启动配置

以`Pangu_infer/serving/disaggregate_prefill/start_serving_1P1D_bf16_24k_16k.sh`为例，启动时需要设置一些环境变量：

```shell
# 以下为示例地址，请替换为实际节点 IP；多个地址用空格分隔。
# Prefill节点IP
export PREFILL_NODE_IPS="10.0.0.1 10.0.0.2"
# Decode节点IP
export DECODE_NODE_IPS="10.0.0.5 10.0.0.6 10.0.0.7 10.0.0.8"

# Prefill节点数
export P_INSTANCE_NUM=1
# Decode节点数
export D_INSTANCE_NUM=1

# Prefill单实例的卡/die数
export NPU_NUM_PER_P_INSTANCE=32
# Decode单实例的卡/die数
export NPU_NUM_PER_D_INSTANCE=64

# Prefill单实例的TP Size
export P_TP_SIZE=8
# Decode单实例的TP Size
export D_TP_SIZE=2

# 设置为最大输入长度+1
export PREFILL_MAX_MODEL_LEN=$((24*1024+1))
# 设置为最大输入长度+输出长度
export DECODE_MAX_MODEL_LEN=$((40*1024))

# 使用chunked prefill时的chunk size，需小于或等于最大输入长度
export PREFILL_MAX_NUM_BATCHED_TOKENS=$((4*1024))
# 由于vllm-ascend会修改graph_batch_size，而且需要DECODE_MAX_NUM_BATCHED_TOKENS大于或等于graph_batch_size，这里设置DECODE_MAX_NUM_BATCHED_TOKENS为一个较大的数值
export DECODE_MAX_NUM_BATCHED_TOKENS=256

# Prefill的--max-num-seqs，也就是推理时每个step的最大序列数
export PREFILL_GRAPH_BATCH=1
# Decode的--max-num-seqs，也就是推理时每个step的最大序列数，vllm-ascend会据此计算合适的入图的graph_batch_size
export DECODE_GRAPH_BATCH=5

# Prefill的--gpu-memory-utilization，默认为0.9
export PREFILL_GPU_MEMORY_UTILIZATION=0.91
# Decode的--gpu-memory-utilization，默认为0.95
export DECODE_GPU_MEMORY_UTILIZATION=0.95

# Prefill的HCCL_BUFFSIZE，默认为200
export PREFILL_HCCL_BUFFSIZE=200
# Decode的HCCL_BUFFSIZE，默认为200
export DECODE_HCCL_BUFFSIZE=200

# vllm服务的--served-model-name
export SERVED_MODEL_NAME="pangu_ultra_moe"
# 代理服务端口
export PROXY_PORT="8000"
# 代理服务IP，这里使用decode的首节点作为代理节点，
export PROXY_HOST=`echo $DECODE_NODE_IPS | cut -d' ' -f1`

# 是否开启投机推理，0关闭，1开启
export USE_SPEC=0

# 是否开启reasoning parser，0关闭，1开启
export USE_REASONING=1
```
