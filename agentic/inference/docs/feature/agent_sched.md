# 调度优化方案

vLLM接收用户请求后会放入调度内的请求队列参与batch组建，默认支持FCFS和Priority。
上述调度方案针对旧的推理和对话场景设计，并未针对Agentic AI场景下，长短序列混合交织，KV Cache容量紧张的场景进行针对性设计。
针对Agentic AI场景，我们提出了如下调度优化方案。

**注意**：本功能开关使用`--additional_config`传入相关配置。

## 方案概览

本方案包含多个子方案
- 最短任务优先（Short Job First, SJF）
- 峰值切分和尾部优化（Peak Split and Long Prefill Optimize）
- 延迟预填充（Prefill Delay）

### 最短任务优先（Short Job First, SJF）

支持SJF策略能够优先处理短作业请求，避免长序列请求调度处理时block导致吞吐下降的情况。
vLLM调度中优先完成running请求（即已经进行prefill计算的请求），其次调度waiting请求（即未有过计算的请求）。下图简述不同请求队列策略对长短请求的调度情况，不同长度的请求先后到达vLLM且单次调度的token budget为4。
FCFS根据到达先后在第一轮调度时完成请求R1的Prefill计算后，第二、三轮持续进行R1的Decode计算及R2的Prefill计算。直至第四轮仅有2条请求完成Prefill计算。
SJF根据请求长度以R4,R3,R1,R2的顺序完成Prefill计算。在第一轮调度中即完成2条请求Prefill计算，在第三轮调度完成3条请求Prefill计算，以较少轮次将Prefill请求转换为Decode请求。

![SJF](../images/feature/agent_sched_sjf.png "SJF")

针对SJF请求队列策略，考虑现实推理系统面临持续请求涌入，易导致长请求优先级较低的情况，SJF策略生效的同时添加队列等待时间作为调度的考量因素，减缓长序列请求饥饿的情况。
作为请求队列的排序策略，队列需要在外部调用加入/弹出请求时，根据SJF的思想插入/弹出最符合短作业优先的请求。如上一部分所述，队列内部排序不仅考虑请求输入prompt长度，而且考虑请求在队列等待时间，在短作业优先的基础上避免长序列请求的饥饿情况。同时，与FCFS和Priority不同，当推理系统中无新请求涌入，仅处理waiting队列的请求时，前两者队列内部请求相对位置不会变化，而SJF会根据队列等待时间而动态调整内部请求相对位置。
因此对SJF的实现有两方面的需求：
1.	实现Scorer能够将请求长度、等待时间处理成归一得分，得分和请求长度、得分和等待时间的相关性相反，即优先处理请求长度短、等待时间长的请求。
2.	实现基于SJF的请求等待队列，在请求加入和弹出时均确保SJF的规则。

#### 启用方式
修改启动脚本中的 **RAW_CFG** 变量:

- agent_scheduler.scheduling_policy = "sjf"

```
# Write additional config in a more friendly way
read -r -d '' RAW_CFG <<'JSON'
{
  "enable_hybrid_graph_mode": true,
  "reasoner_sampler": {
    "enabled": true
  },
  "agent_scheduler": {
    "scheduling_policy": "sjf"
  }
}
JSON
```

修改启动脚本中的 **command** 变量

- --max-num-batched-tokens  ${MAX_NUM_BATCHED_TOKENS:=4096} \
- --max-model-len ${MAX_MODEL_LEN:=131072} \

```
command="python serving/launcher.py serve \
   ${PRECHECKPOINT_PATH} \
  --gpu-memory-utilization 0.95 \
  --block-size ${BLOCK_SIZE:=128} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:=2} \
  --max-num-batched-tokens  ${MAX_NUM_BATCHED_TOKENS:=4096} \
  --max-model-len ${MAX_MODEL_LEN:=131072} \
  --trust-remote-code \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 4567 \
  --port 8300 \
  $reasoning \
  $tools \
  $chat_template \
  --served-model-name "pangu_auto" \
  --additional-config "$ADD_CFG" "
```


### 峰值切分和尾部优化（Peak Split and Long Prefill Optimize）

Chunked prefill在vLLM V1 scheduler中是一项较为重要的调度特性，能够充分利用每轮token的预算完成部分Prefill请求。Splitfuse + chunked prefill场景下，当系统请求数在上升阶段较少时，1条prefill请求需要经历多次切分/调度才能转换成Decode请求，容易造成该阶段batchsize较低，吞吐降低的影响。

峰值切分策略尝试在running请求数小于单轮调度最大可运行请求数的某个阈值时时，关闭chunked prefill进行混跑，快速将Prefill请求转为running请求，如下图第二轮调度；而当系统存在一定数量running请求时，chunked prefill重新开启，以确保不影响Decode性能，如下图第N轮调度。

![PeakSplit](../images/feature/agent_sched_peak_split.png "PeakSplit")

同时针对chunked prefill对请求进行多次切分完成prefill计算，容易造成尾部请求长度较短的情况，提出尾部优化策略。尾部优化在切分的调度过程中，对尾部token进行动态切分。如果尾部token小于chunk size（切分长度）,即当前序列长度小于2*chunk size，这一部分的prefill将不进行切分。该策略相较于现有切分，能够在较小prefill代价下，提早一轮将Prefill请求转为Decode请求。

![LongPrefillOptimize](../images/feature/agent_sched_long_prefill_optimize.png "LongPrefillOptimize")

#### 启用方式
修改启动脚本中的 **RAW_CFG** 变量

- agent_scheduler.long_prefill_token_threshold = 2000
- agent_scheduler.peak_split = true

```
# Write additional config in a more friendly way
read -r -d '' RAW_CFG <<'JSON'
{
  "enable_hybrid_graph_mode": true,
  "reasoner_sampler": {
    "enabled": true
  },
  "agent_scheduler": {
    "scheduling_policy": "fcfs",
    "long_prefill_token_threshold": 4096,
    "peak_split": true,
    "peak_split_factor": 0.5
  }
}
JSON
```

(Optional) If you are using vllm_ascend with a 0.9.1 version(without any version suffix), please:
- find **editable project location** of vllm_ascend by ```pip show vllm_ascend```
- check /path/to/editable/project/location/vllm-ascend/vllm-ascend-0.9.1/vllm_ascend/platform.py
- comment out the if-block below
```
if model_config is not None and not model_config.use_mla:
	logger.info(
		"Non-MLA LLMs forcibly disable the chunked prefill feature,"
		"as the performance of operators supporting this feature "
		"functionality is currently suboptimal.")
	if not envs.VLLM_USE_V1:
		scheduler_config.enable_chunked_prefill = False
		scheduler_config.chunked_prefill_enabled = False
	# if envs.VLLM_USE_V1 and \
	#     not model_config.is_multimodal_model and \
	#     decoding_config.backend == "auto" and \
	#     not scheduler_config.delay_factor > 0 and \
	#     not scheduler_config.send_delta_data and \
	#     scheduler_config.policy == "fcfs" and \
	#     scheduler_config.num_scheduler_steps == 1:
	#     scheduler_config.enable_chunked_prefill = False
	#     scheduler_config.chunked_prefill_enabled = False
	#     ascend_scheduler_config.enabled = True
	#     if hasattr(ascend_scheduler_config, "enable_chunked_prefill"):
	#         ascend_scheduler_config.enable_chunked_prefill = False
```

修改启动脚本中的 **command** 变量

- --max-num-batched-tokens  ${MAX_NUM_BATCHED_TOKENS:=4096} \
- --max-model-len ${MAX_MODEL_LEN:=131072} \
- --max-num-seqs 20

```
command="python serving/launcher.py serve \
   ${PRECHECKPOINT_PATH} \
  --gpu-memory-utilization 0.95 \
  --block-size ${BLOCK_SIZE:=128} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:=2} \
  --max-num-batched-tokens  ${MAX_NUM_BATCHED_TOKENS:=4096} \
  --max-model-len ${MAX_MODEL_LEN:=131072} \
  --trust-remote-code \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 4567 \
  --port 8300 \
  --max-num-seqs 20 \
  $reasoning \
  $tools \
  $chat_template \
  --served-model-name "pangu_auto" \
  --additional-config "$ADD_CFG" "
```

### 延迟预填充（Prefill Delay）

该策略通过动态调整调度阈值和引入延迟机制，来避免在等待队列较短时频繁触发开销较大的 Prefill 操作，从而提高整体吞吐量和资源利用率。核心思想是：当运行中的任务数量足够多，且等待队列中的请求数量较少时，暂时延迟新的 Prefill 调度，以集中资源完成当前的 Decode 阶段。

1.  **动态阈值调整：**
    * 该策略会根据当前的**等待队列长度** (`current_waiting`) 动态调整两个关键参数：
        * **最小调度序列数** (`min_num_seqs_to_schedule`): 运行中任务数达到该值时，才考虑启用延迟逻辑（范围 $2$ 到 $6$）。等待队列越长，该值越大，倾向于更快地进行调度。
        * **最大等待轮次** (`max_num_schedule_to_wait`): 允许跳过调度操作的最大轮次计数（范围 $6$ 到 $10$）。等待队列越长，该值越小，减少延迟调度的机会。

2.  **延迟调度判断 (`should_skip_scheduling`):**
    * 仅当满足以下条件时，系统才考虑**跳过**当前调度周期（即**延迟 Prefill**）：
        * 等待队列非空 (`current_waiting > 0`)。
        * 运行中的任务数量达到动态调整的**最小调度序列数** (`num_running_tasks >= min_num_seqs_to_schedule`)。
        * 等待队列中的请求数量**低于**动态调整的**最小调度序列数** (`current_waiting < min_num_seqs_to_schedule`)。
        * 当前已累计跳过的调度轮次**未达到**动态调整的**最大等待轮次** (`self.last_accessed < max_num_schedule_to_wait`)。
    * 如果决定跳过，`self.last_accessed` 计数器会递增。

3.  **防止请求饥饿（Aging 机制）：**
    * 为了保证请求的公平性，策略加入了**请求排队时长检测**：如果等待队列中**最旧**的请求等待时间超过一个固定的阈值（代码中设置为 $\mathbf{0.5}$ 秒），则**强制立即调度**，清空延迟计数，避免长时延。

4.  **智能计数器管理：**
    * 当跳过轮次达到 **最大等待轮次** 时，`self.last_accessed` 不会直接重置为 $\mathbf{0}$，而是**阶梯式递减**（`max(0, self.last_accessed - 5)`），这可以更平滑地调整调度节奏，而不是突然完全放开 Prefill。

这项优化通过动态平衡 Prefill 带来的启动开销与 Decode 带来的低延迟需求，在系统负载适中时**“囤积”少量等待请求，凑够一定批量后再进行 Prefill，从而优化了 Batching 效果，有效提升了整体的模型推理吞吐量**。

#### 启用方式
修改启动脚本中的 **RAW_CFG** 变量

- agent_scheduler.prefill_schedule.enabled = true

```
# Write additional config in a more friendly way
read -r -d '' RAW_CFG <<'JSON'
{
  "enable_hybrid_graph_mode": true,
  "reasoner_sampler": {
    "enabled": true
  },
  "agent_scheduler": {
    "scheduling_policy": "fcfs",
    "long_prefill_token_threshold": 0,
    "peak_split": false,
    "prefill_schedule": {
      "enabled": true
    }
  }
}
JSON
```
