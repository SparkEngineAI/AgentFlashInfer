# AgentFlashInfer

<p align="center">
  <img src="./figs/logo.png" alt="AgentFlashInfer logo" width="240">
</p>

<p align="center">
  <strong>High-Performance Inference Framework for Agentic AI Scenarios</strong>
</p>

<p align="center">
  <a href="./README_zh.md">中文文档</a>
</p>

<div align="center">
  <p>
    <a href="https://arxiv.org/abs/2512.18337"><img src="https://img.shields.io/badge/Paper-arXiv-f97316" alt="Paper arXiv"></a>
    <a href="https://arxiv.org/abs/2603.26034"><img src="https://img.shields.io/badge/Paper-arXiv-f97316" alt="Paper arXiv"></a>
    <img src="https://img.shields.io/badge/Features-Agent%20%7C%20Acceleration-2563eb" alt="Features">
    <img src="https://img.shields.io/badge/Backend-vLLM%20%7C%20Ascend-16a34a" alt="Backend">
  </p>
</div>

## 🔍 About AgentFlashInfer

AgentFlashInfer is a high-performance inference framework designed for Agent inference scenarios. It consists of two layers of capabilities:

- **Agent Feature Layer**: Pluggable features such as AgentCollab, Early Stop, and Async Compression, controlled independently through configuration switches and combined as needed.
- **Acceleration Patch Layer**: Extends vLLM / vLLM-Ascend through monkey-patching mechanism, providing capabilities like speculative decoding, Agent scheduling, and PD separation.

### Supported Agent Frameworks

| Agent Framework | Model Family | Description |
| --- | --- | --- |
| **DeepDiver2 Agent** | [openPangu-Embedded-7B-DeepDiver](https://huggingface.co/openpangu/openPangu-Embedded-7B-DeepDiver) | DeepDiver2 integration based on the Pangu model. |
| **Tongyi DeepResearch Agent** | Qwen | Tongyi DeepResearch integration based on Qwen-family models. |

**Benchmark Results on AgentFlashInfer Features:**

<table align="center">
  <tr align="center">
    <th><p align="center"><strong>Async Compression</strong></p></th>
    <th><p align="center"><strong>AgentCollab</strong></p></th>
    <th><p align="center"><strong>Early Stop</strong></p></th>
  </tr>
  <tr>
    <td align="center"><p align="center">
      <table>
        <thead>
          <tr><th align="left">Metric</th><th align="center">Result</th></tr>
        </thead>
        <tbody>
          <tr><td>Latency reduction</td><td align="center">6% – 42%</td></tr>
          <tr><td>Context reduction</td><td align="center">~40K → ~20K</td></tr>
        </tbody>
      </table>
    </p></td>
    <td align="center"><p align="center">
      <table>
        <thead>
          <tr><th align="left">Metric</th><th align="center">Result</th></tr>
        </thead>
        <tbody>
          <tr><td>Accuracy</td><td align="center">97% recovery</td></tr>
          <tr><td>Speedup</td><td align="center">~1.32×</td></tr>
        </tbody>
      </table>
    </p></td>
    <td align="center"><p align="center">
      <table>
        <thead>
          <tr><th align="left">Metric</th><th align="center">Result</th></tr>
        </thead>
        <tbody>
          <tr><td>Speedup</td><td align="center">~1.55×</td></tr>
          <tr><td>Latency reduction</td><td align="center">35.3%</td></tr>
          <tr><td>Std reduction</td><td align="center">237.87 → 84.51</td></tr>
        </tbody>
      </table>
    </p></td>
  </tr>
</table>

## ✨ Architecture & Features

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Application                     │
│            (DeepDiver V2 / DeepResearch / ...)           │
├─────────────────────────────────────────────────────────┤
│  Agent Feature Layer (features/)                        │
│  ┌──────────┐ ┌───────────┐ ┌────────────┐             │
│  │  Model   │ │  Early    │ │Compression │             │
│  │ Routing  │ │   Stop    │ │ (Async)    │             │
│  └──────────┘ └───────────┘ └────────────┘             │
├─────────────────────────────────────────────────────────┤
│  Acceleration Patch Layer (inference/agentflashinfer/)  │
│  ┌──────────────────┐  ┌──────────────────────────────┐ │
│  │  Patch System    │  │  Accelerators                 │ │
│  │  @register ──►   │  │  ┌────────┐ ┌─────────────┐ │ │
│  │  vLLM / Ascend   │  │  │Agent   │ │Speculative  │ │ │
│  │  monkey-patch    │  │  │Sched.  │ │Decoding     │ │ │
│  └──────────────────┘  │  └────────┘ └─────────────┘ │ │
│                        └──────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  Inference Engine (vLLM / vLLM-Ascend)                  │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Paged Attention  │  Continuous Batching  │  ...   │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Agent Features (with diagrams):**

| Feature | Description | Config Switch |
|---------|-------------|---------------|
| **AgentCollab** | Self-evaluation-driven collaboration: small model handles tasks first, large model escalates if progress check fails | `MODEL_ROUTING_SWITCH` |
| **Early Stop** | Detect stalled iterations and inject completion guidance | `early_stop_enabled` |
| **Async Compression** | Asynchronously compress conversation history, control token growth | `resum_compress_async` |

**Acceleration Patches (text description):**

| Patch | Description | Config Switch |
|-------|-------------|---------------|
| **Agent Scheduler** | KV-cache aware scheduling, SJF/Peak Split/Prefill Delay strategies | `agent_scheduler` |
| **Speculative Decoding** | SAM/Suffix/Ngram matching, accelerate via draft tokens | `speculative_decoding` |
| **Disaggregate Prefill** | Prefill/Decode separation, cross-node collaborative inference | `disaggregate_prefill_enabled` |

---

## 🎯 Agent Features

### 1. AgentCollab

Self-evaluation-driven collaboration paradigm: small model processes tasks first, performs progress check via self-evaluation, and escalates to large model if needed.

<p align="center">
  <img src="./figs/agentcollab.svg" alt="AgentCollab workflow" width="600">
</p>

**Workflow:**
1. **Small Model Processing**: Small model handles the task
2. **Self-Driven Evaluation**: Check progress and quality
3. **Progress Check**: Determine if result meets requirements
4. **Model Escalation**: Escalate to large model if progress check fails

**Configuration:**

| Config Item | Description | Default |
|-------------|-------------|---------|
| `MODEL_ROUTING_SWITCH` | Enable model routing | `true` |
| `SMALL_MODEL_URL` | Small model OpenAI-compatible endpoint | - |
| `LARGE_MODEL_URL` | Large model OpenAI-compatible endpoint | - |
| `SMALL_MODEL_NAME` | Small model served name | - |
| `LARGE_MODEL_NAME` | Large model served name | - |

> Detailed docs: [agentic/docs/model_routing.md](agentic/docs/model_routing.md)

---

### 2. Early Stop

Detect stalled iterations and inject completion guidance into the next turn. This is guidance injection, not hard cancellation.

<p align="center">
  <img src="./figs/early_stop_compression.svg" alt="Early Stop detection" width="600">
</p>

**Detection Strategies:**
- **Repeated outputs**: Detect repeated thoughts
- **Minimum iteration**: Start checking after the configured turn
- **Auto-injection**: Add completion guidance to the next turn

**Configuration:**

| Config Item | Description | Default |
|-------------|-------------|---------|
| `early_stop_enabled` | Enable early stop | `true` |
| `early_stop_min_iter` | Minimum iteration before checking | `2` |

> Detailed docs: [agentic/docs/early_stop.md](agentic/docs/early_stop.md)

---

### 3. Async Compression

Asynchronously compress long conversation history, controlling token count growth to avoid exceeding context window limits.

**Compression Workflow:**
1. **Threshold Check**: Trigger when tokens exceed threshold
2. **Background Compression**: Submit compression asynchronously
3. **Semantic Preservation**: Retain key semantic information
4. **Resume**: Continue conversation with compressed context

**Configuration:**

| Config Item | Description | Default |
|-------------|-------------|---------|
| `resum_compress_async` | Enable async compression | `true` |
| `TOKEN_THRESHOLD` | Token threshold to trigger | `10` |
| `SEARCH_COMPRESS_URL` | Compression LLM endpoint | `OPENAI_BASE_URL/chat/completions` |
| `SEARCH_COMPRESS_MODEL_NAME` | Compression model name | `EXAMPLE_MODEL_NAME` |

> Detailed docs: [agentic/docs/multi_round_compression.md](agentic/docs/multi_round_compression.md)

---

## ⚡ Acceleration Patches

### 4. Agent Scheduler

KV-cache aware scheduling optimization for Agentic AI scenarios, includes three sub-schemes:

| Sub-scheme | Description |
|------------|-------------|
| **SJF (Short Job First)** | Prioritize short jobs to avoid long sequences blocking throughput |
| **Peak Split** | Close chunked prefill when requests are few, accelerate decode switching |
| **Prefill Delay** | Delay prefill when running requests are sufficient, batch process for efficiency |

**Scheduling Strategies:**

| Strategy | Description |
|----------|-------------|
| `fcfs` | First Come First Served (default) |
| `sjf` | Short Job First |
| `priority` | Priority scheduling |
| `agent` | KV-cache aware scheduling, sort by cache hit rate after first KV eviction |

> Detailed docs: [agentic/inference/docs/feature/agent_sched.md](agentic/inference/docs/feature/agent_sched.md)

---

### 5. Disaggregate Prefill

Prefill/Decode separation deployment, supports cross-node Prefill and Decode collaborative inference, suitable for large-scale service scenarios.

**Configuration:**

| Config Item | Description | Default |
|-------------|-------------|---------|
| `enabled` | Enable PD separation | `false` |
| `prefill_nodes` | List of Prefill nodes | - |
| `decode_nodes` | List of Decode nodes | - |

> Detailed docs: [agentic/inference/docs/feature/disaggregate_prefill.md](agentic/inference/docs/feature/disaggregate_prefill.md)

---

### 6. Speculative Decoding

Support multiple speculative decoding methods to accelerate inference using draft tokens:

| Method | Description | Dependency |
|--------|-------------|------------|
| `sam` | Suffix automaton matching, suitable for Agent scenarios with many repeated patterns | Built-in / `samfast` (optional) |
| `suffix` | Suffix tree matching | `arctic_inference` |
| `ngram` | N-gram statistical matching | Built-in |

**Adaptive Switch**: `batch_size_auto_switch` automatically disables speculative decoding when concurrent requests exceed threshold, avoiding throughput loss under high load.

> Detailed docs: [agentic/inference/docs/feature/speculative_decoding.md](agentic/inference/docs/feature/speculative_decoding.md)

---

## 🚀 Quick Start

### 1. Clone and enter the repository

```bash
git clone git@github.com:SparkEngineAI/AgentFlashInfer.git
cd AgentFlashInfer
```

### 2. Download the model

```bash
git lfs install
mkdir -p weights
git clone https://huggingface.co/openpangu/openPangu-Embedded-7B-DeepDiver \
  weights/openPangu-Embedded-7B-DeepDiver
```

### 3. Create the environments

```bash
conda create -n agentflashinfer python=3.10 -y
conda activate agentflashinfer
cd /path/to/AgentFlashInfer/agentic
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps


The vLLM serving runtime is verified with `vllm==0.18.1`.
```

### 4. Configure DDv2

```bash
cd /path/to/AgentFlashInfer/agentic
cp .env.example .env
```

Set the following values in `.env`:
```dotenv
MODEL_ROUTING_SWITCH=true
SMALL_MODEL_URL=http://127.0.0.1:8103/v1
SMALL_MODEL_NAME=pangu_small
LARGE_MODEL_URL=http://127.0.0.1:8104/v1
LARGE_MODEL_NAME=pangu_large
SMALL_MODEL_API_KEY=EMPTY
LARGE_MODEL_API_KEY=EMPTY
SEARCH_URL=<search-service-url>
SEARCH_API_KEY=<search-api-key>
URL_CRAWLER_API_KEYS=<crawler-api-key>
PROXY_CACHE=
```

When routing is disabled, use the small endpoint as the default model:

```dotenv
MODEL_ROUTING_SWITCH=false
OPENAI_BASE_URL=http://127.0.0.1:8103/v1
EXAMPLE_MODEL_NAME=pangu_small
OPENAI_API_KEY=EMPTY
```

### 5. Start two Pangu vLLM instances

Run each command in a separate terminal:

```bash
MODEL_PATH=/Path/to/Pangu_Small
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve "$MODEL_PATH" --served-model-name pangu_small --host 127.0.0.1 --port 8103 --tensor-parallel-size 1 --trust-remote-code --tokenizer-mode slow --dtype bfloat16 --gpu-memory-utilization 0.90 --disable-uvicorn-access-log
```

```bash
MODEL_PATH=/Path/to/Pangu_Large
CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve "$MODEL_PATH" --served-model-name pangu_large --host 127.0.0.1 --port 8104 --tensor-parallel-size 1 --trust-remote-code --tokenizer-mode slow --dtype bfloat16 --gpu-memory-utilization 0.90 --disable-uvicorn-access-log
```

Health checks:

```bash
curl --noproxy "*" -fsS http://127.0.0.1:8103/v1/models
curl --noproxy "*" -fsS http://127.0.0.1:8104/v1/models
```
### 6. Quick start with `run_ddv2.sh`

Keep vLLM running in a separate terminal, then start MCP and the DDv2 Agent server automatically and enter a query:

```bash
cd /path/to/AgentFlashInfer/agentic
conda run --no-capture-output -n agentflashinfer \
  env PYTHONNOUSERSITE=1 bash run_ddv2.sh --interactive
```

Alternatively, provide the query directly or through an environment variable:

```bash
bash run_ddv2.sh --query "Research the latest advances in agent inference."
DDV2_QUERY="Research the latest advances in agent inference." bash run_ddv2.sh
```

The vLLM service must already be running. The script waits for the completed `/history` result, prints the final answer, saves run logs under `agentic/ci_logs/`, and cleans up only the MCP and Agent server processes it started.

> Additional option — for Tongyi DeepResearch, configure `agentic/.env` for Qwen and start its vLLM service, then use `bash run_tongyi.sh --interactive` or `bash run_tongyi.sh --query "your query"`.

### 7. Start MCP and Agent manually (optional)

```bash
# Terminal 1
cd /path/to/AgentFlashInfer/agentic
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD" DISABLE_DISTRIBUTED_STORAGE=true \
  conda run -n agentflashinfer \
  python agent_infer/mcp_factory/mcp_server.py

# Terminal 2
cd /path/to/AgentFlashInfer/agentic
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD" DISABLE_DISTRIBUTED_STORAGE=true \
  AGENT_PORT=8289 conda run -n agentflashinfer \
  python examples/deepdiver_v2/server/app_ddv2.py
```

### 8. Start and use the Web UI

Keep the vLLM, MCP, and Agent services running.

```bash
cd /path/to/AgentFlashInfer/agentic/agent_infer/ui/ui
npm ci --no-audit --no-fund
PORT=8001 npm run dev
```

1. Enter a question and click `SUBMIT`.
2. `CONCURRENCY CONFIG` sets the number of parallel traces for the same question. Start with `×1`; use `×4` when parallel exploration is useful.
3. Click `▶` on the task card to expand it. After completion, read each thread's full final answer under “📋 任务最终执行总结”.

### 9. Submit a request with the API

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8289/api/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "user_input": "Research the target question with web search.",
    "agent_name": "assistant",
    "stream": false,
    "user_token": "demo",
    "concurrency": 1
  }'
```

Run logs are written to `agentic/ci_logs/`. If both routing URLs target the same endpoint, the script validates routing logic rather than switching between distinct model weights. Early Stop injects guidance; it is not hard cancellation.

---

## 📚 Documentation Index

### Agent Feature Documentation

| Document | Path |
|----------|------|
| AgentCollab | [agentic/docs/model_routing.md](agentic/docs/model_routing.md) |
| Early Stop | [agentic/docs/early_stop.md](agentic/docs/early_stop.md) |
| Async Compression | [agentic/docs/multi_round_compression.md](agentic/docs/multi_round_compression.md) |

### Acceleration Patch Documentation

| Document | Path |
|----------|------|
| Patch System | [agentic/inference/pangu_infer/patches/](agentic/inference/pangu_infer/patches/) |
| Speculative Decoding | [agentic/inference/docs/feature/speculative_decoding.md](agentic/inference/docs/feature/speculative_decoding.md) |
| Agent Scheduler | [agentic/inference/docs/feature/agent_sched.md](agentic/inference/docs/feature/agent_sched.md) |
| Disaggregate Prefill | [agentic/inference/docs/feature/disaggregate_prefill.md](agentic/inference/docs/feature/disaggregate_prefill.md) |

### Development Documentation

| Document | Path |
|----------|------|
| Installation Guide | [agentic/inference/docs/install.md](agentic/inference/docs/install.md) |
| Developer Guide | [agentic/inference/docs/developer_guide.md](agentic/inference/docs/developer_guide.md) |
| Code Style Guide | [agentic/inference/docs/code_style_guide.md](agentic/inference/docs/code_style_guide.md) |
| Development Rules | [agentic/inference/docs/development_rules.md](agentic/inference/docs/development_rules.md) |

---

## License

Copyright (c) 2026 Huawei Technologies Co., Ltd.

AgentFlashInfer is licensed under the [Apache License 2.0](LICENSE). Third-party code remains subject to its original licenses and copyright notices.

## 📖 Citation

If you use AgentFlashInfer in your research, please cite:

```bibtex
@article{lin2025towards,
  title={Towards Efficient Agents: A Co-Design of Inference Architecture and System},
  author={Lin, Weizhe and Zhen, Hui-Ling and Yang, Shuai and Wang, Xian and Liu, Renxi and Chen, Hanting and Zhang, Wangze and Zhou, Chuansai and Li, Yiming and Chen, Chen and others},
  journal={arXiv preprint arXiv:2512.18337},
  year={2025}
}

@article{gao2026agentcollab,
  title={AgentCollab: A Self-Evaluation-Driven Collaboration Paradigm for Efficient LLM Agents},
  author={Gao, Wenbo and Liu, Renxi and Wang, Xian and Guo, Fang and Yang, Shuai and Chen, Xi and Zhen, Hui-Ling and Chen, Hanting and Lin, Weizhe and Li, Xiaosong and others},
  journal={arXiv preprint arXiv:2603.26034},
  year={2026}
}
```
