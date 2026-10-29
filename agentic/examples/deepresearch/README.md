# DeepResearch Agent Service

基于 Alibaba 的 [DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) 的 `MultiTurnReactAgent`，集成到 AgentFlashInfer 中，提供 FastAPI + SSE 流式推送的深度研究 Agent 服务。

## 来源与修改

上游采用 Apache-2.0 许可证，全文见仓库根目录 [LICENSE](../../../LICENSE)。本项目增加了服务接口、SSE 事件推送，并接入模型路由、提前停止和上下文压缩功能。

## 架构概览

```
用户请求 → FastAPI (/api/submit)
              │
              ├─ 创建 Session + 注册 SSE 队列
              ├─ 后台启动 MultiTurnReactAgent
              │       │
              │       ├─ ReAct 循环调用 VLLM
              │       ├─ 解析工具调用 (search/visit/scholar/python/parse_file)
              │       └─ 每步发射 SSE 事件 → _sse_active_queues
              │
              └─ GET /tracestream/{session_id} ← SSE 实时推送
```

**核心设计决策**：
- `MultiTurnReactAgent` 保留 `qwen_agent.FnCallAgent` 继承和内部 ReAct 循环，不改造为 `openai-agents-sdk.Agent`
- SSE 采用 ReAct 步骤级别推送（thinking/tool_call/tool_result/answer），不做 token 级别流式
- SSE 事件由 agent 直接推送到 session 队列，不经过 TracingProcessor 中转

## 目录结构

```
deepresearch/
├── main.py                      # uvicorn 入口
├── run.sh                       # 一键启动脚本 (VLLM + DeepResearch 服务)
├── react_agent.py               # MultiTurnReactAgent (FnCallAgent + SSE 事件发射)
├── config/
│   └── deepresearch_config.py   # 统一配置管理 (环境变量)
├── tools/
│   ├── prompt.py                # SYSTEM_PROMPT, EXTRACTOR_PROMPT
│   ├── tool_search.py           # Google 搜索 (Serper API)
│   ├── tool_visit.py            # 网页访问 (Jina API + 摘要 LLM)
│   ├── tool_scholar.py          # Google Scholar (Serper API)
│   ├── tool_python.py           # Python 代码执行 (SandboxFusion)
│   ├── tool_file.py             # 文件解析入口
│   └── file_tools/
│       ├── file_parser.py       # PDF/DOCX/PPTX/TXT/CSV/XLSX 等解析
│       ├── idp.py               # 阿里云 IDP 文档解析服务
│       ├── utils.py             # qwen_agent 工具函数
│       ├── video_agent.py       # 视频/音频分析 Agent
│       └── video_analysis.py    # 关键帧提取 + 语音转录 + 内容分析
├── server/
│   ├── app_dr.py                # FastAPI 应用 (API 端点)
│   └── session.py               # Session 生命周期管理
└── tracing/
    └── processor.py             # SSE 广播 (直接推送机制)
```

## 快速启动

### 1. 配置环境变量

在 `examples/.env` 中填写必要配置：

```bash
# 必填
VLLM_MODEL_PATH=/path/to/model
SERPER_KEY_ID=your_serper_key
JINA_API_KEYS=your_jina_key
API_KEY=your_summary_api_key
API_BASE=https://api.openai.com/v1
SUMMARY_MODEL_NAME=gpt-4o-mini
SANDBOX_FUSION_ENDPOINT=http://host1:8080

# 可选
DEVICE_ID=0
VLLM_PORT=6001
DR_SERVER_PORT=8290
MAX_LLM_CALL_PER_RUN=100
GENERATION_TEMPERATURE=0.6
```

### 2. 安装依赖

```bash
pip install -r examples/deepresearch/requirements.txt
```

### 3. 启动服务

```bash
# 方式一：一键启动（自动启动 VLLM + DeepResearch 服务）
VLLM_MODEL_PATH=/path/to/model bash examples/deepresearch/run.sh

# 方式二：如果 VLLM 已在运行，run.sh 会自动检测并跳过 VLLM 启动
VLLM_PORT=6001 VLLM_MODEL_PATH=/path/to/model bash examples/deepresearch/run.sh
```

服务启动后可访问：
- API 文档：`http://localhost:8290/docs`
- 健康检查：`http://localhost:8290/api/stats`

## API 接口

### 提交研究问题

```bash
curl -X POST http://localhost:8290/api/submit   -H "Content-Type: application/json"   -d '{
    "user_input": "What are the latest advances in quantum error correction?",
    "user_token": "user1"
  }'

curl -X POST http://localhost:8290/api/submit \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "一位画家，父亲心脏病去世，有一个姐姐，与妻子育有五个子女，后婚姻破裂，后经历三段感情史。后有一部文学作品基于此人撰写，这部作品叫什么？",
    "user_token": "user1"
  }'
```

响应：
```json
{
  "session_id": "sess_abc123",
  "task_id": "task_def456",
  "status": "started"
}
```

### SSE 流式订阅

```bash
curl -N http://localhost:8290/tracestream/sess_abc123
```

SSE 事件流示例：

```
data: {"event": "on_react_step", "data": {"type": "thinking", "content": "I need to search for...", "round": 1}, "timestamp": 1715040000000}

data: {"event": "on_react_step", "data": {"type": "tool_call", "tool_name": "search", "tool_args": {"query": ["quantum error correction 2024"]}, "round": 1}, "timestamp": 1715040001000}

data: {"event": "on_react_step", "data": {"type": "tool_result", "tool_name": "search", "content": "A Google search for 'quantum error correction 2024' found 10 results...", "round": 1}, "timestamp": 1715040003000}

data: {"event": "on_react_step", "data": {"type": "answer", "content": "Recent advances in quantum error correction include...", "round": 5}, "timestamp": 1715040060000}

data: {"event": "on_trace_end", "data": {"termination": "answer", "prediction": "Recent advances in quantum error correction include..."}, "timestamp": 1715040061000}
```

### 其他接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/stop/{session_id}` | 停止运行中的 session |
| GET | `/api/stats` | 系统运行统计 |
| GET | `/history` | 历史对话记录 |

## SSE 事件类型

| 事件 | data.type | 说明 |
|------|-----------|------|
| `on_react_step` | `thinking` | LLM 每轮返回的思考内容 |
| `on_react_step` | `tool_call` | 检测到工具调用，包含 tool_name 和 tool_args |
| `on_react_step` | `tool_result` | 工具执行结果 |
| `on_react_step` | `answer` | 检测到 `<answer>` 标签，包含最终答案 |
| `on_react_step` | `termination` | 因超时等原因终止 |
| `on_trace_end` | - | 整个研究流程结束 |

## 可用工具

| 工具 | 说明 | 必需环境变量 |
|------|------|-------------|
| `search` | Google 搜索 (Serper API) | `SERPER_KEY_ID` |
| `visit` | 网页访问 + 内容摘要 (Jina API) | `JINA_API_KEYS`, `API_KEY`, `API_BASE`, `SUMMARY_MODEL_NAME` |
| `google_scholar` | Google Scholar 学术检索 | `SERPER_KEY_ID` |
| `PythonInterpreter` | Python 代码沙箱执行 | `SANDBOX_FUSION_ENDPOINT` |
| `parse_file` | 文件解析 (PDF/DOCX/PPTX/TXT/CSV/XLSX/ZIP/MP4/MP3) | `USE_IDP`, `IDP_KEY_*`, `DASHSCOPE_*` (可选) |
