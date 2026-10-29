# Model Routing 特性文档

## 概述

Model Routing 是一个独立的可复用组件，支持在推理过程中动态切换大小模型。核心思想：**默认使用小模型降低成本和延迟，当检测到"进展不足"时升级到大模型提升推理质量，完成若干轮后自动降回小模型**。

### 核心能力

- 大小模型动态切换（升级 + 降级闭环）
- 可插拔的切换策略（`SwitchStrategy` ABC）
- 可插拔的模型工厂（`ModelFactory` ABC），支持客户端缓存
- 异步优先设计（`httpx.AsyncClient`），提供同步桥接方法
- 配置化，不依赖全局 config 模块

---

## 模块结构

```
features/model_routing/
├── __init__.py      # 导出 ModelRouter, ModelRoutingConfig, SwitchStrategy, ProgressCheckStrategy, ModelFactory, OpenAIModelFactory
├── config.py        # ModelRoutingConfig — 配置 dataclass
├── router.py        # ModelRouter — 核心门面类
├── strategy.py      # SwitchStrategy (ABC) + ProgressCheckStrategy
├── factory.py       # ModelFactory (ABC) + OpenAIModelFactory
└── prompts.py       # THINK_THOUGHT_PREFIX_EN / THINK_THOUGHT_PREFIX_ZH
```

**依赖方向**：

```
examples/deepdiver_v2  ──→  features/model_routing  ←──  未来其他 agent
                                ↑
                          不依赖任何 example
                          不依赖 config.config 全局模块
```

---

## 核心类

### ModelRoutingConfig — 配置

```python
@dataclass
class ModelRoutingConfig:
    enabled: bool = False
    small_model_url: str = ""
    large_model_url: str = ""
    small_model_name: str = ""
    large_model_name: str = ""
    small_model_api_key: str = ""
    large_model_api_key: str = ""
    small_model_template: str = ""
    large_model_template: str = ""
```

构造方式：

| 方法 | 用途 |
|------|------|
| `ModelRoutingConfig()` | 使用默认值 |
| `ModelRoutingConfig.from_dict(d)` | 从 `context["config"]` 构造，忽略未知 key |
| `ModelRoutingConfig.from_config_module(overrides, config_module)` | 从 context 覆盖 + 全局 config 模块构造，支持 env key 映射 |

**env key 映射**（`from_config_module` 专用）：

| env key | field | 默认值来源 |
|---------|-------|-----------|
| `MODEL_ROUTING_SWITCH` | `enabled` | `config.MODEL_ROUTING_SWITCH` |
| `SMALL_MODEL_URL` | `small_model_url` | `config.SMALL_MODEL_URL` |
| `LARGE_MODEL_URL` | `large_model_url` | `config.LARGE_MODEL_URL` |
| `SMALL_MODEL_NAME` | `small_model_name` | `config.SMALL_MODEL_NAME` |
| `LARGE_MODEL_NAME` | `large_model_name` | `config.LARGE_MODEL_NAME` |
| `SMALL_MODEL_API_KEY` | `small_model_api_key` | `config.SMALL_MODEL_API_KEY` |
| `LARGE_MODEL_API_KEY` | `large_model_api_key` | `config.LARGE_MODEL_API_KEY` |
| `SMALL_MODEL_TEMPLATE` | `small_model_template` | `config.SMALL_MODEL_TEMPLATE` |

优先级：`overrides` 中的值 > `config_module` 中的默认值。

---

### ModelRouter — 核心门面类

统一入口，协调 Strategy 和 Factory 完成路由决策。

```python
class ModelRouter:
    def __init__(
        self,
        config: ModelRoutingConfig,
        strategy: SwitchStrategy | None = None,   # 默认 ProgressCheckStrategy
        factory: ModelFactory | None = None,       # 默认 OpenAIModelFactory
    ): ...

    async def resolve_model(
        self,
        agent_name: str,
        conversation_history: list[dict],
        model: Model,
        model_settings: ModelSettings,
    ) -> Model: ...

    def resolve_model_sync(...) -> Model: ...  # 同步桥接

    @property
    def current_model(self) -> str: ...    # "small_model" | "large_model"
    @property
    def config(self) -> ModelRoutingConfig: ...
```

**`resolve_model` 决策流程**：

```
resolve_model(agent_name, conversation_history, model, model_settings)
  │
  ├─ 1. config.enabled == False → 返回默认 model
  │
  ├─ 2. strategy.resolve_fixed_model(agent_name) != None
  │     → 返回固定模型（如 PlannerAgent 始终用大模型）
  │
  ├─ 3. current_model == "large_model"
  │     ├─ strategy.should_deescalate() == True
  │     │   → 切回 small_model，调用 on_deescalated()
  │     └─ should_deescalate() == False
  │         → 保持 large_model，调用 tick_large_round()
  │
  └─ 4. current_model == "small_model"
        ├─ strategy.should_escalate() == True
        │   → 切到 large_model，调用 on_escalated()
        └─ should_escalate() == False
            → 保持 small_model
```

**关键设计**：
- 升降级决策统一在 `resolve_model` 中完成，消除了旧架构中升级在请求前、降级在响应后的分散问题
- 升级当轮不执行降级检查（因为 `current_model` 仍为 `"small_model"`），避免 off-by-one
- `should_deescalate()` 是纯查询方法，不修改策略内部状态；状态变更通过 `tick_large_round()` / `on_escalated()` / `on_deescalated()` 显式执行

**`resolve_model_sync`**：供同步调用方使用，内部通过 `asyncio.get_event_loop()` + `run_until_complete()` 桥接；若当前线程无事件循环则创建临时循环；若事件循环已在运行则通过 `ThreadPoolExecutor` 提交。

---

### SwitchStrategy — 切换策略

```python
class SwitchStrategy(ABC):
    async def should_escalate(self, agent_name, conversation_history, model, model_settings) -> bool: ...
    def should_deescalate(self) -> bool: ...                    # 纯查询，无副作用
    def resolve_fixed_model(self, agent_name: str) -> str | None: ...
    def on_escalated(self) -> None: ...                         # 升级后回调，默认空实现
    def on_deescalated(self) -> None: ...                       # 降级后回调，默认空实现
    def tick_large_round(self) -> None: ...                     # 递减大模型剩余轮次，默认空实现
```

### ProgressCheckStrategy — 基于 progress_check 的策略

基于小模型自评的升级策略：让小模型判断自身进展是否充足，若不足则升级到大模型。

```python
class ProgressCheckStrategy(SwitchStrategy):
    def __init__(
        self,
        config: ModelRoutingConfig,
        trigger_tool_name: str = "think",         # 触发 progress_check 的工具名
        prompt_template: str | None = None,       # progress_check 提示词，默认中文版
        progress_check_min_iter: int = 3,         # 开始检查的最小迭代数
        large_rounds: int = 2,                    # 升级后大模型执行轮数
        fixed_agents: dict[str, str] | None = None,  # 固定模型 agent 映射
    ): ...
```

**默认配置**：`fixed_agents = {"PlannerAgent": "large_model"}`

#### 升级逻辑（`should_escalate`）

```
should_escalate(agent_name, conversation_history, model, model_settings)
  │
  ├─ iteration < progress_check_min_iter → False
  │
  ├─ 上一轮不包含 trigger_tool_name → False
  │
  └─ 执行 progress_check（异步 HTTP 请求）:
      ├─ 构造检查消息（删除最后一个 trigger_tool 消息及其后所有消息，追加 progress_check 消息）
      ├─ 使用 httpx.AsyncClient 异步调用小模型
      ├─ 解析 <value> 标签:
      │   ├─ "FALSE" → True（小模型自评进展不足，需升级）
      │   ├─ "TRUE" → False（进展充足，保持小模型）
      │   └─ 异常 → True（兜底升级）
      └─ 请求异常 → True（兜底升级）
```

**关键设计**：
- 使用 `httpx.AsyncClient` 异步请求，不阻塞事件循环
- 异常兜底：宁可多用大模型也不让任务卡住

#### 降级逻辑（`should_deescalate`）

基于轮次倒计时的纯查询方法：

```python
def should_deescalate(self) -> bool:
    return self._large_rounds_left <= 0
```

状态变更由 `ModelRouter` 通过以下方法控制：

| 方法 | 行为 |
|------|------|
| `on_escalated()` | 重置 `_large_rounds_left = large_rounds` |
| `on_deescalated()` | 重置 `_large_rounds_left = 0` |
| `tick_large_round()` | `_large_rounds_left -= 1`（仅在 > 0 时） |

---

### ModelFactory — 模型创建

```python
class ModelFactory(ABC):
    def __init__(self, config: ModelRoutingConfig) -> None: ...
    def create(self, model_type: str) -> Model: ...           # "small_model" | "large_model"
    @abstractmethod
    def get_or_create_small_model(self) -> Model: ...
    @abstractmethod
    def get_or_create_large_model(self) -> Model: ...
```

### OpenAIModelFactory — 基于 AsyncOpenAI 的工厂

缓存 `AsyncOpenAI` 客户端和 `OpenAIChatCompletionsModel` 实例，避免每轮重建连接。

```
get_or_create_small_model()
  ├─ self._small_model 已存在 → 直接返回
  └─ 不存在 → 创建 AsyncOpenAI 客户端（缓存） + 创建 Model 实例（缓存）→ 返回
```

---

### 提示词（prompts.py）

| 常量 | 说明 |
|------|------|
| `THINK_THOUGHT_PREFIX_EN` | 英文版 progress_check 提示词 |
| `THINK_THOUGHT_PREFIX_ZH` | 中文版 progress_check 提示词 |
| `THINK_THOUGHT_PREFIX` | 默认使用中文版 |

提示词要求模型输出结构化的 progress 判断：

```
===PROGRESS===
<reason> 是否有显著进展 </reason>
<value> TRUE 或 FALSE </value>
===END_PROGRESS===
```

---

## 集成方式

### 在 Ddv2AgentRunner 中的集成（已实现）

**1. 初始化**（`run()` 方法中，line 185-192）：

```python
routing_config = ModelRoutingConfig.from_config_module(
    context.get("config", {}), config
)
Ddv2AgentRunner._router = ModelRouter(
    config=routing_config,
    strategy=ProgressCheckStrategy(config=routing_config, trigger_tool_name="think"),
)
```

**2. 路由决策**（`_get_new_response()` 中，line 816-823）：

```python
model = cls._get_model(agent, run_config)  # 基类签名，3 参数 sync
router = cls._router
if router and router.config.enabled:
    converted_messages = Ddv2ChatConverter.items_to_messages(filtered.input)
    if system_prompt:
        converted_messages.insert(0, {"role": "system", "content": system_prompt})
    model = await router.resolve_model(agent.name, converted_messages, model, model_settings)
```

### 新 Agent 接入方式

任何 agent runner 只需在 `_run_single_turn` 的 1 个位置插入路由决策：

```python
# 1. 初始化（__init__ 或 run() 中）
router = ModelRouter(
    config=ModelRoutingConfig.from_dict(context.get("config", {})),
    strategy=ProgressCheckStrategy(trigger_tool_name="think"),
)

# 2. 路由决策（_get_model 之后、get_response 之前）
model = cls._get_model(agent, run_config)
if router and router.config.enabled:
    messages = convert_to_messages(filtered.input)
    model = await router.resolve_model(agent.name, messages, model, model_settings)
```

**接入成本**：1 处初始化 + 1 处插入点。

### 同步调用方接入

对于同步架构的 agent（如 Deep Research），使用 `resolve_model_sync`：

```python
model = router.resolve_model_sync(agent_name, messages, model, model_settings)
```

---

## 完整状态流转

```
                    ┌─────────────────────────────┐
                    │   ModelRouter.resolve_model  │
                    │   每轮 LLM 调用前执行        │
                    └──────────────┬──────────────┘
                                   │
                  ┌────────────────┼──────────────────┐
                  │                │                  │
         ┌────────▼──────┐  ┌─────▼──────┐  ┌───────▼────────┐
         │ fixed_agents  │  │ 进展充足    │  │ 进展不足        │
         │ → 固定模型    │  │ → stay small│  │ → large_model  │
         │ (如 Planner)  │  │             │  │ on_escalated() │
         └───────────────┘  └─────────────┘  │ rounds_left=N  │
                                             └───────┬────────┘
                                                     │
                                             ┌───────▼──────────┐
                                             │ 下一轮 resolve    │
                                             │ should_deescalate │
                                             │ tick_large_round  │
                                             └────────┬─────────┘
                                                      │
                                             ┌────────▼────────┐
                                             │ rounds_left==0? │
                                             │ → small_model   │
                                             │ on_deescalated()│
                                             └─────────────────┘
```

---

## 测试

测试位于 `app/agentic/tests/features/model_routing/`：

| 文件 | 覆盖范围 |
|------|----------|
| `conftest.py` | 通用 fixture（config、mock model、conversation history） |
| `test_config.py` | `ModelRoutingConfig` 默认值、`from_dict`、`from_config_module` |
| `test_factory.py` | `ModelFactory.create()` 分发、`OpenAIModelFactory` 缓存 |
| `test_strategy.py` | `ProgressCheckStrategy` 升级/降级/fixed_agents/tick |
| `test_router.py` | `ModelRouter` 完整决策流程、升降级闭环、`resolve_model_sync` |

运行：

```bash
pytest app/agentic/tests/features/model_routing/ -v
```

---

## 对外接口汇总

| 接口 | 签名 | 说明 |
|------|------|------|
| `ModelRouter.resolve_model()` | `async (agent_name, conversation_history, model, model_settings) -> Model` | 核心接口：返回本轮应使用的模型 |
| `ModelRouter.resolve_model_sync()` | `(agent_name, conversation_history, model, model_settings) -> Model` | 同步便捷方法 |
| `ModelRouter.current_model` | `str` (property) | 当前模型标识 |
| `ModelRouter.config` | `ModelRoutingConfig` (property) | 只读配置 |
| `ModelRoutingConfig.from_dict()` | `(d: dict) -> ModelRoutingConfig` | 从 dict 构造 |
| `ModelRoutingConfig.from_config_module()` | `(overrides, config_module) -> ModelRoutingConfig` | 从 context + config 模块构造 |
| `SwitchStrategy.should_escalate()` | `async (...) -> bool` | 是否升级到大模型 |
| `SwitchStrategy.should_deescalate()` | `() -> bool` | 是否降级回小模型 |
| `SwitchStrategy.resolve_fixed_model()` | `(agent_name) -> str \| None` | 返回固定模型标识或 None |
| `ModelFactory.create()` | `(model_type: str) -> Model` | 根据类型创建模型 |
| `ModelFactory.get_or_create_small_model()` | `() -> Model` | 获取或创建小模型 |
| `ModelFactory.get_or_create_large_model()` | `() -> Model` | 获取或创建大模型 |
