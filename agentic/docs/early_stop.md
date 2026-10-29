# Early Stop 特性文档

## 概述

Early Stop 是一个独立的可复用组件，用于在 agent 推理过程中检测低效行为并主动干预。核心思想：**当检测到 agent 陷入重复输出或已达到最小迭代阈值时，注入提示消息引导 agent 自我纠正或进入总结阶段，避免无效迭代浪费 token 和时间**。

### 核心能力

- 重复输出检测（连续两轮 thought 相同即触发）
- 最小迭代次数检查（达到阈值后提醒 agent 分析进展）
- 冷却机制（避免连续触发早停干预）
- 多 agent 独立状态管理（每个 agent 维护独立的早停状态）
- 线程安全（`threading.Lock` 保护内部状态）
- 配置化，不依赖全局 config 模块

---

## 模块结构

```
features/early_stop/
├── __init__.py      # 导出 EarlyStopConfig, EarlyStopState, EarlyStopChecker, EarlyStopResult, EarlyStopManager
├── config.py        # EarlyStopConfig — 配置 dataclass
├── state.py         # EarlyStopState — 每个 agent 的早停状态
├── checker.py       # EarlyStopChecker — 无状态检查器 + EarlyStopResult
├── manager.py       # EarlyStopManager — 有状态管理器（门面类）
├── constants.py     # 日志标记 + 提示词常量
└── tests/           # 单元测试
    ├── conftest.py
    ├── test_config.py
    ├── test_state.py
    ├── test_checker.py
    └── test_manager.py
```

**依赖方向**：

```
examples/deepdiver_v2  ──→  features/early_stop  ←──  未来其他 agent
                                ↑
                          不依赖任何 example
                          不依赖 config.config 全局模块
                          不依赖 openai 运行时（仅类型引用）
```

---

## 核心类

### EarlyStopConfig — 配置

```python
@dataclass
class EarlyStopConfig:
    enabled: bool = True
    min_iter: int = 2
```

构造方式：

| 方法 | 用途 |
|------|------|
| `EarlyStopConfig()` | 使用默认值 |
| `EarlyStopConfig(enabled=False)` | 自定义参数 |
| `EarlyStopConfig.from_context(config_overrides=None)` | 从运行时上下文构建，支持全局 config 模块回退 |

**优先级**：`config_overrides` 中的值 > 全局 `config.config` 模块属性 > 默认值。

**env key 映射**（`from_context` 专用）：

| env key | field | 默认值 |
|---------|-------|--------|
| `early_stop_enabled` | `enabled` | `True` |
| `early_stop_min_iter` | `min_iter` | `2` |

---

### EarlyStopState — 每个 agent 的早停状态

```python
@dataclass
class EarlyStopState:
    last_thought: str = ""           # 上一轮 think 工具的 thought 内容
    last_early_stop_iter: int = -1   # 上一次触发早停的迭代轮次，-1 表示未触发过
```

- 不可变语义：checker 返回新 state 实例，不原地修改
- 支持相等性比较（`@dataclass` 自动生成 `__eq__`）

---

### EarlyStopResult — 检查结果

```python
@dataclass
class EarlyStopResult:
    should_inject: bool = False                                  # 是否应注入干预消息
    injected_message: Optional[EasyInputMessage] = None          # 注入的消息（role="user"）
    state: Optional[EarlyStopState] = None                       # 更新后的状态
    reason: str = "none"                                         # "repeat" | "min_iter" | "none"
```

---

### EarlyStopChecker — 无状态检查器

纯函数式设计，不修改外部状态，通过 `EarlyStopResult` 返回更新后的状态由调用方持久化。

```python
class EarlyStopChecker:
    @staticmethod
    def check(
        config: EarlyStopConfig,
        state: EarlyStopState,
        current_turn: int,
        tool_name: str,
        current_thought: str,
    ) -> EarlyStopResult: ...
```

**检查流程**：

```
EarlyStopChecker.check(config, state, current_turn, tool_name, current_thought)
  │
  ├─ 1. config.enabled == False 或 tool_name != "think"
  │     → 返回 reason="none"，不触发早停
  │
  ├─ 2. state.last_thought == current_thought（重复输出检测）
  │     → 返回 reason="repeat"
  │        should_inject=True, injected_message=REPEAT_OUTPUT_PROMPT
  │        state.last_early_stop_iter = current_turn
  │
  ├─ 3. current_turn >= config.min_iter
  │     且 current_turn - state.last_early_stop_iter > 1（冷却机制）
  │     → 返回 reason="min_iter"
  │        should_inject=True, injected_message=MIN_ITER_ANALYSIS_PROMPT
  │        state.last_early_stop_iter = current_turn
  │
  └─ 4. 其余情况
        → 返回 reason="none"
           state.last_thought = current_thought（仅更新 thought）
           state.last_early_stop_iter 保持不变
```

**关键设计**：

- **优先级**：repeat > min_iter。当同时满足两个条件时，优先判定为重复输出
- **冷却机制**：`current_turn - last_early_stop_iter > 1`，确保两次早停干预之间至少间隔 1 轮，避免连续注入消息干扰 agent 正常推理
- **无副作用**：每次检查返回新的 `EarlyStopState` 实例，不修改传入的 state

---

### EarlyStopManager — 有状态管理器（门面类）

封装 checker 的无状态逻辑，自动维护 `agent_id → EarlyStopState` 映射，对外暴露简洁接口。

```python
class EarlyStopManager:
    def __init__(self): ...

    def check(
        self,
        agent_id: str,
        current_turn: int,
        tool_name: str,
        current_thought: str,
    ) -> EarlyStopResult: ...

    def refresh_config(self, config_overrides: Optional[dict] = None) -> None: ...
    def init_agent(self, agent_id: str) -> None: ...
    def remove_agent(self, agent_id: str) -> None: ...
```

**`check` 内部流程**：

```
EarlyStopManager.check(agent_id, current_turn, tool_name, current_thought)
  │
  ├─ 加锁（threading.Lock）
  │
  ├─ 查找 agent_id 对应的 state（不存在则创建默认 state）
  │
  ├─ 委托 EarlyStopChecker.check() 执行检查
  │
  ├─ 持久化更新后的 state（result.state 不为 None 时）
  │
  └─ 释放锁，返回 result
```

**生命周期方法**：

| 方法 | 行为 |
|------|------|
| `init_agent(agent_id)` | 创建/重置指定 agent 的早停状态（用于子任务上下文创建时） |
| `remove_agent(agent_id)` | 移除指定 agent 的早停状态（用于 agent 运行结束时清理） |
| `refresh_config(config_overrides)` | 刷新早停配置，优先级同 `EarlyStopConfig.from_context` |

---

### 常量（constants.py）

| 常量 | 值 | 用途 |
|------|----|------|
| `LOG_MARKER_REPEAT` | `"repeat output detected"` | 日志标记，与 CI 脚本 grep 模式匹配 |
| `LOG_MARKER_MIN_ITER` | `"min_iter reached"` | 日志标记，与 CI 脚本 grep 模式匹配 |
| `REPEAT_OUTPUT_PROMPT` | `"请停止重复之前的内容。"` | 重复输出时注入的提示消息 |
| `MIN_ITER_ANALYSIS_PROMPT` | 见下方 | 达到最小迭代次数时注入的提示消息 |

**MIN_ITER_ANALYSIS_PROMPT**：

> 在进行任何下一步搜索之前，先分析之前已经完成的搜索工作，以及最近的搜索结果，是否已经达到了任务的目的。如果已经有了置信度足够高的结果，且尚未确认的项目并不那么相关，或者当前缺少足够的上下文信息，应该放弃继续执行剩余任务，进入任务的总结阶段，而非继续调用下一个搜索工具。

---

## 集成方式

### 在 Ddv2AgentRunner 中的集成

**1. 初始化**：

```python
from early_stop import EarlyStopManager

early_stop_manager = EarlyStopManager()
```

**2. 每轮检查**（在 `_run_single_turn` 中，agent 调用 think 工具后）：

```python
result = early_stop_manager.check(
    agent_id=agent.name,
    current_turn=turn_count,
    tool_name="think",
    current_thought=thought_content,
)
if result.should_inject:
    messages.append(result.injected_message)
```

**3. 生命周期管理**：

```python
# 子任务上下文创建时
early_stop_manager.init_agent(agent_id)

# agent 运行结束时
early_stop_manager.remove_agent(agent_id)
```

### 新 Agent 接入方式

任何 agent runner 只需 3 步接入：

```python
# 1. 初始化（__init__ 或 run() 中）
early_stop_manager = EarlyStopManager()

# 2. 每轮检查（think 工具调用后）
result = early_stop_manager.check(agent_id, current_turn, "think", current_thought)
if result.should_inject:
    messages.append(result.injected_message)

# 3. 清理（agent 结束时）
early_stop_manager.remove_agent(agent_id)
```

**接入成本**：1 处初始化 + 1 处检查点 + 1 处清理。

---

## 完整状态流转

```
                    ┌────────────────────────────────────┐
                    │  EarlyStopManager.check(agent_id)  │
                    │  每轮 think 工具调用后执行           │
                    └──────────────┬─────────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  enabled == False 或             │
                    │  tool_name != "think"            │
                    │  → reason="none"，不触发          │
                    └──────────────┬──────────────────┘
                                   │ tool_name == "think" 且 enabled
                    ┌──────────────▼──────────────────┐
                    │  last_thought == current_thought │
                    │  → reason="repeat"               │
                    │  → 注入 REPEAT_OUTPUT_PROMPT     │
                    │  → last_early_stop_iter=turn     │
                    └──────────────┬──────────────────┘
                                   │ thought 不重复
                    ┌──────────────▼──────────────────┐
                    │  turn >= min_iter                │
                    │  且 turn - last_stop_iter > 1    │
                    │  → reason="min_iter"             │
                    │  → 注入 MIN_ITER_ANALYSIS_PROMPT │
                    │  → last_early_stop_iter=turn     │
                    └──────────────┬──────────────────┘
                                   │ 未达阈值或在冷却期
                    ┌──────────────▼──────────────────┐
                    │  reason="none"                   │
                    │  → 仅更新 last_thought           │
                    │  → last_early_stop_iter 不变     │
                    └─────────────────────────────────┘
```

---

## 测试

测试位于 `app/agentic/features/early_stop/tests/`：

| 文件 | 覆盖范围 |
|------|----------|
| `conftest.py` | sys.path 配置 + openai mock |
| `test_config.py` | `EarlyStopConfig` 默认值、自定义参数、`from_context` 优先级与回退 |
| `test_state.py` | `EarlyStopState` 默认值、自定义参数、相等性比较 |
| `test_checker.py` | `EarlyStopChecker` 守卫条件、重复检测、min_iter 检查、冷却机制、优先级、`EarlyStopResult` 默认值 |
| `test_manager.py` | `EarlyStopManager` 初始化、check 委托与状态持久化、多 agent 独立性、生命周期方法、配置刷新、多线程并发安全 |

运行：

```bash
pytest app/agentic/features/early_stop/tests/ -v
```

代码覆盖率：源码模块 **100%**，整体 **98%**。

---

## 对外接口汇总

| 接口 | 签名 | 说明 |
|------|------|------|
| `EarlyStopManager.check()` | `(agent_id, current_turn, tool_name, current_thought) -> EarlyStopResult` | 核心接口：执行早停检查 |
| `EarlyStopManager.init_agent()` | `(agent_id: str) -> None` | 初始化/重置 agent 早停状态 |
| `EarlyStopManager.remove_agent()` | `(agent_id: str) -> None` | 移除 agent 早停状态 |
| `EarlyStopManager.refresh_config()` | `(config_overrides: Optional[dict]) -> None` | 刷新早停配置 |
| `EarlyStopConfig.from_context()` | `(config_overrides: Optional[dict]) -> EarlyStopConfig` | 从运行时上下文构建配置 |
| `EarlyStopChecker.check()` | `(config, state, current_turn, tool_name, current_thought) -> EarlyStopResult` | 无状态检查（低级接口） |
| `EarlyStopResult.should_inject` | `bool` | 是否应注入干预消息 |
| `EarlyStopResult.injected_message` | `Optional[EasyInputMessage]` | 注入的消息 |
| `EarlyStopResult.reason` | `str` | 触发原因："repeat" / "min_iter" / "none" |