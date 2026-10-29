# 多轮异步压缩特性文档

## 概述

多轮异步压缩在后台生成对话摘要，并在后续轮次开始时应用摘要，减少历史内容在上下文中的占用。

模块位于 `agentic/features/compression/`，提供 DDV2 和 DeepResearch 两种对话适配器。

---

## 模块结构

```text
features/compression/
├── config.py                # CompressionConfig：配置
├── manager.py               # CompressionManager：提交和应用压缩结果
├── queue_manager.py         # 后台异步任务队列
├── conversation_adapter.py  # ConversationAdapter：对话适配协议
└── adapters/
    ├── ddv2.py              # DDV2 RunItem 适配器
    └── deepresearch.py      # DeepResearch messages 适配器
```

---

## 核心接口

| 接口 | 用途 |
|------|------|
| `CompressionConfig` | 设置触发条件、摘要服务和历史保留参数 |
| `CompressionManager(config, adapter)` | 管理当前 Agent 的压缩任务 |
| `on_turn_start(current_turn, generated_items)` | 检查已完成任务，返回处理后的历史列表 |
| `on_turn_end(current_turn, generated_items, last_tool_name, token_count)` | 在工具执行后检查条件并提交后台任务 |
| `ConversationAdapter` | 转换消息格式、定位摘要写入点和重构历史 |

每个独立 Agent 使用自己的 `CompressionManager`。调用方应接收 `on_turn_start()` 的返回值，再构造本轮模型输入。

---

## 工作流程

1. **轮次开始**：检查后台任务；尚未完成时继续使用当前历史。
2. **模型推理和工具执行**：按 Agent 的常规流程更新历史。
3. **轮次结束**：功能已启用、工具名匹配、token 数超过阈值且没有待处理压缩任务时，提交后台摘要任务。
4. **后续轮次应用**：结果未过期且摘要格式有效时，定位提交边界之前的思考项，写入摘要并通过适配器重构历史。

摘要写入位置取决于适配器：

- DDV2：`think` 工具调用的 `thought` 参数。
- DeepResearch：assistant 消息的 `<think>...</think>` 内容。

历史重构按适配器规则保留头部、部分工具调用和提交边界之后的新消息。已被压缩的旧工具结果不再逐条进入后续模型输入。

任务失败、结果过期或摘要校验失败时，会记录日志并跳过该结果。摘要属于有损压缩，不保证保留原始历史的全部细节。

---

## 主要配置

以下为 `CompressionConfig` 的主要字段；运行入口可覆盖默认配置，具体取值以实际接入代码为准。

| 字段 | 用途 |
|------|------|
| `enabled` | 功能开关 |
| `token_threshold` | 触发阈值；token 数需严格大于此值 |
| `trigger_tool_names` | 允许触发压缩的工具名 |
| `staleness_threshold` | 允许的提交至消费轮次差 |
| `header_item_count` | 历史重构时保留的头部项数 |
| `think_tool_name` | 摘要写入位置对应的工具名或思考标签名 |
| `compress_llm_url` | 摘要服务的完整 Chat Completions 请求地址 |
| `compress_llm_api_key` | 摘要服务的鉴权配置 |
| `compress_llm_model_name` | 摘要模型名称 |
| `compress_llm_temperature` | 摘要生成温度 |
| `compress_llm_max_tokens` | 摘要生成的最大输出 token 数 |

`token_count` 由调用方提供；为零时，管理器使用历史项数量进行粗略估算。服务地址、鉴权信息和模型名称应通过部署配置提供。
