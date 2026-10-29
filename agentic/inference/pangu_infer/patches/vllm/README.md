# vLLM patches

本目录基于 [vLLM](https://github.com/vllm-project/vllm) 的配置和调度器实现进行修改。上游采用 Apache-2.0 许可证，全文见仓库根目录 [LICENSE](../../../../../LICENSE)。

本项目通过 patch 注册机制接入修改后的实现：`config.py` 增加模型配置与投机解码适配，`v1/core/sched/scheduler.py` 接入 Agent 调度策略及相关统计。
