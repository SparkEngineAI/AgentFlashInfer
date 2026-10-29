# 日志标记（与 run_ddv2.sh 中的 grep 模式匹配）
LOG_MARKER_REPEAT = "repeat output detected"
LOG_MARKER_MIN_ITER = "min_iter reached"

# 提示词（从 deepdiver_v2_agent_runner.py 提取）
REPEAT_OUTPUT_PROMPT = "请停止重复之前的内容。"

MIN_ITER_ANALYSIS_PROMPT = (
    "在进行任何下一步搜索之前，先分析之前已经完成的搜索工作，以及最近的搜索结果，"
    "是否已经达到了任务的目的。如果已经有了置信度足够高的结果，且尚未确认的项目并不那么相关，"
    "或者当前缺少足够的上下文信息，应该放弃继续执行剩余任务，进入任务的总结阶段，"
    "而非继续调用下一个搜索工具。"
)