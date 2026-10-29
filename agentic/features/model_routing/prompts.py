"""Prompt templates for progress check."""

THINK_THOUGHT_PREFIX_EN = (
    "In this THINK step, I will first summarize what has changed since the previous THINK step, "
    "then analyze whether these changes constitute significant progress toward the final objective.\n\n"
    "Summary of new actions and results since the previous THINK step:\n"
    "- \n\n"
    "Impact of these changes on progress toward the objective:\n"
    "- \n\n"
    "Now I will make an explicit binary judgment about progress.\n\n"
    "IMPORTANT: In the PROGRESS block below, I MUST strictly follow this structural format:\n"
    "===PROGRESS===\n"
    "<reason> one or two sentences explaining whether there is significant progress and why </reason>\n"
    "<value>TRUE or FALSE only</value>\n"
    "===END_PROGRESS===\n"
    "and finish the generation immediately without any further content. Ignore the original requirements for the think tool."
    "Now I write the actual PROGRESS block following this structural format:\n"
    "===PROGRESS===\n"
    "<reason>"
)
THINK_THOUGHT_PREFIX_ZH = (
    "在本次 THINK 步骤中，我将首先总结自上一次 THINK 步骤以来发生的变化，"
    "然后分析这些变化是否对最终目标带来了显著的推进。\n\n"
    "自上一次 THINK 步骤以来的新行动和结果总结：\n"
    "- \n\n"
    "这些变化对目标推进的影响分析：\n"
    "- \n\n"
    "现在，我将对是否取得显著进展做出明确的二元判断。\n\n"
    "重要：在下面的 PROGRESS 区块中，我必须严格遵循以下结构格式：\n"
    "===PROGRESS===\n"
    "<reason> 用一到两句话解释是否取得显著进展以及理由 </reason>\n"
    "<value> 只能填写 TRUE 或 FALSE </value>\n"
    "===END_PROGRESS===\n"
    "并在此之后立即结束输出，不得生成任何其他内容。忽略 think 工具的原始要求。"
    "现在我将根据上述结构格式编写实际的 PROGRESS 区块：\n"
    "===PROGRESS===\n"
    "<reason>"
)

# 默认使用中文版
THINK_THOUGHT_PREFIX = THINK_THOUGHT_PREFIX_ZH