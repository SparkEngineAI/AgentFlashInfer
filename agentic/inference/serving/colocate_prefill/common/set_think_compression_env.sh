echo "Read and set parameters for thinking compression..."

# 重复输出相关
export REASONER_EARLY_THINK_STOPPING_STRING_FOR_REPEATED_OUTPUT=$'\n\nI think I already thought for a long time, and I should be able to deduce the final answer from the content above.[unused17]'
if [ -z "${REASONER_EARLY_THINK_STOPPING_REPEAT_DETECTION}" ]; then
  export REASONER_EARLY_THINK_STOPPING_REPEAT_DETECTION="1"
fi

# 思维链压缩相关
if [ -z "${REASONER_EARLY_THINK_STOPPING_ENABLE_COMPRESSION}" ]; then
  export REASONER_EARLY_THINK_STOPPING_ENABLE_COMPRESSION="1"
fi

if [ -z "${REASONER_EARLY_THINK_STOPPING_COMPRESSION_OVERTHINKING}" ]; then
  export REASONER_EARLY_THINK_STOPPING_COMPRESSION_OVERTHINKING="1"
fi
export REASONER_EARLY_THINK_STOPPING_STRING_FOR_REPEATED_CONCLUSION=$'\n\nI think I already thought for a long time, and I should be able to put the simple final answer after **Final Answer** with \\boxed{} and end thinking now. **Final Answer**\n'

if [ -z "${REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING}" ]; then
  export REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING="1"
fi
if [ -z "${REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING_STOP_TOKENS}" ]; then
  export REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING_STOP_TOKENS="18000"
fi
if [ -z "${REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING_STOP_ROUNDS}" ]; then
  export REASONER_EARLY_THINK_STOPPING_COMPRESSION_UNDERTHINKING_STOP_ROUNDS="20"
fi

export REASONER_EARLY_THINK_STOPPING_COMPRESSION_PROMPT_PREFIX="[unused9]系统：你生成综合质量（有用性，事实性，无害性）极好的回复。[unused10][unused9]系统：你是一个擅长于文章写作、文本分析、代码编写、数学运算、逻辑推理的AI助手，专注于针对用户的问题给出高质量解答。[unused10][unused9]用户："
export REASONER_EARLY_THINK_STOPPING_COMPRESSION_PROMPT_SUFFIX="[unused10][unused9]助手："

export REASONER_EARLY_THINK_STOPPING_COMPRESSION_ANSWER_CHECK_PROMPT_TEMPLATE=$'Check if the following sentences from a reasoning model reaches a conclusion, regardless of formatting or phrasing and solve the question. Return \'Yes\' if the content finds a solution, otherwise \'No\'. Return only \'Yes\' or \'No\' with no explanation.\nExample 1: Question: 2 + 3 = ?\nContent: The answer is 5.\n You should return Yes.\nExample 2: Question: 2 + 3 = ?\nContent: I think it should be 5, but I am not sure.\n You should return Yes.\nExample 3: Question: 2 + 3 = ?\nContent: Wait, I think I made a mistake.\n You should return Yes.\nExample 4: Question: If $f(x) = \\frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction.\nContent: 6 + 5 is 11, and 11 + 3 is 14. Yes, so 14/3. So, f(-2) + f(-1) + f(0) = 14/3.\n You should return Yes.\nExample 5: Question: If $f(x) = \\frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction.\nContent: Since all denominators are 3, we can add the numerators: 6 + 5 + 3 = 14. Therefore, the sum is 14/3.\n You should return Yes.\nExample 6: Question: If $f(x) = \\frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction.\nContent: Wait, another thought: When adding the fractions, is 14/3 the correct sum? Let\'s compute it in decimal to cross-verify. 14 divided by 3 is approximately 4.666...\n You should return No.\n \nQuestion: [QUESTION_PLACEHOLDER] \nContent: [CONTENT_PLACEHOLDER]\nDoes the content has a solution?\n[Yes/No]:'

export REASONER_EARLY_THINK_STOPPING_COMPRESSION_ANSWER_EQUIVALENCE_PROMPT_TEMPLATE=$'Check if the two answers provided are logically or mathematically equivalent, regardless of formatting or phrasing. Return \'Yes\' if they are equal in meaning/value and a valid solution to the question, otherwise \'No\'. Return only \'Yes\' or \'No\' with no explanation. \n\nExample 1:\nQuestion: 2 + 3 = ? \nAnswer1: the answer is 5.\nAnswer2: the answer seems to be five.\nAre they equivalent?\n[Yes/No]:Yes\n\nExample 2:\nQuestion:  Define\\[p = \\sum_{k = 1}^\\infty \\frac{1}{k^2} \\quad \\text{and} \\quad q = \\sum_{k = 1}^\\infty \\frac{1}{k^3}.\\]Find a way to write \\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3}\\]in terms of $p$ and $q.$ \nAnswer 1: (p - 1) - (q - 1) = p - 1 - q + 1 = p - q \\] Therefore, the original double sum is equal to \\(p - q\\).. \nAnswer 2: Given that the terms are decreasing and positive, the convergence seems plausible. However, since \\(p - q \\approx 0.4429\\) and our partial sum is about 0.3513, it\'s possible that the analytical result is correct. Therefore, unless there\'s a mistake in the analytical steps, the answer should be \\(p - q\\)..\nAre they equivalent?\n[Yes/No]:Yes\n\n\nQuestion: [QUESTION_PLACEHOLDER]\nAnswer 1: [ANSWER1_PLACEHOLDER].\n Answer 2: [ANSWER2_PLACEHOLDER].\nAre they equivalent?\n[Yes/No]:'

if [ -z "${REASONER_EARLY_THINK_STOPPING_USER_PATTERN}" ]; then
export REASONER_EARLY_THINK_STOPPING_USER_PATTERN='\[unused9\]用户：(.*?)\[unused10\]'
fi

if [ -z "${REASONER_EARLY_THINK_STOPPING_COMPRESSION_SERVER_URL}" ]; then
  export REASONER_EARLY_THINK_STOPPING_COMPRESSION_SERVER_URL="from_rank_table"
fi

if [ -z "${REASONER_EARLY_THINK_STOPPING_COMPRESSION_SERVER_INDEX}" ]; then
  export REASONER_EARLY_THINK_STOPPING_COMPRESSION_SERVER_INDEX="[0]"
fi

if [ -z "${REASONER_EARLY_THINK_STOPPING_COMPRESSION_SERVER_PORT}" ]; then
  export REASONER_EARLY_THINK_STOPPING_COMPRESSION_SERVER_PORT="8000"
fi

if [ -z "${REASONER_EARLY_THINK_STOPPING_COMPRESSION_SERVER_GROUP}" ]; then
  export REASONER_EARLY_THINK_STOPPING_COMPRESSION_SERVER_GROUP="1"
fi