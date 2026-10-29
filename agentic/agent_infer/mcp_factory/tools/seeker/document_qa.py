import asyncio

from ..tool_register import tool_info_decorator, base_client, base_client_async
from ..filer.file_read import file_read
import logging
from typing import Any, List, Dict
import os
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import config.config as config
import time
load_dotenv()

logger = logging.getLogger(__name__)


@tool_info_decorator
async def document_qa(
    tasks: List[Dict],
    session_id: str,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    max_workers: int = 5
) -> Any:
    """
    Answer questions based on content stored in local files. 
    Each task contains a file path and a question to be answered using that file's content.

    Args:
        tasks: List of task dictionaries containing:
            - file_path: Relative path to the file (relative to workspace root) to read
            - question: Question to ask about this file
        session_id: The conversation ID
        temperature: Creativity level for the AI response (0-1)
        max_tokens: Maximum tokens for the AI response
        max_workers: Maximum number of concurrent model API requests
    """
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    await asyncio.to_thread(lambda: workspace_path.mkdir(exist_ok=True))

    async def process_single_task(task: Dict) -> Dict:
        file_path = task['file_path']
        question = task['question']

        # 1. 读取文件内容
        read_result = await file_read(file_path, session_id)

        content = read_result

        # 2. 构建系统提示
        system_prompt = (
            "You are an expert document analyst. Answer the user's question "
            "based ONLY on the provided context. If the answer cannot be found "
            "in the context, say 'I don't know'.\n\n"
            "CONTEXT:\n{context}"
        ).format(context=content)

        # 3. 调用大模型API
        def _extract_response_data( response: Any) -> tuple:
            """
            Extract response content and token usage from LLM response.

            Returns:
                Tuple of (output, prompt_tokens, completion_tokens, total_tokens)
            """
            # Extract response content
            if hasattr(response, 'choices') and response.choices:
                output = response.choices[0].message.content
            else:
                if isinstance(response, dict):
                    output = response["choices"][0]["message"]["content"]
                else:
                    output = str(response)

            # Extract token usage if available
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

            if hasattr(response, 'usage'):
                prompt_tokens = getattr(response.usage, 'prompt_tokens', None)
                completion_tokens = getattr(response.usage, 'completion_tokens', None)
                total_tokens = getattr(response.usage, 'total_tokens', None)
            elif isinstance(response, dict):
                if 'usage' in response:
                    prompt_tokens = response['usage'].get("prompt_tokens", None)
                    completion_tokens = response['usage'].get("completion_tokens", None)
                    total_tokens = response['usage'].get("total_tokens", None)

            return output, prompt_tokens, completion_tokens, total_tokens
        st = int(time.time() * 1000)
        response = await base_client_async.chat.completions.create(model=config.DEFAULT_MODEL_NAME,
                                                       messages=[
                                                           {"role": "system",
                                                               "content": system_prompt},
                                                           {"role": "user",
                                                               "content": question}
                                                       ],
                                                       temperature=temperature,
                                                       max_tokens=max_tokens,
                                                       stream=False
                                                       )
        content, prompt_tokens, completion_tokens, total_tokens = _extract_response_data(response)

        et = int(time.time() * 1000)
        return {
            'file_path': file_path,
            'question': question,
            'success': True,
            'answer': content,
            'metadata': {
                'file_size': len(content),
                'line_count': len(content.splitlines()),
                'llm_calls': [
                    {
                        "model": config.DEFAULT_MODEL_NAME,
                        "messages": [],
                        "start_time": st,
                        "end_time": et,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                ]
            }
        }
    coroutinetasks = [
        process_single_task(task)
        for task in tasks
    ]
    results = await asyncio.gather(*coroutinetasks, return_exceptions=True)
    # 5. 保持原始任务顺序
    task_order = {task['file_path']: i for i, task in enumerate(tasks)}
    results.sort(key=lambda x: task_order.get(x['file_path'], float('inf')))

    return results
