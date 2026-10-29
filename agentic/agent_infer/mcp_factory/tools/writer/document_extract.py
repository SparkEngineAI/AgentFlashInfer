import asyncio
import os
import aiofiles
from ..tool_register import tool_info_decorator, base_client, base_client_async
import logging
import json
from typing import Any, List, Dict
from ..filer.file_read import file_read
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import config.config as config

load_dotenv()
logger = logging.getLogger(__name__)


@tool_info_decorator
async def document_extract(
    task_info: List[Dict],
    output_file: str,
    session_id: str,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    max_workers: int = 5
) -> Any:
    """
    Extract structured information from documents using LLM analysis.
    Each task contains a file path and task content describing the relevance assessment purpose.

    Args:
        task_info: List of task dictionaries containing:
            - file_path: Relative path to the file (relative to workspace root) to read
            - task_content: Description of the task for which the file is being evaluated
        output_file: Path to save the JSONL output file
        session_id: The conversation ID
        temperature: Creativity level for the AI response (0-1)
        max_tokens: Maximum tokens for the AI response
        max_workers: Maximum number of concurrent model API requests
    """
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    await asyncio.to_thread(lambda: workspace_path.mkdir(exist_ok=True))

    async def process_single_task(task: Dict) -> Dict:
        file_path = task['file_path']
        task_content = task['task_content']

        # 1. 读取文件内容
        read_result = await file_read(file_path, session_id)
        # 如果文件读取失败，会直接引发异常，由外部捕获
        content = read_result

        # 2. 构建用户输入
        user_input = f"""Document Content:
            {content}

            Task for Relevance Assessment:
            {task_content}"""
        # 3. 构建系统提示
        system_prompt = (
            "You are a text expert. Next, you will be given a document and task content. You need to analyze this document carefully and then provide multiple dimensional information for this document.\n\n"

            "The following are some dimensional information extracted:\n"
            "1. Web page time: According to the content of the document, extract the web page time of the document content. If it cannot be judged, it is expressed as \"unable to determine the web page time\"; otherwise, the time of the web page is output, accurate to the month, in the format of \"YYYY year MM month\", such as \"2023 June\";\n"
            "2. Authority: According to the information of the document, judge the source of the web page to confirm the credibility of the web page.\n"
            "3. Relevance: According to the current task (task_content) and the given document, judge whether the current document is related to the current task.\n"
            "4. Core content: Based on this document, you make a core content summary to ensure the richness of information, with a word count of about 200 words.\n"
            "Information richness: Estimate the total word count of substantive content in the document. Less than 200 words indicates scarcity; over 800 words suggests high richness; between these thresholds denotes moderate richness. Be careful not to just give the word count results, but also give a corresponding text description of how informative the content is.\n"

            "Note:\n1. Ensure the document's language aligns with the extracted dimensions (e.g., Chinese content requires Chinese extraction).\n2. For **source_authority** and **task_relevance**, first provide a brief description before concluding.  \n"
            "- **Authority**: Briefly assess the source's credibility (e.g., expertise, reputation). *Conclusion*: [High/Medium/Low].  \n"
            "- **Relevance**: Summarize content alignment with the topic. *Conclusion*: [High/Medium/Low].\n"

            "The final output format must be a valid JSON object:\n"
            "{\n"
            "  \"doc_time\": \"xxx\",\n"
            "  \"source_authority\": \"xxx\",\n"
            "  \"task_relevance\": \"xxx\",\n"
            "  \"core_content\": \"xxx\",\n"
            "  \"information_richness\": \"xxx\"\n"
            "}\n\n"
            "Important: Return ONLY the JSON object, no additional text or formatting."
        )

        # 4. 调用模型 API (如果失败会直接抛出异常)
        response = await base_client_async.chat.completions.create(model=config.DEFAULT_MODEL_NAME,
                                                       messages=[
                                                           {"role": "system",
                                                               "content": system_prompt},
                                                           {"role": "user",
                                                               "content": user_input}
                                                       ],
                                                       temperature=temperature,
                                                       max_tokens=max_tokens,
                                                       stream=False
                                                       )
        raw_response = response.choices[0].message.content

        # 5. 提取并解析 JSON (如果解析失败会直接抛出异常)
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        json_str = raw_response[json_start:json_end]
        extraction_result = json.loads(json_str)

        # 确保 file_path 被包含
        extraction_result['file_path'] = file_path

        # 6. 准备并返回成功的任务结果
        return {
            'file_path': file_path,
            'success': True,
            'extraction': extraction_result,
            'metadata': {
                'file_size': len(content),
                'line_count': len(content),
                'response_length': len(raw_response)
            }
        }

    # 7. 并发处理所有任务
    tasks = [
        process_single_task(task)
        for task in task_info
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 8. 保持原始任务顺序
    task_order = {task['file_path']: i for i, task in enumerate(task_info)}
    results.sort(key=lambda x: task_order.get(x['file_path'], float('inf')))

    # 9. 将成功的结果保存到 JSONL 文件
    successful_extractions = []
    output_path = Path(workspace_path) / \
        output_file if workspace_path else Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            # 假设所有任务都成功，因为任何失败都会提前抛出异常
            if result['success'] and result['extraction']:
                await f.write(json.dumps(result['extraction'], ensure_ascii=False))
                await f.write('\n')
                successful_extractions.append(result['extraction'])

    # 10. 统计结果并返回
    successful_tasks = len(successful_extractions)

    return f"Document extraction task completed successfully. Processed {len(task_info)} files with {successful_tasks} successful extractions. Results are stored in {output_file} as JSONL format. Use file reading tools to examine the detailed extraction results."
