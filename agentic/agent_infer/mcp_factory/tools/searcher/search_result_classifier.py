import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any

import aiofiles

from agent_infer.mcp_factory.tools.tool_register import tool_info_decorator, base_client_async
from agent_infer.mcp_factory.tools.tool_result_format import MCPToolResult
import logging
import time
from config import config

logger = logging.getLogger(__name__)


@tool_info_decorator
async def search_result_classifier(
        # save_analysis_file_path: str,
        outline: str,
        key_files: List[Dict],
        session_id: str,
        model: str = "pangu_auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
) -> Any:
    """
        Classify and organize search result files according to a structured outline for comprehensive long-form content generation.

        Args:
            outline: Structured outline defining the sections and subsections for organizing the long-form content
            key_files: List of key files to classify
            session_id: str,
            model: AI model to use for classification and organization
            temperature: Creativity level for the AI classification (0-1)
            max_tokens: Maximum tokens for the AI response
        """
    try:
        # Get configuration
        from config.config import get_storage_config
        # model_config = get_model_config()
        storage_config = get_storage_config()

        # Use config values or defaults
        if temperature is None:
            temperature =  getattr(config, 'model_temperature', 0.3)
        if max_tokens is None:
            max_tokens = getattr(config, 'model_max_tokens', 2000)

        logger.debug(f"Starting search result classification: outline={outline}, key_files={key_files}")

        key_files_dict = {}
        # Create full path relative to workspace using config
        analysis_path = storage_config.get('document_analysis_path', './doc_analysis')
        file_analysis_list = await load_json(file_path=f"{analysis_path}/file_analysis.jsonl", session_id=session_id)

        for file_info in file_analysis_list.data:
            if file_info.get('file_path'):
                key_files_dict[file_info.get('file_path')] = file_info

        prompt_files = ""
        if key_files:
            prompt_files += f"Key Files with Multi-Dimensional Analysis:\n"
            for i, file_info in enumerate(key_files, 1):
                if file_info.get('file_path') in key_files_dict:
                    file_info = key_files_dict[file_info.get('file_path')]
                    prompt_files += f"\n{i}. File: {file_info.get('file_path', 'Unknown')}\n"
                    prompt_files += f"   Document Time: {file_info.get('doc_time', 'Not specified')}\n"
                    prompt_files += f"   Source Authority: {file_info.get('source_authority', 'Not specified')}\n"
                    prompt_files += f"   Core Content: {file_info.get('core_content', 'Not specified')}\n"
                    prompt_files += f"   Task Relevance: {file_info.get('task_relevance', 'Not specified')}\n"
                    prompt_files += f"   Information Richness: {file_info.get('information_richness', 'Not specified')}\n"
                    prompt_files += "\n"

            prompt_files += "\n"

        system_prompt = (
            "You are an expert content organizer specializing in multi-dimensional file classification. "
            "Your task is to classify research files according to a given outline by according the five key dimensions.\n\n"

            "FIVE KEY DIMENSIONS:\n"
            "1. DOCUMENT TIME: Consider the temporal relevance and recency of the source\n"
            "2. SOURCE AUTHORITY: Consider the credibility and expertise of the source\n"
            "3. CORE CONTENT: Focus on the main themes and key insights\n"
            "4. TASK RELEVANCE: Assess alignment with the specific research goals\n"
            "5. INFORMATION RICHNESS: Determine the information richness of the document\n"

            "CLASSIFICATION INSTRUCTIONS:\n"
            "1. Read the outline and understand its structure\n"
            "2. Analyze each file across the five dimensions provided\n"
            "3. Accept files with moderate source authority, moderate information richness and reasonable task relevance\n"
            "4. Use core content insights to find connections to outline sections, ensure the relevance of the assignment.\n"
            "5. Files should be assigned to multiple sections when they contain information spanning different topics\n"
            "6. Prioritize comprehensive coverage: Ensure every paragraph/chapter gets file assignments when any relevant content exists\n"
            "7.Note that if the first chapter is an abstract or introduction, a corresponding file must be assigned and there cannot be an empty file.\n"

            "CRITICAL REQUIREMENT - OUTLINE STRUCTURE PRESERVATION:\n"
            "- The number of files assigned to each chapter cannot exceed 11, so if the number of files exceeds 11, you need the most relevant files.\n"
            "- When the outline is divided into paragraphs/sections, you must preserve the original content and structure exactly.\n "
            "- Do NOT modify, rephrase, or alter any paragraph titles, headings, or structural elements from the original outline. \n"
            "- Keep all outline content completely intact in your output, including formatting and wording.\n"
            "- You should split each paragraph according to the granularity of the first-level heading. You are not allowed to separate any second-level headings or smaller headings under the first-level heading into individual paragraphs.\n"
            "- Formatting requirements for split paragraphs: 1. You are not allowed to modify the format of the paragraphs in the original outline, including all bold symbols, etc.; 2. If the given outline includes the title of the article, you must include the article title in the first paragraph, maintaining its original bold symbols.\n"

            "Strictly follow the following format for output:\n"
            "paragraph 1: ...\n"
            "file_path_list: file_path1, file_path2, ...\n"
            "\n"
            "paragraph 2: ...\n"
            "file_path_list: file_path3, file_path4, ...\n"
            "..."
        )

        # 构建用户提示 - 仅包含输入数据
        user_prompt = f"""
OUTLINE TO ORGANIZE CONTENT:
{outline}

{prompt_files}
"""

        try:
            # Add retry logic for AI model call
            max_retries = 5
            response = None

            for attempt in range(max_retries):
                try:
                    response = await base_client_async.chat.completions.create(model=config.DEFAULT_MODEL_NAME,
                                                                               messages=[
                                                                                    {"role": "system", "content": system_prompt},
                                                                                    {"role": "user", "content": user_prompt + " /no_think"}
                                                                                ],
                                                                               temperature=temperature,
                                                                               max_tokens=max_tokens,
                                                                               stream=False,
                                                                               timeout=getattr(config, 'model_request_timeout', 180)
                                                                               )

                    logger.debug(f"API response received")
                    break  # Success, exit retry loop
                except Exception as e:
                    logger.warning(f"LLM API call attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise e  # Last attempt, re-raise the exception
                    time.sleep(5)  # Simple 1 second delay between retries

            if response is None:
                raise Exception("Failed to get response after all retries")

            # ai_response = response.choices[0].message.content.strip()
            # Extract response content
            if hasattr(response, 'choices') and response.choices:
                output = response.choices[0].message.content
            else:
                if isinstance(response, dict):
                    output = response["choices"][0]["message"]["content"]
                else:
                    output = str(response)
            ai_response = output.strip()


            # 切换保存的方式
            conversation_history = [
                # {"role": "system", "content": "system 1"},
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + " /no_think"},
                {"role": "assistant", "content": ai_response}
            ]

            return {
                "success": True,
                "data": ai_response.split('think>')[-1].strip(),
            }

        except Exception as e:
            logger.error(f"AI model call failed: {e}")
            return {
                "success":False,
                "error": f"AI classification failed: {str(e)}"
            }

    except Exception as e:
        logger.error(f"Search result classifier failed: {e}")
        return {
            "success":False,
            "error": str(e)
        }

async def load_json(file_path: str, session_id:str, encoding: str = 'utf-8') -> MCPToolResult:
    """
    Read JSON format file
    """
    try:
        full_path = _safe_join(file_path, session_id)
        if not full_path.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
        if not full_path.exists():
            return MCPToolResult(
                success=False,
                error=f"File does not exist: {file_path}"
            )

        res = []
        try:
            async with aiofiles.open(full_path, "r", encoding=encoding, errors='ignore') as f:
                idx = 0
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ele = json.loads(line)
                        res.append(ele)
                    except Exception as e:
                        logger.warning(f"Failed to parse JSON on line {idx + 1} in {full_path}: {e}")
                    finally:
                        idx += 1  # 无论成功失败都递增行号
        except Exception as e:
            logger.error(f"Failed to open file {full_path}: {e}")
            raise

        return MCPToolResult(
            success=True,
            data=res,
            metadata={
                'line_count': len(res),
                'encoding': encoding
            }
        )

    except Exception as e:
        logger.error(f"File read failed: {e}")
        return MCPToolResult(success=False, error=str(e))

def _safe_join(path: str, session_id:str) -> Path:
    if os.path.isabs(path):
        raise Exception(f"Path '{path}' is absolute. Only relative paths are allowed.")
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    joined_path = os.path.join(workspace_path, path)
    full_joined_path = os.path.realpath(joined_path)
    full_workspace_path = os.path.realpath(workspace_path)
    if not full_joined_path.startswith(full_workspace_path):
        raise Exception(f"Path '{path}' is outside workspace directory.")
    return Path(full_joined_path)