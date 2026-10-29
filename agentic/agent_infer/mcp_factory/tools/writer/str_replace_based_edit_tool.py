import asyncio

import aiofiles

from ..tool_register import tool_info_decorator
import logging
from typing import Any
from pathlib import Path
from dotenv import load_dotenv
import os
import config.config as config
load_dotenv()
logger = logging.getLogger(__name__)


@tool_info_decorator
async def str_replace_based_edit_tool(
    action: str,
    file_path: str,
    session_id: str,
    content: str = None,
    old_str: str = None,
    new_str: str = None,
    line_number: int = None,
    max_char_len: int = 10000,

) -> Any:
    """
    Focused file editing tool for text replacement, line insertion, and file deletion

    Args:
        action: 'create', 'view', 'str_replace', 'insert', 'delete'
        file_path: Path to the file
        session_id: The conversation ID
        content: Content for insert action
        old_str: String to replace (for str_replace)
        new_str: Replacement string (for str_replace)
        line_number: Line number for insert action
    """
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    await asyncio.to_thread(lambda: workspace_path.mkdir(exist_ok=True))
    full_path = workspace_path / file_path

    if action == 'str_replace':
        original_content = await read_text_async(full_path)

        new_content = original_content.replace(old_str, new_str)
        await write_text_async(full_path, new_content)

        result_dict = {
            "data": f"Replaced {original_content.count(old_str)} occurrence(s)",
            "metadata": {
                'old_size': len(original_content),
                'new_size': len(new_content)
            }
        }
        return result_dict

    elif action == 'insert':
        lines = await read_text_async(full_path)
        lines = lines.splitlines()
        lines.insert(line_number, content)
        await write_text_async(full_path, '\n'.join(lines))
        result_dict = {
            "success": True,
            "data": f"Inserted content at line {line_number}",
            "metadata": {
                'new_line_count': len(lines)
            }
        }
        return result_dict

    elif action == 'delete':
        await asyncio.to_thread(full_path.unlink)
        return f"Deleted file: {file_path}"

    elif action == 'create':
        exists = await asyncio.to_thread(full_path.exists)
        if exists:
            return {
            "success": False,
            "data": f"File already exists: {file_path}",
        }

        await asyncio.to_thread(lambda: full_path.parent.mkdir(parents=True, exist_ok=True))
        await write_text_async(full_path, content or '')
        return {
            "success": True,
            "data": f"File created: {file_path}",
            "metadata": {
                'file_size': full_path.stat().st_size
            }
        }
    elif action == 'view':
        exists = await asyncio.to_thread(full_path.exists)
        if not exists:
            return {
            "success": False,
            "data": f"File does not exist: {file_path}",
        }

        content = await read_text_async(full_path)
        if len(content) > max_char_len:
            content = ("Due to the content being too long, only the first 10,000 characters are returned. "
                       "It is recommended to use other tools such as `document_qa` to extract the required content from the file. "
                       "Below is the returned portion of the file content: \n\n") + content[:max_char_len]

        return {
            "success": True,
            "data": content,
            "metadata": {
                'file_size': len(content),
                'line_count': len(content.splitlines())
            }
        }
    else:
        return f"Unknown action: {action}. Valid actions are: str_replace, insert, delete"

async def read_text_async(path):
    async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
        return await f.read()

async def write_text_async(path, content):
    async with aiofiles.open(path, mode='w', encoding='utf-8') as f:
        await f.write(content)
