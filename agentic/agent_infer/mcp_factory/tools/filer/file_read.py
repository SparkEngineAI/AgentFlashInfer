import asyncio

import aiofiles
import config.config as config
from pathlib import Path
import os
from ..tool_register import tool_info_decorator
import logging
from typing import Any
logger = logging.getLogger(__name__)


@tool_info_decorator
async def file_read(file_path: str, session_id: str, encoding: str = 'utf-8') -> Any:
    """
    Read file content

    Args:
        file_path (str): The relative path of the file to read.
        session_id: The conversation ID
        encoding (str, optional): The file encoding to use. Defaults to 'utf-8'.
    Returns:
        Any: The file's content
    """
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    await asyncio.to_thread(lambda: workspace_path.mkdir(exist_ok=True))
    full_path = os.path.join(workspace_path, file_path)
    exists = await asyncio.to_thread(os.path.exists, full_path)
    if not exists:
        return f"File does not exist: {file_path}"

    async with aiofiles.open(full_path, mode='r', encoding=encoding) as f:
        content = await f.read()

    if len(content) > 10000:
        content = "Due to the content being too long, only the first 10,000 characters are returned. It is recommended to use other tools such as `document_qa` to extract the required content from the file. Below is the returned portion of the file content: \n\n" + \
            content[:10000]

    return content
