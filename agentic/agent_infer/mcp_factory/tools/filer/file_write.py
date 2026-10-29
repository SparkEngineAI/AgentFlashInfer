import asyncio

import aiofiles

from ..tool_register import tool_info_decorator
import logging
from pathlib import Path
from typing import Any
import os
import config.config as config
logger = logging.getLogger(__name__)


@tool_info_decorator
async def file_write(
    file_path: str,
    content: str,
    session_id: str,
    encoding: str = 'utf-8',
    create_dirs: bool = True
) -> Any:
    """
    Appends text content to a file in the workspace.

    This function adds the given `content` to the end of the file specified
    by `file_path`. If the parent directories for the file do not exist,
    they will be created automatically by default.

    Args:
        file_path (str): The relative path to the file within the workspace
        content (str): The text content to append to the file.
        session_id: The conversation ID
        encoding (str, optional): The character encoding for writing.
                                  Defaults to 'utf-8'.
        create_dirs (bool, optional): If True, automatically create parent
                                      directories for the file_path.
                                      Defaults to True.

    Returns:
        str: A confirmation message indicating how many characters were
             successfully written to the file.
    """
    # Get the current directory as a Path object
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    await asyncio.to_thread(lambda: workspace_path.mkdir(exist_ok=True))

    # Join paths using the intuitive '/' operator
    full_path = workspace_path / file_path

    if create_dirs:
        # This now works because full_path is a Path object
        await asyncio.to_thread(lambda: full_path.parent.mkdir(parents=True, exist_ok=True))

    async with aiofiles.open(full_path, "a", encoding=encoding) as f:
        await f.write(content)

    return f"Written {len(content)} characters to {file_path}"
