from ..tool_register import tool_info_decorator
import logging
import subprocess
import re
import os
from pathlib import Path
from typing import Any
import config.config as config
logger = logging.getLogger(__name__)


@tool_info_decorator
def bash(
    command: str,
    session_id: str,
    timeout: int = 30,
    capture_output: bool = True,
    working_directory: str = None
) -> Any:
    """
    Execute bash command

    Args:
        command: Command to execute
        session_id: The conversation ID
        timeout: Command timeout in seconds
        capture_output: Whether to capture stdout/stderr
        working_directory: Directory to run command (relative to session workspace)
    """
    # Security check - prevent dangerous commands
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)
    dangerous_patterns = [
        r'rm\s+-rf\s+/',
        r'sudo\s+rm',
        r'mkfs',
        r'dd\s+if=.*of=/dev/',
        r'>\s*/dev/sd[a-z]',
        r'cat\s+.*>\s*/dev/sd[a-z]'
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            raise ValueError(
                f"Potentially dangerous command blocked: {command}")

    # Set working directory
    cwd = workspace_path
    if working_directory:
        # Ensure working_directory is relative to session workspace for security
        cwd = workspace_path / working_directory
        if not cwd.exists():
            raise FileNotFoundError(
                f"Working directory does not exist: {working_directory}")

    # Execute command
    result = subprocess.run(
        command,
        shell=True,
        cwd=str(cwd),
        capture_output=capture_output,
        text=True,
        timeout=timeout
    )

    result_dict = {
        "data": {
            'stdout': result.stdout if capture_output else None,
            'stderr': result.stderr if capture_output else None,
            'returncode': result.returncode,
            'command': command
        },
        "metadata": {
            'execution_time': timeout,
            'working_directory': str(cwd)
        }
    }
    return result_dict
