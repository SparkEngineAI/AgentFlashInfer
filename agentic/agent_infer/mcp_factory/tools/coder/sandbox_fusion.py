from typing import Any
import logging
import requests
from ..tool_register import tool_info_decorator
logger = logging.getLogger(__name__)

# ================ CODE SANDBOX TOOLS ================


@tool_info_decorator
def sandbox_fusion_tool(code: str, language: str) -> Any:
    """
    A tool that can execute code in a sandbox environment. Useful for running and testing code snippets.
    The tool service primarily offers two functions: running code and evaluating the correctness of problems.
    Execute a code snippet in a secure sandbox environment.

    Args:
        prompt: The code snippet to be executed.
        language: The programming language of the code snippet (e.g., "python").
        ctx: MCP context for logging and progress tracking. Defaults to None.

    Returns: MCPToolResult
    """
    response = requests.post('http://127.0.0.1:8081/run_code', json={
        'code': code,
        'language': language,
    }, timeout=30)
    response.raise_for_status()
    result = response.json()
    code_result = result["run_result"]["stdout"]
    return code_result
