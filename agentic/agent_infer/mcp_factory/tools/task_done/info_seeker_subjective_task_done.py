from ..tool_register import tool_info_decorator
import logging
from typing import Any, List, Dict

logger = logging.getLogger(__name__)


@tool_info_decorator
def info_seeker_subjective_task_done(
    task_summary: str,
    key_files: List[Dict[str, str]],
    completion_status: str,
    completion_analysis: str
):
    """
    Reports the completion of an Information Seeker Agent's task, including a summary and related files.

    Args:
        task_summary (str): A simple summary of the information collected for the current task and any new discoveries made.
        key_files (List[Dict[str, str]]): A list of files highly relevant to this task. Each dictionary should contain a 'file_path' key with the relative path to the file.
        completion_status (str): The final status of the information gathering task. Must be one of 'completed', 'partial', or 'failed'.
        completion_analysis (str): A brief analysis of the task's completion quality, information thoroughness, and any limitations or gaps.
    """
    completion_data = {
        "task_summary": task_summary,
        "key_files": key_files,
        "completion_status": completion_status,
        "completion_analysis": completion_analysis
    }

    return completion_data
