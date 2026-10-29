from ..tool_register import tool_info_decorator
import logging
from typing import Any, List, Dict

logger = logging.getLogger(__name__)


@tool_info_decorator
def info_seeker_objective_task_done(
    task_summary: str,
    task_name: str,
    key_files: List[Dict[str, Any]],
    completion_status: str
):
    """
    Structured reporting of task completion details including summary, decisions, outputs, and status

    Args:
        task_summary (str): Comprehensive markdown covering what the agent was asked to do, steps taken, tools used, key findings, files created, challenges, and final deliverables.
        task_name (str): The name of the task currently assigned to the agent, usually with underscores (e.g., 'web_research_ai_trends')
        key_files (List[Dict[str, Any]]): List of key files generated or modified during the task, with their details. Each dictionary in the list should contain:
            - 'file_path' (str): Relative path to the created/modified file.
            - 'desc' (str): File contents and creation purpose
            - 'is_final_output_file' (bool): Whether file is primary deliverable
        completion_status (str): Final task status. Must be one of 'completed', 'partial', or 'failed'.
    """
    completion_data = {
        "task_summary": task_summary,
        "task_name": task_name,
        "key_files": key_files,
        "completion_status": completion_status,
    }

    return completion_data
