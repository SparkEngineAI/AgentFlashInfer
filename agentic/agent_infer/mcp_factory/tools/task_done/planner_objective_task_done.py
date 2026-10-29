from ..tool_register import tool_info_decorator
import logging
from typing import Any, List, Dict

logger = logging.getLogger(__name__)


@tool_info_decorator
def planner_objective_task_done(
    task_summary: str,
    task_name: str,
    key_files: List[Dict[str, Any]],
    completion_status: str,
    final_answer: str
):
    """
    Provides a structured report on the completion details of a task, including a summary, key decisions, and the final answer.

    Args:
        task_summary (str): A comprehensive markdown report detailing what the agent was asked to do, the steps taken, tools used, key findings, files created, and any challenges encountered.
        task_name (str): The name of the task currently assigned to the agent, typically using underscores (e.g., 'web_research_ai_trends').
        key_files (List[Dict[str, Any]]): A list of key files that were generated or modified during the task. Each dictionary in the list should contain 'file_path' (str), 'desc' (str), and 'is_final_output_file' (bool).
        completion_status (str): The final status of the task. Must be one of 'completed', 'partial', or 'failed'.
        final_answer (str): The definitive final response to be displayed to the user.
    """
    completion_data = {
        "task_summary": task_summary,
        "task_name": task_name,
        "key_files": key_files,
        "completion_status": completion_status,
        "final_answer": final_answer
    }

    return completion_data
