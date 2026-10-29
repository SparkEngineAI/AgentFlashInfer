from ..tool_register import tool_info_decorator
import logging
from typing import Any

logger = logging.getLogger(__name__)


@tool_info_decorator
def planner_subjective_task_done(final_article_path: str, task_summary: str, task_name: str, completion_status: str):
    """
    When the writer agent is executed, the task done tool is called to end the planner's task.

    Args:
        final_article_path (str): The file path where the final article is saved.
        task_summary (str): This field is mainly used to describe the main content of the article, briefly summarize it, and finally indicate the path where the final article is saved.
        task_name (str): The name of the task currently assigned to the agent, usually with underscores (e.g., 'web_research_ai_trends').
        completion_status (str): Final task status. Must be one of 'completed', 'partial', or 'failed'.
    """
    completion_data = {
        "final_article_path": final_article_path,
        "task_summary": task_summary,
        "task_name": task_name,
        "completion_status": completion_status,
    }

    return completion_data
