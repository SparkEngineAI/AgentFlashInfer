from ..tool_register import tool_info_decorator
import logging
from typing import Any

logger = logging.getLogger(__name__)


@tool_info_decorator
def writer_subjective_task_done(
    final_article_path: str,
    article_summary: str,
    completion_status: str,
    completion_analysis: str
):
    """
    Reports the completion of a Writer Agent's task for long-form content.

    This function is called after all chapters or sections are written. It provides
    a summary of the complete article, its final completion status, an analysis of the
    writing process, and the storage path of the final consolidated document.

    Args:
        final_article_path (str): The file path where the final article is saved.
        article_summary (str): A comprehensive summary of the long-form article,
            including main themes, key points, and narrative structure.
        completion_status (str): The final status of the writing task. Must be
            one of 'completed', 'partial', or 'failed'.
        completion_analysis (str): An analysis of the writing project, covering
            coherence, quality, organization, flow, challenges encountered,
            and an overall evaluation of the content creation success.
    """
    completion_data = {
        "final_article_path": final_article_path,
        "article_summary": article_summary,
        "completion_status": completion_status,
        "completion_analysis": completion_analysis
    }

    return completion_data
