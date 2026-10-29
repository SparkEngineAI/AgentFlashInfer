from .config import APIConfig
from ..tool_register import tool_info_decorator
import logging
from typing import Any, Dict
from pathlib import Path
from dotenv import load_dotenv
import os
from .workspace_knowledge import get_workspace_knowledge_manager
import config.config as real_config
load_dotenv()
logger = logging.getLogger(__name__)


@tool_info_decorator
def knowledge_status(session_id: str) -> Any:
    """
    Get status and statistics about the workspace knowledge management system.
    """
    config = APIConfig()
    # Extract workspace ID from path
    workspace_path = Path(os.path.join(real_config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)

    workspace_id = session_id
    workspace_base = str(real_config.WORKSPACE_PATH)

    # Get workspace knowledge manager
    knowledge_manager = get_workspace_knowledge_manager(config, workspace_base)
    if workspace_id not in knowledge_manager.workspace_indexes:
        knowledge_manager.get_or_create_workspace_index(workspace_id)

    # Get workspace stats
    stats = knowledge_manager.get_workspace_stats(workspace_id)

    # Add system information
    stats['vector_store_system'] = 'Workspace Faiss (High Performance)'
    stats['performance'] = 'O(log n) search complexity'
    stats['isolation'] = 'Per-workspace isolated indexing'
    stats['file_monitoring'] = 'Automatic with watchdog'
    stats['embedding_model'] = 'text-embedding-3-small'

    return stats
