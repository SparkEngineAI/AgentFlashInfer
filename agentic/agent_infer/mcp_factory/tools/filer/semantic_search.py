
from .config import APIConfig
from config.config import WORKSPACE_PATH
from ..tool_register import tool_info_decorator
import logging
from typing import Any
from pathlib import Path
from dotenv import load_dotenv
import os
from .workspace_knowledge import get_workspace_knowledge_manager
import config.config as real_config
load_dotenv()
logger = logging.getLogger(__name__)


# @tool_info_decorator
def semantic_search(query: str,
                    session_id: str,
                    max_results: int = 5,
                    similarity_threshold: float = 0.7) -> Any:
    """
    Search semantically through workspace knowledge using OpenAI embeddings.
    Uses workspace-specific Faiss index with automatic file monitoring.
    """

    # Extract parameters
    config = APIConfig()
    # Import workspace knowledge manager

    # Extract workspace ID from path
    workspace_path = Path(os.path.join(real_config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)

    workspace_id = session_id
    workspace_base = str(real_config.WORKSPACE_PATH)

    # Get workspace knowledge manager
    knowledge_manager = get_workspace_knowledge_manager(config, workspace_base)

    if workspace_id not in knowledge_manager.workspace_indexes:
        knowledge_manager.get_or_create_workspace_index(workspace_id)
    # Perform workspace-specific search
    search_result = knowledge_manager.search_workspace(
        workspace_id=workspace_id,
        query=query,
        max_results=max_results,
        threshold=similarity_threshold
    )

    if search_result.get('success', False):
        # Add metadata about the search
        search_result['search_metadata'] = {
            'similarity_threshold': similarity_threshold,
            'max_results_requested': max_results,
            'embedding_model': 'text-embedding-3-small',
            'vector_store': 'Workspace Faiss IndexFlatIP',
            'isolation': 'Per-workspace isolated search',
            'file_monitoring': 'Automatic with watchdog'
        }

    return search_result
