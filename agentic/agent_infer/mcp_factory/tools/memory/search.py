from ..tool_register import tool_info_decorator
import logging
from typing import Any, List
from dotenv import load_dotenv
import config.config as config
from fusion_mem.client import AsyncMemoryClient

load_dotenv()

logger = logging.getLogger(__name__)
MEMORY_BASE_URL = config.MEMORY_BASE_URL

@tool_info_decorator
async def search_shared_memories(
    query: str,
    session_id: str,
) -> Any:
    """
    Retrieving memories related to the question from the memory system

    Args:
        query: The question for memory retrieval
    """
    memories = await AsyncMemoryClient(MEMORY_BASE_URL).search_shared_memories(query=query)
    if not memories:
        return []
    results_list = memories.get('results', [])
    sorted_results = sorted(results_list, key=lambda x: x['metadata']['time_stamp'])
    print(sorted_results)
    results = [memory["memory"] for memory in sorted_results]
    return {
        "memories": results
    }
