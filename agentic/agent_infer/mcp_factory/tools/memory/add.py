from ..tool_register import tool_info_decorator
import logging
from typing import Any, List,Dict
from dotenv import load_dotenv
import config.config as config
from fusion_mem.client import MemoryClient

load_dotenv()

logger = logging.getLogger(__name__)
MEMORY_BASE_URL = config.MEMORY_BASE_URL

def add_memories(
    memory_contents: List[Dict[str, str]],
    session_id: str,
) -> Any:
    """
    Add memories to memory system

    Args:
        memory_contents: List of memory to be added containing:
            - content: The textual content to be stored
            - role: The speaker in the conversation ("user" or "assistant" or "system")
        session_id: The conversation ID
    """
    if not MEMORY_BASE_URL:
        logger.warning("No MEMORY_BASE_URL set")
        return
    # contents = [{"content": memory_content} for memory_content in memory_contents]
    print("memory_contents", memory_contents)
    MemoryClient(MEMORY_BASE_URL).add(messages=memory_contents)
    return
