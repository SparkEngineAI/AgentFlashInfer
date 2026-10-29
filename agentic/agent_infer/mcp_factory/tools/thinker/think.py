import time
from ..tool_register import tool_info_decorator
from typing import Any


@tool_info_decorator
def think(thought: str) -> Any:
    """
    Use the tool to think about something. 
    It will not obtain new information or make any changes to the repository, but just log the thought. 
    Use it when complex reasoning or brainstorming is needed.

    Args:
        thought: Your thoughts.
    """

    return "You can proceed to invoke other tools if needed. "
