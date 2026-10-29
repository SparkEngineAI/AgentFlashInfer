from ..tool_register import tool_info_decorator
from typing import Any


@tool_info_decorator
def reflect(reflect: str) -> Any:
    """
    When multiple attempts yield no progress, 
    use this tool to reflect on previous reasoning and planning, 
    considering possible overlooked clues and exploring more possibilities. 
    It will not obtain new information or make any changes to the repository.

    Args:
        reflect: The specific content of your reflection.
    """

    return "You can proceed to invoke other tools if needed. "
