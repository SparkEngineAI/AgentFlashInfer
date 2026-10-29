from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class MCPToolResult:
    """Standard result format for MCP func_tools"""
    success: bool
    data: Any = None
    error: str = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata or {}
        }
