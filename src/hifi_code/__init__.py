"""hifi_code — an ast-bro-driven autonomous coding loop.

Mirrors the search_agent layout: src-layout package, Pydantic settings, stdio and
MCP transports, newline-delimited JSON output.
"""

from .config import Settings
from .planner import Planner

__all__ = ["Planner", "Settings"]
