"""MCP server transport: expose the hifi_code loop as a tool.

Mirrors search_agent.adapters.mcp.MCPAdapter: builds a Tool definition and a
low-level Server, then runs over stdio.
"""

import asyncio
import json
import logging

import mcp.types as types
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server

from ..config import Settings
from .stdio import _drive

logger = logging.getLogger(__name__)


def _run_sync(settings: Settings, goal: str, root: str, max_steps: int) -> str:
    """Drive the loop to completion and return newline-joined JSON lines."""
    original = settings.max_steps
    settings.max_steps = max_steps
    lines = []

    async def _collect():
        async for result in _drive(settings, goal, root):
            lines.append(json.dumps(result, ensure_ascii=False))

    try:
        asyncio.run(_collect())
    finally:
        settings.max_steps = original
    return "\n".join(lines)


class MCPAdapter:
    """MCP server that exposes the coding loop as a single ``code_loop`` tool."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.tool_def = types.Tool(
            name="code_loop",
            title="Run the hifi_code coding loop",
            description=(
                "Runs an ast-bro-driven coding loop to accomplish a coding goal. "
                "Emits newline-delimited JSON for each loop step."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The coding goal to accomplish.",
                    },
                    "root": {
                        "type": "string",
                        "description": "Codebase root to seed the loop from.",
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum loop steps (default from settings).",
                    },
                },
                "required": ["goal"],
            },
        )
        self.server = Server(
            "hifi_code",
            on_list_tools=self._list_tools,
            on_call_tool=self._call_tool,
        )

    async def _list_tools(self, ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[self.tool_def])

    async def _call_tool(
        self, ctx, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        try:
            name = params.name
            arguments = params.arguments or {}

            if name == "code_loop":
                goal = arguments.get("goal", "")
                if not goal:
                    return types.CallToolResult(
                        is_error=True,
                        content=[types.TextContent(
                            type="text", text="Error: 'goal' is required")],
                    )
                root = arguments.get("root", ".")
                max_steps = arguments.get("max_steps", self.settings.max_steps)
                output = _run_sync(self.settings, goal, root, int(max_steps))
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=output or "done")]
                )

            return types.CallToolResult(
                is_error=True,
                content=[types.TextContent(
                    type="text", text=f"Unknown tool: {name}")],
            )
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.exception("error in hifi_code tool call")
            return types.CallToolResult(
                is_error=True,
                content=[types.TextContent(type="text", text=f"Error: {exc}")],
            )

    async def run(self):
        """Run the MCP server over stdio."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )
