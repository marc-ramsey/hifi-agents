"""MCP server adapter for hifi_search."""

import logging

from mcp import types
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server

logger = logging.getLogger(__name__)


class MCPAdapter:
    """MCP server that exposes search_memory as a tool."""

    def __init__(self, retrieval_engine, db_path: str = "search_memory.db"):
        self.retrieval_engine = retrieval_engine
        self.db_path = db_path

        # Build the tool definition
        self.tool_def = types.Tool(
            name="search_memory",
            title="Search Persistent Memory",
            description=(
                "Search the persistent search memory store. "
                "Returns relevant chunks from previously searched web results. "
                "Use 'preference' mode for memory-first with live SearXNG fallback, "
                "or 'parallel' for memory-only (faster, no live search)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["preference", "parallel"],
                        "description": (
                            "preference = memory-first with live SearXNG fallback "
                            "(default). parallel = memory-only, no live search."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                    },
                },
                "required": ["query"],
            },
        )

        # Build the server with tool handlers
        self.server = Server(
            "hifi-search",
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

            if name == "search_memory":
                query = arguments.get("query", "")
                if not query:
                    return types.CallToolResult(
                        is_error=True,
                        content=[types.TextContent(type="text", text="Error: 'query' is required")],
                    )

                mode = arguments.get("mode", "preference")
                top_k = arguments.get("top_k", 5)

                results = await self.retrieval_engine.retrieve(
                    query, mode=mode, top_k=top_k
                )

                # Format results as text
                if not results:
                    return types.CallToolResult(
                        content=[types.TextContent(type="text", text="No results found.")]
                    )

                lines = []
                for i, r in enumerate(results, 1):
                    lines.append(
                        f"Result {i} (score: {r['score']:.4f}):\n"
                        f"  Title: {r['title']}\n"
                        f"  URL: {r['url']}\n"
                        f"  Text: {r['text'][:500]}"
                    )
                    lines.append("")

                text = "\n".join(lines)
                return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

            else:
                return types.CallToolResult(
                    is_error=True,
                    content=[types.TextContent(type="text", text=f"Unknown tool: {name}")],
                )

        except Exception as e:
            logger.exception("Error in MCP tool call")
            return types.CallToolResult(
                is_error=True,
                content=[types.TextContent(type="text", text=f"Error: {e}")],
            )

    async def run(self):
        """Run the MCP server over stdio."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )
