from mcp.server import Server
from mcp.types import Tool, TextContent
from typing import Dict, Any
import asyncio

class MCPAdapter:
    def __init__(self, retrieval_engine):
        self.retrieval_engine = retrieval_engine
        self.server = Server("search-agent-mcp")
        self._register_tools()

    def _register_tools(self):
        @self.server.tool("search_memory", "Search the persistent memory store")
        async def search_memory(query: str, mode: str = "preference") -> list[TextContent]:
            results = await self.retrieval_engine.retrieve(query, mode)
            return [TextContent(type="text", text=str(results))]

    async def run(self):
        await self.server.run()
