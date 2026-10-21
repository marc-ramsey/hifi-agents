"""StreamableHTTP transport for hifi_search.

Wraps hifi_search's MCPAdapter as a StreamableHTTP endpoint so it can be reached by
URL (e.g. from OpenWebUI) instead of stdio. Reuses the existing MCPAdapter (which
exposes the ``search_memory`` tool) and mounts it with the mcp low-level Server's
``streamable_http_app``.
"""

from hifi_search.adapters.mcp import MCPAdapter
from hifi_search.retrieval import RetrievalEngine


def create_app(
    retrieval_engine: RetrievalEngine,
    db_path: str = "search_memory.db",
    *,
    path: str = "/mcp",
):
    """Build the StreamableHTTP ASGI app exposing hifi_search's tools."""
    adapter = MCPAdapter(retrieval_engine, db_path)
    return adapter.server.streamable_http_app(streamable_http_path=path)
