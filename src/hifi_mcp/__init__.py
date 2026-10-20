"""HTTP transport bridge for search-agent.

Wraps search-agent's stdio MCP server as a StreamableHTTP endpoint so it can be
reached by URL (e.g. from OpenWebUI) instead of stdio. Reuses the existing
MCPAdapter (which exposes the ``search_memory`` tool) and mounts it with the
mcp low-level Server's ``streamable_http_app``.
"""

import argparse
import logging

from search_agent.adapters.mcp import MCPAdapter
from search_agent.config import Settings
from search_agent.ingestion import IngestionEngine
from search_agent.memory_store import MemoryStore
from search_agent.retrieval import RetrievalEngine

logger = logging.getLogger(__name__)


def create_app(
    retrieval_engine: RetrievalEngine,
    db_path: str = "search_memory.db",
    *,
    path: str = "/mcp",
):
    """Build the StreamableHTTP ASGI app exposing search-agent's tools."""
    adapter = MCPAdapter(retrieval_engine, db_path)
    return adapter.server.streamable_http_app(streamable_http_path=path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="search-agent StreamableHTTP bridge"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--path", default="/mcp", help="Mount path")
    parser.add_argument("--db", default=None, help="Path to SQLite database")
    args = parser.parse_args()

    settings = Settings(db_path=args.db) if args.db else Settings()
    store = MemoryStore(settings.db_path, settings.embedding.dimensions)

    ingestion = IngestionEngine(settings, store)
    retrieval = RetrievalEngine(store, ingestion, settings)

    app = create_app(retrieval, settings.db_path, path=args.path)

    import uvicorn

    logger.info(
        "Starting search-agent HTTP bridge at http://%s:%d%s",
        args.host,
        args.port,
        args.path,
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        store.close()
        import asyncio

        asyncio.run(ingestion.close())


if __name__ == "__main__":
    main()
