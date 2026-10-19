import asyncio
import argparse
import json
import logging
import sys

logger = logging.getLogger(__name__)

from .config import Settings
from .memory_store import MemoryStore
from .retrieval import RetrievalEngine
from .ingestion import IngestionEngine
from .adapters.mcp import MCPAdapter


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="SearXNG Persistent Memory Agent")
    parser.add_argument("query", nargs="?", help="Search query (stdio mode)")
    parser.add_argument("--mode", choices=["preference", "parallel"],
                        default="preference", help="Retrieval mode")
    parser.add_argument("--transport", choices=["stdio", "mcp"],
                        default="stdio", help="I/O transport")
    parser.add_argument("--db", default=None, help="Path to SQLite database")
    args = parser.parse_args()

    settings = Settings(db_path=args.db) if args.db else Settings()
    store = MemoryStore(settings.db_path, settings.embedding.dimensions)

    try:
        ingestion = IngestionEngine(settings, store)
        retrieval = RetrievalEngine(store, ingestion, settings)

        if args.transport == "mcp":
            adapter = MCPAdapter(retrieval, settings.db_path)
            logger.info("Starting MCP server (stdio)...")
            asyncio.run(adapter.run())
        else:
            # stdio: single query via command line
            if not args.query:
                parser.print_help()
                sys.exit(1)

            results = asyncio.run(
                retrieval.retrieve(args.query, mode=args.mode)
            )

            for r in results:
                print(json.dumps(r, ensure_ascii=False))

            if not results:
                print(json.dumps({"error": "No results found"}))
                sys.exit(1)

    finally:
        store.close()
        asyncio.run(ingestion.close())


if __name__ == "__main__":
    main()
