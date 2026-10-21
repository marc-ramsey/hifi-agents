import argparse
import asyncio
import json
import logging
import sys

logger = logging.getLogger(__name__)

from .adapters.http import create_app
from .adapters.mcp import MCPAdapter
from .config import Settings
from .ingestion import IngestionEngine
from .memory_store import MemoryStore
from .retrieval import RetrievalEngine


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="hifi_search — SearXNG persistent memory agent")
    parser.add_argument("query", nargs="?", help="Search query (stdio mode)")
    parser.add_argument("--mode", choices=["preference", "parallel"],
                        default="preference", help="Retrieval mode")
    parser.add_argument("--transport", choices=["stdio", "mcp", "http"],
                        default="stdio", help="I/O transport")
    parser.add_argument("--db", default=None, help="Path to SQLite database")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host (http transport)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP bind port (http transport)")
    parser.add_argument("--path", default="/mcp", help="MCP mount path (http transport)")
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
        elif args.transport == "http":
            app = create_app(retrieval, settings.db_path, path=args.path)
            logger.info(
                "Starting hifi_search HTTP bridge at http://%s:%d%s",
                args.host,
                args.port,
                args.path,
            )
            import uvicorn

            uvicorn.run(app, host=args.host, port=args.port)
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
