import asyncio
import argparse
from .config import Settings
from .retrieval import RetrievalEngine
from .ingestion import SearXNGIngestionEngine
from .memory_store import MemoryStore
from .adapters.stdin_stdout import StdinStdoutAdapter
from .adapters.mcp import MCPAdapter

def main():
    parser = argparse.ArgumentParser(description="SearXNG Persistent Memory Agent")
    parser.add_argument("--transport", choices=["stdio", "mcp"], default="stdio", help="I/O transport mode")
    args = parser.parse_args()

    settings = Settings()
    memory_store = MemoryStore(settings.vector, settings.db, settings.cache)
    ingestion_engine = SearXNGIngestionEngine(settings, memory_store, memory_store.graph, memory_store.cache)
    retrieval_engine = RetrievalEngine(memory_store, settings, ingestion_engine)

    if args.transport == "stdio":
        adapter = StdinStdoutAdapter(retrieval_engine)
        asyncio.run(adapter.run())
    elif args.transport == "mcp":
        adapter = MCPAdapter(retrieval_engine)
        asyncio.run(adapter.run())

if __name__ == "__main__":
    main()
