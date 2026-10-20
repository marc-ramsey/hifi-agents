"""CLI entry point for hifi_code: arg parsing + transport dispatch."""

import argparse
import asyncio
import logging
import sys

from .config import Settings
from .transports.mcp import MCPAdapter
from .transports.stdio import run

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="ast-bro-driven coding loop")
    parser.add_argument("goal", nargs="?", help="Coding goal (stdio mode)")
    parser.add_argument("--transport", choices=["stdio", "mcp"],
                        default="stdio", help="I/O transport")
    parser.add_argument("--root", default=".", help="Codebase root to seed from")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max loop steps")
    parser.add_argument("--db", default=None, help="Path to state database")
    args = parser.parse_args()

    settings = Settings(db_path=args.db) if args.db else Settings()
    if args.max_steps is not None:
        settings.max_steps = args.max_steps

    if args.transport == "mcp":
        adapter = MCPAdapter(settings)
        logger.info("Starting hifi_code MCP server (stdio)...")
        asyncio.run(adapter.run())
    else:
        if not args.goal:
            parser.print_help()
            sys.exit(1)
        asyncio.run(run(settings, args.goal, args.root))


if __name__ == "__main__":
    main()
