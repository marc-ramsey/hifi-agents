# Project Instructions

## Overview
`hifi-search` is a SearXNG-powered persistent memory agent for LLM tool-calling. It fetches web results via SearXNG, chunks and embeds them into SQLite (with vector blobs + an FTS5 keyword index), and answers follow-up queries by recalling stored context instead of re-searching live. It runs as a CLI (stdio) or as an MCP server.

## Build & Test
```
pip install -e .
pytest
ruff check .
```
Run the CLI:
```
hifi-search "<query>"                 # stdio, single query
hifi-search "<query>" --mode parallel # parallel retrieval mode
hifi-search --transport mcp           # MCP stdio server
```

## Code Style
- Python 3.10+, src-layout package (`src/hifi_search/`), setuptools.
- Pydantic v2 `BaseModel` for all config.
- Async where I/O hits the network (`asyncio.run` in `__main__`); sync SQLite in `MemoryStore`.
- Public query results are emitted as newline-delimited JSON (`json.dumps(..., ensure_ascii=False)`).

## Architecture
Entry is `__main__.main`, which builds `Settings` and wires `MemoryStore` (SQLite persistence) with `IngestionEngine` (fetch → chunk → embed → persist) and `RetrievalEngine` (recall). `RetrievalEngine.retrieve` is the single recall path shared by both transports: stdio prints JSON, the `adapters/mcp.py` MCPAdapter exposes it as an MCP tool. See the Module Map in `.zap/understanding.md`.

## Important Files
- `src/hifi_search/__main__.py` — CLI entry, arg parsing, transport dispatch.
- `src/hifi_search/config.py` — all settings/env vars (SearXNG, embedding, retention, DB path).
- `src/hifi_search/memory_store.py` — SQLite schema, vector blob (de)serialize, upsert/query/prune.
- `src/hifi_search/ingestion.py` — SearXNG fetch, chunking, embedding, persistence.
- `src/hifi_search/retrieval.py` — query embedding and ranked recall.
- `src/hifi_search/adapters/mcp.py` — MCP stdio transport (tool list + tool call).
- `src/hifi_search/adapters/http.py` — StreamableHTTP transport (tool list + tool call).

## Do Not Touch
- Do not change `EmbeddingConfig.dimensions` (768) without also changing the embedding model — vector blob shape depends on it.
- Don't assume `adapters/` has more than `mcp.py` — `stdin_stdout.py` was removed.
- Don't hardcode secrets (SearXNG/embedding URLs) — keep them in `config.py` defaults or env vars.
