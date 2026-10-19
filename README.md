# search-agent

A persistent-memory agent for LLM tool-calling. It remembers web searches so
follow-up queries can recall previously stored context instead of re-searching
the live web every time.

You give it a query. It looks in its own memory store first; if the memory isn't
confident enough, it falls back to a live SearXNG search, ingests the fresh
results, and returns the best matches. Everything it ingests is stored locally
in SQLite for next time.

## How it works

```
query ──▶ embed ──▶ vector search in memory
                    │
                    ├─ empty / low confidence ──▶ live SearXNG search
                    │                                 │
                    │                                 ▼
                    ◀──────────── ingest: chunk → embed → store ──┘
```

1. **Fetch** — `IngestionEngine.fetch` queries a SearXNG instance for JSON results.
2. **Chunk** — each result's content is split into overlapping word chunks (200 words, 30-word overlap).
3. **Embed** — chunks and the query are embedded in batches via an OpenAI-compatible
   `/v1/embeddings` endpoint (default model: `nomic-embed-text-v2`, 768-d).
4. **Store** — `MemoryStore` persists each chunk in SQLite: text, URL, title, metadata,
   and the vector as a packed `float32` blob.
5. **Retrieval** — cosine similarity is computed in numpy; a `chunks_fts` FTS5 table
   provides keyword fallback. Results are ranked by score.

### Retrieval modes

- **`preference`** (default): search memory first. If memory is empty or the best
  match scores below `similarity_threshold` (0.75), trigger a live SearXNG search,
  ingest it, and re-query.
- **`parallel`**: memory only — no live search. Faster, fully offline.

## Install

```bash
pip install -e .
```

Requires Python 3.10+. Runtime dependencies: `aiohttp`, `numpy`, `pydantic`, `mcp`.

## Usage

### CLI (stdio)

Runs a single query and prints newline-delimited JSON results to stdout:

```bash
search-agent "<your query>"                 # preference mode (memory-first)
search-agent "<your query>" --mode parallel # memory-only, no live search
search-agent "<your query>" --db ./custom.db
```

### MCP server

Exposes a single `search_memory` tool over stdio for use with MCP clients:

```bash
search-agent --transport mcp
```

The `search_memory` tool accepts:

| Argument | Type   | Required | Default      | Description |
|----------|--------|----------|--------------|-------------|
| `query`  | string | yes      | —            | The search query |
| `mode`   | string | no       | `preference` | `preference` (memory-first) or `parallel` (memory-only) |
| `top_k`  | integer| no       | `5`          | Number of results to return |

## Configuration

All settings live in `src/search_agent/config.py` as Pydantic models and can be
overridden via environment variables.

| Setting | Default | Description |
|---------|---------|-------------|
| `SEARCH_AGENT_DB` | `search_memory.db` | SQLite database path |
| `SearXNGConfig.base_url` | `http://maceo.local:8888` | SearXNG instance URL |
| `SearXNGConfig.max_results` | `10` | Max results per SearXNG query |
| `EmbeddingConfig.base_url` | `http://localhost:8080/v1` | Embedding endpoint URL |
| `EmbeddingConfig.model` | `nomic-embed-text-v2` | Embedding model name |
| `EmbeddingConfig.dimensions` | `768` | Vector dimensions (must match the model) |
| `RetentionConfig.similarity_threshold` | `0.75` | Min confidence before live search |
| `RetentionConfig.max_chunks` | `10000` | Prune oldest when exceeded |
| `RetentionConfig.top_k` | `5` | Default results per query |

> The embedding dimensions are coupled to the model — changing `nomic-embed-text-v2`
> requires updating `dimensions` or the stored vector blobs become incompatible.
> Point `base_url` values at your real SearXNG / embedding endpoints.

## Architecture

| Module | Owns |
|--------|------|
| `config.py` | All settings (`Settings`, `SearXNGConfig`, `EmbeddingConfig`, `RetentionConfig`) + env-var DB path |
| `memory_store.py` | SQLite persistence: `chunks` table, FTS5 `chunks_fts`, vector blob (de)serialize, upsert/query/prune |
| `ingestion.py` | SearXNG fetch, chunking, batch embedding, persistence |
| `retrieval.py` | Query embed + ranked recall (preference/parallel modes) |
| `adapters/mcp.py` | MCP stdio transport (tool list + tool call) |
| `__main__.py` | CLI wiring, arg parsing, transport dispatch |

## Development

```bash
ruff check .
pytest
```
