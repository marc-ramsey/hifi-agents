# SearXNG Persistent Memory Architecture

## 1. Objective
Construct a session-persistent knowledge store using SearXNG as the primary data ingestion engine. The system should operate in two modes:
- **Parallel Mode**: Background memory construction runs concurrently with live agent queries.
- **Preference Mode**: Memory retrieval is prioritized; live SearXNG queries are only triggered when memory confidence or coverage falls below a threshold.

## 2. SearXNG Integration Method
- **Endpoint**: `GET /search?q={query}&format=json&categories={cat}&language={lang}`
- **Authentication**: Use self-hosted instance with API key or local network access.
- **Rate Limiting**: Implement exponential backoff and respect `search.json` response headers.
- **Result Extraction**: Parse `results[]` array for `title`, `content`, `url`, `engine`, `score`, `publishedDate`.

## 3. Persistent Memory Architecture
| Layer | Technology | Purpose |
|-------|------------|---------|
| Raw Ingestion | SQLite / PostgreSQL | Store raw SearXNG responses, timestamps, and source metadata |
| Semantic Index | Chroma / Qdrant / FAISS | Vector embeddings for fast similarity retrieval |
| Graph/Relational | Neo4j or SQLite JSON1 | Entity relationships, URL deduplication, confidence scoring |
| Cache Layer | Redis | Hot memory entries, session context, TTL-managed stale data |

## 4. Data Processing & Indexing Pipeline
1. **Chunking**: Split `content` into 256–512 token segments with overlap. Preserve `title` and `url` as metadata.
2. **Embedding**: Generate vectors using `sentence-transformers/all-MiniLM-L6-v2` or equivalent lightweight model.
3. **Metadata Enrichment**: Attach `source_engine`, `query_intent`, `confidence_score`, `ingestion_timestamp`, `decay_factor`.
4. **Deduplication**: 
   - Exact: SHA256 of `url` + normalized `title`
   - Semantic: Cosine similarity > 0.92 against existing chunks → merge or discard
5. **Storage**: Upsert into vector DB + relational store. Maintain versioned entries for rollback.

## 5. Execution Models

### Parallel Mode
- **Scheduler**: APScheduler or Celery beat running every 15–60 mins
- **Trigger**: Context gap detection, topic drift, or manual queue
- **Worker**: Async pool (asyncio/uvloop) queries SearXNG → processes → writes to memory
- **Agent Interaction**: Agent reads memory concurrently; live search runs only on explicit user request or timeout

### Preference Mode
- **Retrieval-First**: On query, embed prompt → query vector DB → top-k results
- **Confidence Gate**: If max similarity < threshold (e.g., 0.75) or coverage < N entities → fallback to live SearXNG
- **Memory Update**: Live results are processed and injected into persistent store for future preference
- **Latency Optimization**: Cache frequent queries; use hybrid search (BM25 + vector) for precision

## 6. Memory Retrieval & Fallback Strategy
