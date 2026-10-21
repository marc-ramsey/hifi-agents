"""Shared fixtures and fakes for the test suite.

The fakes let us exercise the real retrieval/ingestion/MCP code paths without
any network access (no live SearXNG or embedding endpoint needed).
"""

import numpy as np
import pytest

from hifi_search.config import Settings
from hifi_search.ingestion import IngestionEngine
from hifi_search.memory_store import MemoryStore
from hifi_search.retrieval import RetrievalEngine

# A tiny deterministic topic-embedding scheme: each known topic lights up one
# dimension. Similar texts share dimensions; unrelated texts are orthogonal.
TOPICS = {
    "python": 0,
    "bread": 1,
    "quantum": 2,
}
DIM = 8


class FakeIngestionEngine:
    """Ingestion engine that embeds deterministically and serves canned results.

    ``fetch`` returns ``live_results``; ``ingest`` actually chunks and stores into
    the real MemoryStore so retrieval re-queries see the new data.
    """

    def __init__(self, store, settings, live_results=None):
        self.store = store
        self.settings = settings
        self.live_results = live_results or []
        self.fetch_calls = []
        self.ingest_calls = []

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(DIM, dtype=np.float32)
        low = text.lower()
        for topic, idx in TOPICS.items():
            if topic in low:
                vec[idx] += 1.0
        return vec

    async def embed(self, texts):
        return np.array([self._embed_one(t) for t in texts], dtype=np.float32)

    async def fetch(self, query, categories=None, language=None):
        self.fetch_calls.append(query)
        return self.live_results

    async def ingest(self, query, results):
        self.ingest_calls.append((query, results))
        for res in results:
            url = res.get("url", "")
            content = res.get("content", "") or res.get("snippet", "")
            if not url or not content:
                continue
            chunks = IngestionEngine.chunk(
                content, res.get("title", ""), url
            )
            if not chunks:
                continue
            chunk_texts = [c["text"] for c in chunks]
            vectors = await self.embed(chunk_texts)
            for i, c in enumerate(chunks):
                self.store.upsert(
                    text=c["text"],
                    vector=vectors[i],
                    url=url,
                    title=res.get("title", ""),
                    metadata={"query": query},
                )
        return len(self.ingest_calls)

    async def close(self):
        pass


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def store(tmp_path):
    mstore = MemoryStore(str(tmp_path / "test.db"), dimensions=DIM)
    yield mstore
    mstore.close()


@pytest.fixture
def fake_ingestion(store, settings):
    return FakeIngestionEngine(store, settings)


@pytest.fixture
def retrieval(store, fake_ingestion, settings):
    return RetrievalEngine(store, fake_ingestion, settings)
