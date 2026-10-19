"""Tests for IngestionEngine: chunking and embed response-format parsing.

Network calls (fetch/embed) are exercised against a fake aiohttp session so no
live SearXNG / embedding endpoint is required.
"""


import numpy as np
import pytest

from search_agent.ingestion import IngestionEngine

DIM = 8  # matches the topic-embedding scheme in conftest


# --- chunk (pure static method) -------------------------------------------

def test_chunk_splits_into_overlapping_windows():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = IngestionEngine.chunk(text, "T", "https://u.com")
    assert len(chunks) == 2  # 250 words, step=170 -> windows [0:200], [170:370]
    assert chunks[0]["text"].startswith("w0")
    assert chunks[1]["text"].startswith("w170")
    # overlap: last word of window 0 (w199) reappears in window 1
    assert "w199" in chunks[1]["text"]
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[0]["metadata"]["total_words"] == 250


def test_chunk_short_text_returns_empty():
    assert IngestionEngine.chunk("too short", "T", "https://u.com") == []


def test_chunk_empty_text_returns_empty():
    assert IngestionEngine.chunk("", "T", "https://u.com") == []


def test_chunk_single_window_when_under_max():
    text = " ".join(f"w{i}" for i in range(50))
    chunks = IngestionEngine.chunk(text, "T", "https://u.com")
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


# --- embed (response-format parsing via fake session) ----------------------

class FakeResponse:
    def __init__(self, data):
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._data

    def raise_for_status(self):
        pass


class FakeSession:
    closed = False

    def __init__(self, data):
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None, timeout=None):
        # aiohttp's session.post() returns an async context manager, not a
        # coroutine, so this must be a plain method returning FakeResponse.
        return FakeResponse(self._data)

    def get(self, url, params=None, timeout=None):
        return FakeResponse(self._data)


@pytest.fixture
def engine(settings):
    return IngestionEngine(settings, memory_store=None)


@pytest.mark.parametrize("payload", [
    {"data": [{"embedding": [1.0, 2.0, 3.0]}, {"embedding": [4.0, 5.0, 6.0]}]},
    {"embedding": [1.0, 2.0, 3.0], "index": 0},
    [{"embedding": [1.0, 2.0, 3.0]}, {"embedding": [4.0, 5.0, 6.0]}],
])
async def test_embed_parses_supported_formats(engine, monkeypatch, payload):
    monkeypatch.setattr(engine, "_session", FakeSession(payload))
    vectors = await engine.embed(["a", "b"])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[1] == 3


async def test_embed_returns_float32(engine, monkeypatch):
    monkeypatch.setattr(engine, "_session", FakeSession({"data": [{"embedding": [1.0, 0.0]}]}))
    vectors = await engine.embed(["a"])
    assert vectors.dtype == np.float32
    assert np.allclose(vectors[0], [1.0, 0.0])


async def test_embed_unsupported_format_raises(engine, monkeypatch):
    monkeypatch.setattr(engine, "_session", FakeSession({"unexpected": 1}))
    with pytest.raises(ValueError):
        await engine.embed(["a"])


async def test_fetch_parses_searxng_json(settings, monkeypatch):
    eng = IngestionEngine(settings, memory_store=None)
    monkeypatch.setattr(eng, "_session", FakeSession(
        {"results": [{"url": f"https://i{i}.com"} for i in range(3)]}
    ))
    results = await eng.fetch("q")
    assert len(results) == 3
    assert results[0]["url"] == "https://i0.com"


async def test_ingest_chunks_embeds_and_stores(store, settings, monkeypatch):
    eng = IngestionEngine(settings, store)

    async def fake_embed(texts):
        return np.array(
            [np.ones(DIM, dtype=np.float32) * (i + 1) for i, _ in enumerate(texts)],
            dtype=np.float32,
        )

    monkeypatch.setattr(eng, "embed", fake_embed)

    results = [{"url": "https://a.com", "title": "A", "content": " ".join(f"w{i}" for i in range(220))}]
    count = await eng.ingest("q", results)

    # 220 words -> 2 chunks processed
    assert count == 2
    # url is the UNIQUE key, so both chunks collapse to a single stored row
    assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    text = store.conn.execute("SELECT text FROM chunks").fetchone()[0]
    assert text.endswith("w219")  # the last chunk survives


async def test_ingest_skips_duplicate_and_short_urls(store, settings, monkeypatch):
    eng = IngestionEngine(settings, store)

    async def fake_embed(texts):
        return np.ones((len(texts), DIM), dtype=np.float32)

    monkeypatch.setattr(eng, "embed", fake_embed)
    results = [
        {"url": "https://dup.com", "title": "D", "content": " ".join(f"w{i}" for i in range(220))},
        {"url": "https://dup.com", "title": "D", "content": "second time same url"},
        {"url": "https://short.com", "title": "S", "content": "too short"},
    ]
    count = await eng.ingest("q", results)
    assert count == 2  # dup url deduped; short url skipped
    assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1

