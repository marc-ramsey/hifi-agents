"""Tests for MemoryStore: vector round-trip, cosine ranking, FTS5, pruning."""

import numpy as np
import pytest

from hifi_search.memory_store import MemoryStore

DIM = 8  # matches the topic-embedding scheme in conftest


def test_upsert_returns_id_and_roundtrips(store):
    vec = np.array([1.0, 0.0] + [0.0] * (DIM - 2), dtype=np.float32)
    cid = store.upsert("hello world", vec, url="https://a.com", title="A")
    assert isinstance(cid, int)

    results = store.query(vec)
    assert len(results) == 1
    r = results[0]
    assert r["url"] == "https://a.com"
    assert r["title"] == "A"
    assert r["score"] == pytest.approx(1.0, abs=1e-4)
    assert r["text"] == "hello world"


def test_query_ranks_by_cosine_similarity(store):
    q = np.array([1.0, 0.0] + [0.0] * (DIM - 2), dtype=np.float32)
    # stored: one identical (score 1.0), one orthogonal (excluded, score 0)
    store.upsert("python", np.array([1.0] + [0.0] * (DIM - 1), dtype=np.float32), url="https://py")
    store.upsert("bread", np.array([0.0, 1.0] + [0.0] * (DIM - 2), dtype=np.float32), url="https://bd")

    results = store.query(q, top_k=5)
    assert [r["url"] for r in results] == ["https://py"]
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)


def test_upsert_on_unique_url_updates(store):
    v1 = np.array([1.0] + [0.0] * (DIM - 1), dtype=np.float32)
    v2 = np.array([0.0, 1.0] + [0.0] * (DIM - 2), dtype=np.float32)
    first = store.upsert("old", v1, url="https://same.com")
    second = store.upsert("new", v2, url="https://same.com")

    # same row updated, not a new one
    assert first == second
    assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1

    results = store.query(v2)
    assert len(results) == 1
    assert results[0]["text"] == "new"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)


def test_query_empty_store_returns_empty(store):
    assert store.query(np.zeros(DIM, dtype=np.float32)) == []


def test_metadata_persisted_and_parsed(store):
    vec = np.array([1.0] + [0.0] * (DIM - 1), dtype=np.float32)
    store.upsert("meta text", vec, url="https://m.com", metadata={"query": "q1", "chunk_index": 3})
    results = store.query(vec)
    assert results[0]["metadata"] == {"query": "q1", "chunk_index": 3}


def test_text_search_fts5(store):
    vec = np.zeros(DIM, dtype=np.float32)
    store.upsert("python is a programming language", vec, url="https://py.com", title="Python")
    store.upsert("baking sourdough bread at home", vec, url="https://bread.com", title="Bread")

    results = store.text_search("programming")
    assert len(results) == 1
    assert results[0]["url"] == "https://py.com"
    assert results[0]["score"] > 0


def test_prune_oldest_keeps_most_recent(store):
    for i in range(12):
        store.upsert(f"chunk {i}", np.zeros(DIM, dtype=np.float32), url=f"https://c{i}.com")
    assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 12

    store.prune_oldest(keep=5)
    assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 5

    # 12 inserts, delete 7 oldest -> keep ids 8..12
    ids = [r[0] for r in store.conn.execute("SELECT id FROM chunks ORDER BY id").fetchall()]
    assert ids == [8, 9, 10, 11, 12]


def test_prune_noop_when_under_limit(store):
    for i in range(3):
        store.upsert(f"c{i}", np.zeros(DIM, dtype=np.float32), url=f"https://x{i}.com")
    store.prune_oldest(keep=10000)
    assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3


def test_serialize_deserialize_roundtrip():
    v = np.random.randn(8).astype(np.float32)
    blob = MemoryStore._serialize(v)
    out = MemoryStore._deserialize(blob, 8)
    assert out.shape == (1, 8)
    assert np.allclose(out[0], v, atol=1e-6)
