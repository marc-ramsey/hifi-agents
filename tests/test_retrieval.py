"""Tests for RetrievalEngine control flow (preference vs parallel mode).

Uses the FakeIngestionEngine: no network. Topic-embedding makes queries about
"python" match stored python chunks (high confidence) and queries about
"bread" orthogonal (low confidence).
"""




def _store_python_chunk(store):
    # embed a python-ish text the same way the fake ingestion does
    from conftest import FakeIngestionEngine as Fake
    fake = Fake(store, None)
    vec = fake._embed_one("python programming language tutorial")
    store.upsert("python tutorial content", vec, url="https://py.com", title="Python")


async def test_parallel_mode_never_searches_live(retrieval, store):
    _store_python_chunk(store)
    results = await retrieval.retrieve("python", mode="parallel")
    assert len(results) == 1
    assert results[0]["url"] == "https://py.com"
    assert retrieval.ingestion.fetch_calls == []  # no live search


async def test_preference_high_confidence_no_live_search(retrieval, store):
    _store_python_chunk(store)
    results = await retrieval.retrieve("python", mode="preference")
    assert len(results) == 1
    assert retrieval.ingestion.fetch_calls == []  # memory was confident


async def test_preference_empty_memory_triggers_live_search(retrieval, store, settings):
    from conftest import FakeIngestionEngine as Fake
    live = [{"url": "https://live.com", "title": "Live", "content": " ".join(
        f"python w{i}" for i in range(220))}]
    # swap in a fake whose fetch returns live results and ingest stores them
    ingestion = Fake(store, settings, live_results=live)
    from search_agent.retrieval import RetrievalEngine
    engine = RetrievalEngine(store, ingestion, settings)

    results = await engine.retrieve("python", mode="preference")
    assert ingestion.fetch_calls == ["python"]
    assert len(ingestion.ingest_calls) == 1
    assert len(results) == 1
    assert results[0]["url"] == "https://live.com"


async def test_preference_low_confidence_triggers_live_search(retrieval, store, settings):
    # store a chunk that partially overlaps "bread": lights dims 0 (python) and 1
    # (bread) -> cosine with a pure "bread" query is 1/sqrt(2) ~= 0.707, above the
    # score>0 filter but below the 0.75 similarity threshold.
    from conftest import FakeIngestionEngine as Fake
    fake = Fake(store, None)
    vec = fake._embed_one("python and bread cooking together")
    store.upsert("mixed content", vec, url="https://mixed.com", title="Mixed")
    live = [{"url": "https://bread.com", "title": "Bread", "content": " ".join(
        f"bread w{i}" for i in range(220))}]
    # swap in a fake whose fetch returns live results and ingest stores them
    ingestion = Fake(store, settings, live_results=live)
    from search_agent.retrieval import RetrievalEngine
    engine = RetrievalEngine(store, ingestion, settings)

    results = await engine.retrieve("bread", mode="preference")
    assert ingestion.fetch_calls == ["bread"]  # low confidence -> live search
    assert len(results) == 2  # mixed (0.707) + newly ingested bread (1.0)
    assert results[0]["url"] == "https://bread.com"  # highest score first


async def test_preference_live_search_no_results_returns_empty(retrieval, store, settings):
    # low confidence triggers a live search, but it returns nothing -> empty result
    from conftest import FakeIngestionEngine as Fake
    fake = Fake(store, None)
    vec = fake._embed_one("python and bread cooking together")
    store.upsert("mixed content", vec, url="https://mixed.com", title="Mixed")
    ingestion = Fake(store, settings, live_results=[])  # live search returns nothing
    from search_agent.retrieval import RetrievalEngine
    engine = RetrievalEngine(store, ingestion, settings)

    results = await engine.retrieve("bread", mode="preference")
    assert ingestion.fetch_calls == ["bread"]
    assert ingestion.ingest_calls == []  # nothing to ingest
    assert len(results) == 1  # original low-confidence mixed chunk retained
    assert results[0]["url"] == "https://mixed.com"


async def test_top_k_is_respected(retrieval, store):
    _store_python_chunk(store)
    results = await retrieval.retrieve("python", mode="parallel", top_k=1)
    assert len(results) <= 1
