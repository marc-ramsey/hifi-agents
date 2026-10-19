"""Tests for Settings / config defaults and env-var overrides."""



from search_agent.config import Settings


def test_defaults():
    s = Settings()
    assert s.searxng.base_url == "http://maceo.local:8888"
    assert s.searxng.max_results == 10
    assert s.embedding.model == "nomic-embed-text-v2"
    assert s.embedding.dimensions == 768
    assert s.retention.similarity_threshold == 0.75
    assert s.retention.max_chunks == 10000
    assert s.retention.top_k == 5
    assert s.db_path == "search_memory.db"


def test_db_path_from_env(monkeypatch):
    monkeypatch.setenv("SEARCH_AGENT_DB", "/tmp/custom_search.db")
    assert Settings().db_path == "/tmp/custom_search.db"


def test_no_env_var_uses_default(monkeypatch):
    monkeypatch.delenv("SEARCH_AGENT_DB", raising=False)
    assert Settings().db_path == "search_memory.db"


def test_partial_override():
    s = Settings(embedding={"model": "my-model", "dimensions": 512})
    assert s.embedding.model == "my-model"
    assert s.embedding.dimensions == 512
    # untouched sub-configs keep their defaults
    assert s.searxng.base_url == "http://maceo.local:8888"
