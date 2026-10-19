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


def test_db_path_from_env(monkeypatch):
    monkeypatch.setenv("SEARCH_AGENT_DB", "/tmp/custom_search.db")
    assert Settings().db_path == "/tmp/custom_search.db"


def test_db_path_default_uses_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("SEARCH_AGENT_DB", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert Settings().db_path == str(tmp_path / "search-agent" / "memory.db")


def test_db_path_home_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("SEARCH_AGENT_DB", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert Settings().db_path == str(tmp_path / ".local" / "share" / "search-agent" / "memory.db")


def test_partial_override():
    s = Settings(embedding={"model": "my-model", "dimensions": 512})
    assert s.embedding.model == "my-model"
    assert s.embedding.dimensions == 512
    # untouched sub-configs keep their defaults
    assert s.searxng.base_url == "http://maceo.local:8888"
