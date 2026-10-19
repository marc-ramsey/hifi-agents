"""Tests for the MCPAdapter tool definition and call handlers.

The handlers (_list_tools / _call_tool) are pure w.r.t. their arguments, so we
call them directly with a fake retrieval engine — no transport or network
needed.
"""

import pytest
from mcp import types

from search_agent.adapters.mcp import MCPAdapter


class FakeRetrievalEngine:
    def __init__(self, results=None, raise_exc=False):
        self._results = results or []
        self.raise_exc = raise_exc
        self.calls = []

    async def retrieve(self, query, mode="preference", top_k=None):
        self.calls.append((query, mode, top_k))
        if self.raise_exc:
            raise RuntimeError("boom")
        return self._results


def make_adapter(results=None, raise_exc=False):
    engine = FakeRetrievalEngine(results, raise_exc)
    adapter = MCPAdapter(engine, db_path=":memory:")
    return adapter, engine


@pytest.fixture
def adapter():
    a, _ = make_adapter()
    return a


# --- tool definition -------------------------------------------------------

def test_tool_definition(adapter):
    assert adapter.tool_def.name == "search_memory"
    assert adapter.tool_def.input_schema["required"] == ["query"]
    modes = adapter.tool_def.input_schema["properties"]["mode"]["enum"]
    assert modes == ["preference", "parallel"]


async def test_list_tools(adapter):
    res = await adapter._list_tools(None, None)
    assert [t.name for t in res.tools] == ["search_memory"]


# --- call tool -------------------------------------------------------------

async def test_call_tool_missing_query(adapter):
    params = types.CallToolRequestParams(name="search_memory", arguments={})
    res = await adapter._call_tool(None, params)
    assert res.is_error is True
    assert "required" in res.content[0].text


async def test_call_tool_success(adapter):
    results = [
        {"score": 0.9, "title": "T", "url": "https://u.com", "text": "body"},
    ]
    a, engine = make_adapter(results)
    params = types.CallToolRequestParams(
        name="search_memory", arguments={"query": "python", "mode": "preference"}
    )
    res = await a._call_tool(None, params)
    assert res.is_error is False
    assert "Result 1" in res.content[0].text
    assert engine.calls == [("python", "preference", 5)]


async def test_call_tool_no_results(adapter):
    params = types.CallToolRequestParams(name="search_memory", arguments={"query": "x"})
    res = await adapter._call_tool(None, params)
    assert res.is_error is False
    assert "No results found" in res.content[0].text


async def test_call_tool_unknown_tool(adapter):
    params = types.CallToolRequestParams(name="other", arguments={})
    res = await adapter._call_tool(None, params)
    assert res.is_error is True
    assert "Unknown tool" in res.content[0].text


async def test_call_tool_error_propagates_as_error_result(adapter):
    a, _ = make_adapter(raise_exc=True)
    params = types.CallToolRequestParams(name="search_memory", arguments={"query": "x"})
    res = await a._call_tool(None, params)
    assert res.is_error is True
    assert "Error:" in res.content[0].text
