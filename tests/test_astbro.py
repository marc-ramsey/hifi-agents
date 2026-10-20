"""Tests for the AstBro MCP client using an injected fake session."""

import pytest

from hifi_code.astbro import AstBro


class FakeContent:
    def __init__(self, text):
        self.text = text


class FakeResult:
    def __init__(self, text, is_error=False):
        self.content = [FakeContent(text)]
        self.is_error = is_error


class FakeSession:
    def __init__(self, responses=None, tools=None):
        self.responses = responses or {}
        self.tools = tools or []
        self.initialized = False
        self.calls = []

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        class _T:
            def __init__(self, name, description):
                self.name = name
                self.description = description
        return _Tools([_T(t["name"], t["description"]) for t in self.tools])

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return FakeResult(self.responses.get(name, '{"ok": true}'))


class _Tools:
    def __init__(self, tools):
        self.tools = tools


@pytest.fixture
def session():
    return FakeSession(
        responses={"map": '{"files": [1, 2]}', "show": '{"symbols": []}'},
        tools=[{"name": "map", "description": "d"}],
    )


async def test_call_parses_json(session):
    b = AstBro(session=session)
    data = await b.map()
    assert data == {"files": [1, 2]}
    assert session.initialized is True


async def test_call_returns_raw_on_bad_json(session):
    session.responses["show"] = "not json"
    b = AstBro(session=session)
    data = await b.show(path="x.py")
    assert data == {"raw": "not json"}


async def test_call_raises_on_error_flag(session):
    session.responses["map"] = '{"ok": false}'
    session.calls = []
    b = AstBro(session=session)
    with pytest.raises(RuntimeError):
        await b.map()


async def test_list_tools(session):
    b = AstBro(session=session)
    tools = await b.list_tools()
    assert tools[0]["name"] == "map"


async def test_generic_call_dispatch(session):
    b = AstBro(session=session)
    data = await b.call("search", {"query": "q"})
    assert data == {"ok": True}
    assert ("search", {"query": "q"}) in session.calls
