"""End-to-end integration test for the search-agent HTTP bridge.

Spawns the real ``hifi-mcp`` bridge as a subprocess plus two stdlib stub servers
(an OpenAI-compatible embedding stub and a SearXNG stub) as subprocesses, points
the bridge at the stubs *purely via env vars*, then drives it with a real MCP
StreamableHTTP client. This proves the subprocesses actually talk to each other
over HTTP — the exact surface OpenWebUI will use — without hardcoding the user's
live servers (localhost:8080 / maceo.local:8888).
"""

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

STUB = Path(__file__).parent / "stub_server.py"
BRIDGE_PATH = "/mcp"


def find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_http_ready(url: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except urllib.error.HTTPError:
            return  # server responded (even a 4xx/406) => it's listening
        except urllib.error.URLError as exc:
            last = exc
            time.sleep(0.1)
    raise RuntimeError(f"{url} not ready within {timeout}s (last: {last})")


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _score(text: str) -> float:
    m = re.search(r"score:\s*([0-9.]+)", text)
    return float(m.group(1)) if m else -1.0


@pytest.fixture(scope="session")
def backends():
    """Spawn the embedding + SearXNG stub servers once for the whole session."""
    embed_port, searxng_port = find_free_port(), find_free_port()
    embed_base = f"http://127.0.0.1:{embed_port}/v1"
    searxng_base = f"http://127.0.0.1:{searxng_port}"
    embed = subprocess.Popen(
        [sys.executable, str(STUB), "--kind", "embed", "--port", str(embed_port)],
        env=os.environ,
    )
    searxng = subprocess.Popen(
        [sys.executable, str(STUB), "--kind", "searxng", "--port", str(searxng_port)],
        env=os.environ,
    )
    try:
        wait_http_ready(f"{searxng_base}/__stats")
        wait_http_ready(f"{embed_base}/__stats")
        yield {"embed": embed_base, "searxng": searxng_base}
    finally:
        terminate(embed)
        terminate(searxng)


@pytest.fixture
def bridge(backends, tmp_path):
    """Spawn a fresh bridge with an isolated DB, pointed at the stubs via env."""
    port = find_free_port()
    env = dict(os.environ)
    env["SEARCH_AGENT_EMBEDDING_BASE_URL"] = backends["embed"]
    env["SEARCH_AGENT_SEARXNG_BASE_URL"] = backends["searxng"]
    cmd = [
        sys.executable,
        "-c",
        "import sys; from hifi_mcp import main; main()",
        "--port",
        str(port),
        "--path",
        BRIDGE_PATH,
        "--db",
        str(tmp_path / "memory.db"),
    ]
    proc = subprocess.Popen(cmd, env=env)
    url = f"http://127.0.0.1:{port}{BRIDGE_PATH}"
    try:
        wait_http_ready(url)
        yield {"url": url, "searxng": backends["searxng"]}
    finally:
        terminate(proc)


@asynccontextmanager
async def _session(bridge):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with (
        streamable_http_client(bridge["url"]) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        yield session


async def test_initialize_and_list_tools(bridge):
    async with _session(bridge) as session:
        init = await session.initialize()
        assert init.server_info.name == "search-agent"
        tools = await session.list_tools()
        assert [t.name for t in tools.tools] == ["search_memory"]
        schema = tools.tools[0].input_schema
        assert schema["required"] == ["query"]
        assert schema["properties"]["mode"]["enum"] == ["preference", "parallel"]


async def test_parallel_empty_store_returns_no_results(bridge):
    async with _session(bridge) as session:
        await session.initialize()
        res = await session.call_tool(
            "search_memory", arguments={"query": "python", "mode": "parallel"}
        )
        assert res.is_error is False
        assert "No results found" in res.content[0].text


async def test_preference_empty_store_triggers_live_search(bridge):
    async with _session(bridge) as session:
        await session.initialize()
        res = await session.call_tool(
            "search_memory", arguments={"query": "python", "mode": "preference"}
        )
        assert res.is_error is False
        text = res.content[0].text
        assert "Result 1" in text  # memory populated from live SearXNG stub
        assert "stub result" in text


async def test_preference_is_memory_first_on_second_query(bridge):
    def stats():
        with urllib.request.urlopen(f"{bridge['searxng']}/__stats") as r:
            return json.load(r)["requests"]

    async with _session(bridge) as session:
        await session.initialize()

        # 1st call: empty store -> live search seeds memory; recall clears threshold.
        r1 = await session.call_tool(
            "search_memory", arguments={"query": "python", "mode": "preference"}
        )
        assert "Result 1" in r1.content[0].text
        assert _score(r1.content[0].text) > 0.75

        before = await asyncio.to_thread(stats)
        # 2nd call: memory has a high-confidence match -> no live search.
        r2 = await session.call_tool(
            "search_memory", arguments={"query": "python", "mode": "preference"}
        )
        after = await asyncio.to_thread(stats)

    assert before == after  # SearXNG stub not hit again: memory-first path taken
    assert "Result 1" in r2.content[0].text
    assert _score(r2.content[0].text) > 0.75
