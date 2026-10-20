"""MCP client for the ast-bro server.

ast-bro exposes 19 tools over stdio JSON-RPC 2.0. This client keeps a single
persistent session (via the mcp SDK's stdio transport) and exposes the
read-only tools as thin async methods. ``run --write`` is intentionally *not*
surfaced here — the loop performs edits through the gated editor instead.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class AstBro:
    """Typed MCP client over the ``ast-bro mcp`` server."""

    def __init__(
        self,
        cmd: list[str] | None = None,
        *,
        session: Any | None = None,
        server_name: str = "ast-bro",
    ):
        self._cmd = cmd or ["ast-bro", "mcp"]
        # ``session`` is an injected ClientSession (used by tests with an
        # InMemoryTransport-backed server). When None we own the stdio client.
        self._session = session
        self._owns_session = session is None
        self._server_name = server_name
        self._client = None
        self._initialized = False

    async def _ensure(self):
        if self._session is None:
            from mcp.client.session import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client

            command, *args = self._cmd
            server = StdioServerParameters(command=command, args=args)
            self._client = stdio_client(server)
            self._read, self._write = await self._client.__aenter__()
            # This SDK version starts the dispatcher on __aenter__, so the
            # session must be entered as an async context manager before
            # initialize() can send requests.
            self._session = await ClientSession(self._read, self._write).__aenter__()
            await self._session.initialize()
        elif not self._initialized:
            await self._session.initialize()
        self._initialized = True

    async def close(self):
        if self._owns_session and self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 - teardown best-effort
                logger.debug("error closing ast-bro session: %s", exc)
            self._session = None
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:  # noqa: S110, BLE001
                pass
            self._client = None

    async def call(self, name: str, arguments: dict | None = None) -> dict:
        """Generic dispatch to any tool by name; returns parsed JSON."""
        await self._ensure()
        result = await self._session.call_tool(name, arguments)
        text = "".join(getattr(part, "text", "") for part in result.content)
        if result.is_error:
            raise RuntimeError(f"ast-bro {name} error: {text}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
        if isinstance(data, dict) and data.get("ok") is False:
            raise RuntimeError(f"ast-bro {name} failed: {text}")
        return data

    async def list_tools(self) -> list[dict]:
        await self._ensure()
        resp = await self._session.list_tools()
        return [
            {"name": t.name, "description": getattr(t, "description", "")}
            for t in resp.tools
        ]

    # --- read-only tool wrappers -------------------------------------------
    async def map(self, paths=None, **kwargs) -> dict:
        return await self.call("map", {"paths": paths, **kwargs})

    async def digest(self, paths=None, **kwargs) -> dict:
        return await self.call("digest", {"paths": paths, **kwargs})

    async def show(self, path=None, **kwargs) -> dict:
        return await self.call("show", {"path": path, **kwargs})

    async def callers(self, target, **kwargs) -> dict:
        return await self.call("callers", {"target": target, **kwargs})

    async def callees(self, target, **kwargs) -> dict:
        return await self.call("callees", {"target": target, **kwargs})

    async def search(self, query, **kwargs) -> dict:
        return await self.call("search", {"query": query, **kwargs})

    async def impact(self, target, **kwargs) -> dict:
        return await self.call("impact", {"target": target, **kwargs})

    async def context(self, target, **kwargs) -> dict:
        return await self.call("context", {"target": target, **kwargs})

    async def find_related(self, path, **kwargs) -> dict:
        return await self.call("find_related", {"path": path, **kwargs})

    async def implements(self, target, **kwargs) -> dict:
        return await self.call("implements", {"target": target, **kwargs})

    async def surface(self, path, **kwargs) -> dict:
        return await self.call("surface", {"path": path, **kwargs})
