"""Self-contained stub servers for the bridge integration test.

These let the test exercise the real hifi_search HTTP bridge end-to-end without
hardcoding the user's live servers (localhost:8080 / maceo.local:8888). The test
spawns one of these as a subprocess, points the bridge at it via env vars, and
verifies the two processes can talk to each other over HTTP.

Run as either:
    python stub_server.py --kind embed   --port 9001 [--dim 768]
    python stub_server.py --kind searxng --port 9002

Both are stdlib-only (http.server) so the subprocess needs no third-party deps.
Embeddings are deterministic (text -> hashed, normalized float vector) so the
same text always maps to the same vector, which is what makes semantic recall
reproducible across the bridge.
"""

import argparse
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_KIND = None
_PORT = None
_DIM = 768
_lock = threading.Lock()
_requests = 0


def _embed(text: str) -> list:
    """Deterministic, unit-normalized embedding of ``text`` into ``_DIM`` dims."""
    vec = [0.0] * _DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.sha1(token.encode()).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    norm = (sum(v * v for v in vec)) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _searxng_results(query: str) -> list:
    """Canned SearXNG results: chunkable content dominated by the query term.

    The query word is repeated so its embedding dimension dominates the vector,
    which makes semantic recall deterministic and clears the similarity threshold.
    """
    q = (query or "python").strip() or "python"
    words = [q] * 200 + ["programming", "tutorial", "documentation"] * 4
    content = " ".join(words)
    return [
        {
            "title": f"{q} - stub result",
            "url": f"https://example.com/stub/{hashlib.md5(q.encode()).hexdigest()[:8]}",
            "content": content,
            "snippet": content,
        }
    ]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence default stderr logging
        pass

    def _send_json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _count(self) -> None:
        global _requests
        with _lock:
            _requests += 1

    def do_GET(self) -> None:
        if self.path.endswith("/__stats"):
            with _lock:
                return self._send_json(200, {"requests": _requests})
        self._count()
        if _KIND == "searxng":
            return self._send_json(200, {"query": "", "results": _searxng_results("")})
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        self._count()
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if _KIND == "embed":
            inputs = payload.get("input", [])
            data = [
                {"object": "embedding", "index": i, "embedding": _embed(t)}
                for i, t in enumerate(inputs)
            ]
            return self._send_json(
                200,
                {
                    "object": "list",
                    "model": payload.get("model", ""),
                    "data": data,
                },
            )
        self._send_json(404, {"error": "not found"})


def main() -> None:
    global _KIND, _PORT, _DIM
    parser = argparse.ArgumentParser(description="test stub MCP backend")
    parser.add_argument("--kind", required=True, choices=["embed", "searxng"])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dim", type=int, default=768)
    args = parser.parse_args()

    _KIND, _PORT, _DIM = args.kind, args.port, args.dim
    server = ThreadingHTTPServer(("127.0.0.1", _PORT), _Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
