"""The coding loop: goal -> plan -> act -> verify -> repeat.

The planner holds the goal and accumulated state, asks the LLM for the next
action each tick, and dispatches to the navigation (ast-bro), edit (gated
editor), or verify (pytest/ruff) primitives. It stops on ``finish`` or when the
step budget is exhausted.
"""

import ast
import json
import logging
import re

import httpx

from .config import Settings
from .editor import Edit
from .prompt import system_prompt
from .sandbox import Sandbox

logger = logging.getLogger(__name__)


def _literal_dict(text: str) -> dict:
    """ast.literal_eval the text if it's a (Python-style) dict, else raise."""
    value = ast.literal_eval(text)
    if isinstance(value, dict):
        return value
    raise ValueError("not a dict")


_BARE_VALUE_CHARS = r"[A-Za-z0-9_./~#-]"


def _quote_bare(text: str) -> str:
    """Quote bare (unquoted) keys and values in a JSON-like string.

    ``{action: navigate, path: a/b.py}`` -> ``{"action": "navigate", "path": "a/b.py"}`.

    Bare keys are always quoted; bare numbers (``123``, ``-3.14``) are left
    unquoted; bare strings are quoted. The delimiter (``{``, ``"``, or ``:``)
    and any surrounding whitespace are captured and re-emitted so the
    separators survive the substitution.
    """
    def repl(m: re.Match) -> str:
        if m.group(3) is not None:  # bare key: group1=delim, group2=ws, group3=key
            return f'{m.group(1)}{m.group(2)}"{m.group(3)}"'
        if m.group(4) is not None:  # bare number: leave unquoted
            return f'{m.group(4)}{m.group(5)}{m.group(6)}'
        return f'{m.group(7)}{m.group(8)}"{m.group(9)}"'  # bare string: quote

    return re.sub(
        r"([{,])(\s*)([A-Za-z_]\w*)(?=\s*:)"                                  # bare key
        r"|([,:])(\s*)(-?\d+(?:\.\d+)?)(?=[\s,}\]])"                          # bare number
        r"|([,:])(\s*)(?!(?:true|false|null)\b)(" + _BARE_VALUE_CHARS + r"+)(?=[\s,}\]])",  # bare string
        repl,
        text,
    )


def _extract_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` object in ``text``.

    Unlike ``find``/``rfind`` (which grab from the first ``{`` to the last
    ``}`` and can swallow trailing prose or a second object), this walks the
    string tracking brace depth while skipping quoted regions, stopping at the
    first balanced close.
    """
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    quote = None
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    return text[start:]


class Planner:
    def __init__(self, settings: Settings, llm, astbro, editor, verifier):
        self.settings = settings
        self.llm = llm
        self.astbro = astbro
        self.editor = editor
        self.verifier = verifier
        self.goal = None
        self.step = 0
        self.state = {
            "map": None,
            "history": [],
            "last_verify": None,
            "stall": 0,
        }

    async def seed(self, goal: str, root: str = "."):
        self.goal = goal
        self.state["map"] = await self.astbro.digest(paths=[root])
        self.state["tools"] = [t["name"] for t in await self.astbro.list_tools()]
        return self.state["map"]

    @staticmethod
    def _history_messages(history: list[dict]) -> list[dict]:
        """Turn stored tick results into valid OpenAI assistant messages.

        History entries are ``{step, action, decision, result}`` dicts with no
        ``role`` key; the backend rejects such messages with a 502, so serialize
        the decision as the assistant message before sending.
        """
        out = []
        for entry in history[-6:]:
            decision = entry.get("decision")
            content = (
                json.dumps(decision, ensure_ascii=False)
                if isinstance(decision, dict)
                else str(decision)
            )
            out.append({"role": "assistant", "content": content})
        return out

    async def _llm_response(self) -> dict:
        tools = "\n".join(f"- {t}" for t in self.state.get("tools", []))
        messages = [
            {"role": "system", "content": system_prompt(
                self.goal, self._map_text(self.state["map"]), tools=tools)},
        ]
        messages.extend(self._history_messages(self.state["history"]))
        messages.append(
            {"role": "user", "content": (
                f"Step {self.step}. Goal: {self.goal}. "
                "Choose the next action; respond with JSON only."
            )}
        )
        content = self.llm.chat(messages)
        return self._parse(content)

    @staticmethod
    def _parse(content: str) -> dict:
        """Parse a JSON decision out of a possibly-messy LLM string.

        Tolerates markdown fences, surrounding prose, single-quoted Python-style
        dicts, and bare (unquoted) keys/values, e.g. ``action: navigate``.
        """
        text = content.strip()
        if text.startswith("```"):  # strip a ```json ... ``` fence
            text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
            text = re.sub(r"\s*\n?```$", "", text)
        text = _extract_json_object(text)
        for parser in (json.loads, _literal_dict):
            try:
                return parser(text)
            except (ValueError, SyntaxError):
                pass
        # Quote bare keys/values ({action: navigate} -> {"action": "navigate"}).
        try:
            return json.loads(_quote_bare(text))
        except (ValueError, SyntaxError):
            pass
        raise ValueError(f"could not parse LLM JSON from: {content[:200]!r}")

    async def tick(self) -> dict:
        if self.step >= self.settings.max_steps:
            return {"action": "finish", "reason": "step budget exhausted"}

        self.step += 1
        try:
            decision = await self._llm_response()
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            # LLM transport/status/parse failure: degrade gracefully, don't crash.
            return {
                "action": "finish",
                "status": "error",
                "ok": False,
                "reason": f"llm turn failed: {exc}",
                "hint": "check the LLM endpoint and model availability",
            }
        action = decision.get("action", "finish")
        try:
            result = await self._dispatch(action, decision)
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            # A primitive failed (malformed edit dict, unexpected tool kwargs, or
            # an ast-bro tool error): degrade gracefully like an LLM failure
            # instead of crashing the loop with a traceback.
            return {
                "action": "finish",
                "status": "error",
                "ok": False,
                "reason": f"action failed: {exc}",
                "hint": "the requested edit or tool call could not be applied",
            }
        self.state["history"].append({
            "step": self.step, "action": action,
            "decision": decision, "result": result,
        })
        if self._stalled(action, result):
            return {
                "action": "finish",
                "status": "stalled",
                "ok": False,
                "reason": (f"no progress for {self.settings.max_stalls} "
                           f"consecutive steps"),
                "hint": ("the loop is repeating itself or edits keep failing "
                         "verify; re-check the goal, the target file, and the "
                         "verify command"),
            }
        return result

    async def _dispatch(self, action: str, decision: dict) -> dict:
        if action == "navigate":
            return await self._navigate(decision)
        if action == "read":
            return await self._read(decision)
        if action == "edit":
            return await self._edit(decision)
        if action == "verify":
            return self._verify(decision)
        if action == "finish":
            return {"action": "finish", "ok": True, "reason": decision.get("reason", "done")}
        return {
            "action": "finish",
            "status": "error",
            "ok": False,
            "reason": f"unknown action: {action!r}",
            "hint": "valid actions: navigate | read | edit | verify | finish",
        }

    async def _navigate(self, decision: dict) -> dict:
        tool = decision.get("tool") or (
            decision.get("args") or {}).get("tool")
        args = decision.get("args") or {}
        known = set(self.state.get("tools", []))
        if known and tool not in known:
            return {
                "status": "error",
                "reason": f"unknown ast-bro tool: {tool}",
                "hint": "valid tools: " + ", ".join(sorted(known)),
            }
        fn = getattr(self.astbro, tool, None)
        if fn is None:
            return await self.astbro.call(tool, args)
        return await fn(**args)

    async def _read(self, decision: dict) -> dict:
        args = decision.get("args")
        if not isinstance(args, dict):
            args = {}
        path = decision.get("path") or args.get("path")
        symbols = self._resolve_symbols(decision)
        if not symbols:
            symbols = self._symbols_from_map(self._map_text(self.state["map"]), path)
        if not path or not symbols:
            return {
                "status": "error",
                "reason": "read needs a path and at least one symbol",
                "hint": 'pass {path, symbols:["Planner"]} using names from the map',
            }
        return await self.astbro.show(path=path, symbols=symbols)

    @staticmethod
    def _resolve_symbols(decision: dict) -> list[str]:
        """Collect symbol names the model named, from {symbols} or {symbol}.

        The model is prompted to emit {path, symbols} but sometimes omits
        ``symbols`` or writes a single ``symbol`` string, so both shapes are
        accepted and normalized (``class Foo`` -> ``Foo``, ``run()`` -> ``run``).
        """
        raw = list(Planner._symbol_values(decision))
        args = decision.get("args")
        if isinstance(args, dict):
            raw.extend(Planner._symbol_values(args))
        return [s for s in (Planner._normalize_symbol(s) for s in raw) if s]

    @staticmethod
    def _symbol_values(node: dict) -> list[str]:
        """Yield the raw symbol strings under ``symbols`` / ``symbol`` keys."""
        out: list[str] = []
        for key in ("symbols", "symbol"):
            val = node.get(key)
            if isinstance(val, list):
                out.extend(str(s) for s in val)
            elif isinstance(val, str):
                out.append(val)
        return out

    @staticmethod
    def _normalize_symbol(name: str) -> str:
        s = re.sub(
            r'^(async\s+)?(?:class|func|function|method|property|attribute|var)\s+',
            "", str(name).strip(), flags=re.IGNORECASE,
        )
        s = re.sub(r'\s*\[.*?\]\s*$', "", s)
        s = re.sub(r"\(\)\s*$", "", s)
        return s.strip()

    @staticmethod
    def _map_text(state_map) -> str:
        if isinstance(state_map, str):
            return state_map
        if isinstance(state_map, dict):
            for key in ("raw", "text", "content", "digest"):
                val = state_map.get(key)
                if isinstance(val, str):
                    return val
        return ""

    @staticmethod
    def _symbols_from_map(text: str, path: str) -> list[str]:
        """Derive symbol names for ``path`` from a text digest map.

        The digest is a text outline like::

            /\n  planner.py ([small], 142 lines, ...)
              class Planner  L21-141
                __init__()  [async] seed()  [async] tick()

        so we locate the file section and pull out every ``class Foo`` / ``foo()``
        name to use as a fallback ``symbols`` list for ``show``.
        """
        if not text or not path:
            return []
        target = path.strip("./")
        base = target.split("/")[-1]
        header_re = re.compile(r"^(\s*)(\S+?)\s*\(")
        kind_re = re.compile(
            r"\b(?:class|func|function|method|property|attribute|var)\s+([A-Za-z_]\w*)"
        )
        call_re = re.compile(r"\b([A-Za-z_]\w*)\s*\(\)")

        def header_matches(stripped: str) -> bool:
            m = header_re.match(stripped)
            if not m:
                return False
            hdr = m.group(2).strip("./")
            return hdr == target or hdr.split("/")[-1] == base

        symbols: list[str] = []
        in_file = False
        file_indent = -1
        for raw in text.splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            stripped = raw.strip()
            if not in_file:
                if header_matches(stripped):
                    in_file = True
                    file_indent = len(raw) - len(raw.lstrip())
                continue
            indent = len(raw) - len(raw.lstrip())
            if indent <= file_indent:
                in_file = False
                if header_matches(stripped):
                    in_file = True
                    file_indent = indent
                continue
            symbols.extend(kind_re.findall(stripped))
            symbols.extend(call_re.findall(stripped))

        seen = set()
        return [s for s in symbols if not (s in seen or seen.add(s))]

    async def _edit(self, decision: dict) -> dict:
        results = []
        for edit in decision.get("edits", []):
            sandbox = Sandbox(
                self.settings.working_dir,
                self.settings.denylist,
                enabled=self.settings.sandbox_enabled,
            )
            if not sandbox.check_edit(edit["path"]):
                results.append({"status": "blocked", "reason": "denylist",
                                "edit": edit})
                continue
            results.append(self.editor.apply(Edit(
                path=edit["path"], old=edit["old"], new=edit["new"]
            )))
        return {"edits": results}

    def _verify(self, decision: dict) -> dict:
        self.state["last_verify"] = self.verifier.verify(self.settings.verify)
        return self.state["last_verify"]

    def _stalled(self, action: str, result: dict) -> bool:
        """True once the loop has made no meaningful progress for ``max_stalls``
        consecutive ticks.

        Progress is an applied edit (a real file change) or a passing verify.
        Everything else -- reads, navigations, rejected / no-op edits, and
        failing verifies -- is stagnation. That is how the loop gets stuck
        re-reading the same thing or re-submitting the same edit, so we surface
        it as a terminal ``stalled`` finish instead of burning the whole step
        budget.
        """
        if action == "edit":
            if any(
                e.get("status") == "applied" for e in result.get("edits", [])
            ):
                self.state["stall"] = 0
            else:
                self.state["stall"] += 1
        elif action == "verify":
            if result.get("ok"):
                self.state["stall"] = 0
            return False
        else:
            self.state["stall"] += 1
        return self.state["stall"] >= self.settings.max_stalls
