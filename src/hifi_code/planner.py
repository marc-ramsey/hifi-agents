"""The coding loop: goal -> plan -> act -> verify -> repeat.

The planner holds the goal and accumulated state, asks the LLM for the next
action each tick, and dispatches to the navigation (ast-bro), edit (gated
editor), or verify (pytest/ruff) primitives. It stops on ``finish`` or when the
step budget is exhausted.
"""

import json
import logging
import re

import httpx

from .config import Settings
from .editor import Edit
from .prompt import system_prompt
from .sandbox import Sandbox

logger = logging.getLogger(__name__)


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
        text = content.strip()
        # Strip markdown code fences: ```json ... ```
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
            text = re.sub(r"\s*\n?```$", "", text)
        # Isolate the first JSON object in the text.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Single-quoted / Python-literal JSON: {'action': 'read'} -> dict.
        try:
            import ast

            value = ast.literal_eval(text)
            if isinstance(value, dict):
                return value
        except (ValueError, SyntaxError):
            pass
        # Tolerate bare (unquoted) keys and values: {action: navigate} ->
        # {"action": "navigate"}. Keys are quoted first (bare word before :);
        # then any remaining bare word after : or , that isn't a JSON literal.
        # Values may be path-like (a/b/c.py), so allow / . - ~ # in the run.
        fixed = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', text)
        fixed = re.sub(
            r"([:,]\s*)(?!(?:true|false|null)\b)"
            r"([A-Za-z_][A-Za-z0-9_/\.\-~#]*)(?=[\s,}\]])",
            r'\1"\2"',
            fixed,
        )
        if fixed != text:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
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
        result = await self._dispatch(action, decision)
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
        return {"action": "finish", "ok": True, "reason": decision.get("reason", "done")}

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
        raw: list[str] = []
        for key in ("symbols", "symbol"):
            val = decision.get(key)
            if isinstance(val, list):
                raw.extend(str(s) for s in val)
            elif isinstance(val, str):
                raw.append(val)
        args = decision.get("args")
        if isinstance(args, dict):
            for key in ("symbols", "symbol"):
                val = args.get(key)
                if isinstance(val, list):
                    raw.extend(str(s) for s in val)
                elif isinstance(val, str):
                    raw.append(val)
        return [s for s in (Planner._normalize_symbol(s) for s in raw) if s]

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
