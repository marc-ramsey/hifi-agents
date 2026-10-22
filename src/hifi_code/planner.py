"""The coding loop: goal -> plan -> act -> verify -> repeat.

The planner holds the goal and accumulated state, asks the LLM for the next
action each tick, and dispatches to the navigation (ast-bro), edit (gated
editor), or verify (pytest/ruff) primitives. It stops on ``finish`` or when the
step budget is exhausted.
"""

import ast
import asyncio
import json
import logging
import re
from typing import Any

import httpx

from .config import Settings
from .editor import Edit, Editor
from .prompt import system_prompt
from .sandbox import Sandbox
from .schema import Action, SchemaError, validate_decision
from .spec_schema import RevisionAction
from .verify import Verifier

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
            "verify_failures": 0,
            "suggest_revise": False,
        }

    async def seed(self, goal: str, root: str = "."):
        self.goal = goal
        # Build the ast-bro search index once before the first tick so `search`
        # / `find_related` work. Idempotent and safe in an existing repo; the
        # index may be empty. This runs unconditionally (not a model action) so
        # semantic search needs no explicit init and can't be missed.
        try:
            await self.astbro.index(paths=[root])
        except Exception as exc:  # noqa: BLE001 - search is best-effort
            logger.debug("ast-bro index build failed at seed: %s", exc)
        self.state["map"] = await self.astbro.digest(paths=[root])
        self.state["tools"] = [t["name"] for t in await self.astbro.list_tools()]
        return self.state["map"]

    @staticmethod
    def _history_messages(history: list[dict]) -> list[dict]:
        """Turn stored tick results into valid OpenAI assistant messages.

        History entries are ``{step, action, decision, result}`` dicts with no
        ``role`` key; the backend rejects such messages with a 502, so serialize
        the outcome as the assistant message before sending. The ``result`` is the
        harness's feedback for the model's previous turn -- the source a ``read``
        returned, an ``edit``'s applied/rejected status, or a ``verify`` outcome --
        which is what the model must see to react and self-correct, so it is
        serialized in preference to the raw ``decision`` (falling back to it when
        there is no result dict). The action label is preserved so the model knows
        which turn each response belongs to.
        """
        out = []
        for entry in history[-6:]:
            decision = entry.get("decision") or {}
            payload = entry.get("result")
            if not isinstance(payload, dict):
                payload = decision
            payload = dict(payload)
            payload.setdefault("action", decision.get("action", "finish"))
            out.append({
                "role": "assistant",
                "content": json.dumps(payload, ensure_ascii=False),
            })
        return out

    async def _llm_response(self) -> dict:
        tools = "\n".join(f"- {t}" for t in self.state.get("tools", []))
        messages = [
            {"role": "system", "content": system_prompt(
                self.goal, self._map_text(self.state["map"]), tools=tools)},
        ]
        messages.extend(self._history_messages(self.state["history"]))
        user = (
            f"Step {self.step}. Goal: {self.goal}. "
            "Choose the next action; respond with JSON only."
        )
        if self.state.get("suggest_revise"):
            # A concrete, heuristic-driven nudge: repeated unfixing verify failures
            # are the canonical "the spec is wrong" moment. Surface it so the model
            # can `revise` the offending field instead of burning the step budget.
            user += (
                f" Note: verify has failed {self.state['verify_failures']} times "
                "without a passing result or an applied edit. If this points to a "
                "wrong or unmodeled spec assumption (a structural gap), emit "
                "`revise` to re-draft that field; otherwise keep fixing the code."
            )
        messages.append({"role": "user", "content": user})
        content = await self.llm.chat(messages)
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
        try:
            validate_decision(decision)
        except SchemaError as exc:
            # The decision parsed but violated the action schema (unknown action,
            # missing required field, or wrong type). Reject it before dispatch
            # so the contract is enforced, not just advised, and record the
            # rejection so the model sees it and can self-correct. The stall
            # budget still bounds a repeat loop.
            decision = {
                "action": decision.get("action"),
                "status": "invalid",
                "ok": False,
                "reason": f"invalid decision: {exc}",
                "hint": "valid actions: " + " | ".join(Action.ALL),
            }
            self.state["history"].append({
                "step": self.step, "action": decision["action"],
                "decision": decision, "result": decision,
            })
            if self._stalled(decision["action"], decision):
                return self._stalled_finish()
            return decision
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
        self._update_revise_signal(action, result)
        if self._stalled(action, result):
            return self._stalled_finish()
        return result

    async def _dispatch(self, action: str, decision: dict) -> dict:
        if action == "navigate":
            return await self._navigate(decision)
        if action == "read":
            return await self._read(decision)
        if action == "edit":
            return await self._edit(decision)
        if action == "plan":
            return await self._plan(decision)
        if action == "verify":
            return await self._verify(decision)
        if action == "revise":
            return await self._revise(decision)
        if action == "finish":
            return {"action": "finish", "ok": True, "reason": decision.get("reason", "done")}
        return {
            "action": "finish",
            "status": "error",
            "ok": False,
            "reason": f"unknown action: {action!r}",
            "hint": "valid actions: navigate | read | edit | plan | verify | finish",
        }

    async def _navigate(self, decision: dict) -> dict:
        tool = decision.get("tool") or (
            decision.get("args") or {}).get("tool")
        args = decision.get("args") or {}
        # Drop null args: the LLM sometimes passes None, which ast-bro rejects
        # as an invalid type. A read tool with no usable input is served the
        # map already fetched at seed time instead of crashing.
        args = {k: v for k, v in args.items() if v is not None}
        known = set(self.state.get("tools", []))
        if known and tool not in known:
            return {
                "status": "error",
                "reason": f"unknown ast-bro tool: {tool}",
                "hint": "valid tools: " + ", ".join(sorted(known)),
            }
        if tool in ("map", "digest") and not args.get("paths"):
            return self.state.get("map") or {"ok": True, "map": ""}
        # `show` pulls a symbol's full source, which is the `read` action's
        # job. The model sometimes routes it here via navigate with only a
        # path (no symbols), which would make ast-bro reject with
        # "missing field symbols". Route it through _read so it resolves
        # symbols, falls back to the map, or fails with a helpful hint.
        if tool == "show":
            return await self._read(decision)
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
        # The edit path may live inside each edit ({path, old, new}) or once at
        # the top level ({path, edits: [...]}); accept both. The schema only
        # requires ``edits`` to be a list, and the model sometimes names the file
        # once rather than repeating it on every edit.
        top_path = decision.get("path") or (decision.get("args") or {}).get("path")
        for edit in decision.get("edits", []):
            path = edit.get("path") or top_path
            if not path:
                results.append({"status": "rejected",
                                "reason": "edit missing 'path'; include the file "
                                          "path on the edit or at the top level",
                                "edit": edit})
                continue
            edit = {**edit, "path": path}
            sandbox = Sandbox(
                self.settings.working_dir,
                self.settings.denylist,
                enabled=self.settings.sandbox_enabled,
            )
            if not sandbox.check_edit(path):
                results.append({"status": "blocked", "reason": "denylist",
                                "edit": edit})
                continue
            # Symbol-scoped edit: the model names a symbol and a substring to
            # change; the harness fetches the exact source via ast-bro ``show`` so
            # the model never has to reproduce the file body verbatim.
            if "symbol" in edit:
                results.append(await self._edit_symbol_scope(edit))
                continue
            results.append(await self.editor.apply(Edit(
                path=path, old=edit["old"], new=edit["new"]
            )))
        return {"edits": results}

    @staticmethod
    def _source_of_symbol(shown: Any, symbol: str) -> str | None:
        """Return the exact source of ``symbol`` from an ast-bro ``show`` result.

        The real server returns a ``raw`` outline where each entry is a header
        line ``# <path>:<start>-<end> <qualified_name> (<kind>)`` (optionally
        followed by a ``# in: class X`` annotation) and the symbol's body
        beneath it, terminated by the next header line. ``symbol`` is matched
        against the header's qualified name either exactly (``greet``) or as its
        trailing component (``_read`` within ``Planner._read``), and that body is
        returned verbatim so the edit targets real bytes.

        A structured ``{files: [{matches: [{qualified_name, source}]}]}`` result
        (as the test fake emits) is still accepted for backward compatibility.
        """
        if isinstance(shown, dict):
            for f in shown.get("files", []):
                for m in f.get("matches", []):
                    if m.get("qualified_name") == symbol:
                        return m.get("source")
            raw = shown.get("raw")
        else:
            raw = None
        if not isinstance(raw, str):
            return None
        header_re = re.compile(
            r"^#\s*(?P<path>.*?)(?::(?P<start>\d+)-(?P<end>\d+))\s+"
            r"(?P<qual>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\((?P<kind>\w+)\)\s*$"
        )
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            m = header_re.match(line)
            # Require a real path (contains '/') so a body comment that merely
            # looks like a header line is not mistaken for one.
            if not m or "/" not in m.group("path"):
                continue
            qual = m.group("qual")
            if qual != symbol and qual.split(".")[-1] != symbol:
                continue
            body = i + 1
            if body < len(lines) and lines[body].lstrip().startswith("# in:"):
                body += 1
            end = body
            while end < len(lines) and not header_re.match(lines[end]):
                end += 1
            return "\n".join(lines[body:end])
        return None

    async def _edit_symbol_scope(self, edit: dict) -> dict:
        """Change a substring inside one symbol, fetching exact source via show.

        The model supplies {path, symbol, from, to}; the harness reads the exact
        symbol source (re-parsed live by ast-bro) so the edit targets real bytes
        instead of text the model would otherwise have to reproduce from memory.
        """
        path = edit.get("path")
        symbol = edit.get("symbol")
        from_str = edit.get("from")
        to_str = edit.get("to", "")
        if not path or not symbol or from_str is None:
            return {"status": "rejected",
                    "reason": "symbol-scoped edit needs {path, symbol, from}",
                    "edit": edit}
        shown = await self.astbro.show(path=path, symbols=[symbol])
        source = self._source_of_symbol(shown, symbol)
        if source is None:
            return {"status": "rejected",
                    "reason": f"symbol {symbol!r} not found in {path}",
                    "edit": edit}
        if from_str not in source:
            return {"status": "rejected",
                    "reason": f"'{from_str}' not found in {symbol}; re-read the "
                              f"symbol via read, then retry",
                    "edit": edit}
        # Delegate to the editor with the symbol source as the region so the
        # uniqueness guard fires *inside* the symbol: a substring that appears
        # twice would otherwise be replaced everywhere it occurs.
        return await self.editor.apply(Edit(path=path, old=from_str, new=to_str),
                                       region=source)

    async def _verify(self, decision: dict) -> dict:
        self.state["last_verify"] = await self.verifier.verify(self.settings.verify)
        return self.state["last_verify"]

    async def _revise(self, decision: dict) -> dict:
        """Validate a revise request and return a terminal result for the outer loop.

        The revise action is the back-channel that makes the spec a living
        document: the inner loop emits one when it hits a structural
        contradiction or an unmodeled gap, and the outer loop applies it and
        re-seeds. ``validate_decision`` already enforced ``field``/``op``, so
        this only guards ``op`` is a known mutation and returns the terminal
        result the outer loop consumes. Unlike ``finish``, a revise is not
        success or failure -- it is a request to re-draft -- so the outer loop
        drives the next coding pass after applying it.
        """
        ops = (RevisionAction.ADD.value, RevisionAction.REMOVE.value,
               RevisionAction.REPLACE.value)
        op = decision.get("op")
        if not decision.get("field") or op not in ops:
            return {
                "action": "revise",
                "ok": False,
                "reason": "revise needs a field and op (add|remove|replace)",
                "hint": "valid ops: add | remove | replace",
            }
        return {
            "action": "revise",
            "field": decision.get("field"),
            "op": op,
            "value": decision.get("value"),
            "reason": decision.get("reason", ""),
            "ok": True,
        }

    async def _plan(self, decision: dict) -> dict:
        """Drive a goal decomposition in parallel across disjoint paths.

        The model supplies ``{subgoals: [{goal, path}, ...]}``; each subgoal is a
        self-contained coding target for one file. Paths must be disjoint -- that
        is the invariant that makes running them concurrently safe (no two tasks
        write the same file, so the editor's uniqueness guard and the file writes
        never race). Each subgoal is driven to completion by its own nested
        Planner, and their terminal results are concatenated.
        """
        subgoals = decision.get("subgoals", [])
        if not subgoals:
            return {
                "action": "finish",
                "status": "error",
                "ok": False,
                "reason": "plan produced no subgoals",
                "hint": "provide subgoals: [{goal, path}, ...]",
            }

        # Enforce the disjoint-path invariant: a path claimed by 2+ subgoals would
        # let two concurrent tasks write the same file. Fail loudly rather than
        # race.
        claimed: dict[str, int] = {}
        for g in subgoals:
            path = g.get("path") or "."
            claimed[path] = claimed.get(path, 0) + 1
        conflicts = sorted(p for p, n in claimed.items() if n > 1)
        if conflicts:
            return {
                "action": "finish",
                "status": "error",
                "ok": False,
                "reason": ("subgoals share paths (violate disjoint-path "
                           f"invariant): {conflicts}"),
                "hint": "each subgoal must target a distinct file path",
            }

        results = await asyncio.gather(
            *(self._drive_subgoal(g) for g in subgoals)
        )
        return {"subgoals": results}

    async def _drive_subgoal(self, subgoal: dict) -> dict:
        """Drive one subgoal to a terminal finish in its own Planner + session."""
        goal = subgoal.get("goal", "")
        path = subgoal.get("path") or "."
        if not goal:
            return {"status": "error",
                    "reason": "subgoal missing 'goal'",
                    "subgoal": subgoal}
        astbro = self._make_astbro()
        editor = Editor(propose_only=self.settings.propose_only)
        verifier = Verifier()
        sub_planner = Planner(self.settings, self.llm, astbro, editor, verifier)
        try:
            await sub_planner.seed(goal, path)
            while True:
                result = await sub_planner.tick()
                if result.get("action") == "finish":
                    return result
        finally:
            await astbro.close()

    def _make_astbro(self):
        """Construct the ast-bro client for a nested sub-planner.

        Each concurrent subgoal gets its own session because the underlying MCP
        stdio stream is not safe to share across coroutines. Tests monkeypatch
        this to inject a fake.
        """
        from .astbro import AstBro
        return AstBro(cmd=self.settings.server_cmd)

    def _stalled_finish(self) -> dict:
        """Terminal stalled-finish result shared by the stall detectors."""
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

    def _update_revise_signal(self, action: str, result: dict) -> None:
        """Track whether the loop is stuck failing verify without converging.

        This is the policy signal behind the `revise` back-channel: a single
        failing verify is normal (edit, verify, fix more), but ``revise_threshold``
        consecutive failing verifies with no applied edit or passing verify in
        between means the loop cannot fix the problem by editing -- the spec's
        structural assumptions (trigger/data_model/modules/seams) are likely wrong.
        When the counter crosses the threshold we flag ``suggest_revise`` so the
        next prompt carries a concrete suggestion to `revise` that field. It is
        only a suggestion -- the model still decides -- and ordinary code fixes
        keep the counter at zero by applying an edit or passing verify, so the gate
        fires on structural stuckness, not every code fix.

        The counter grows only on a failing verify and resets on a passing verify
        or an applied edit; other actions leave it untouched.
        """
        if action == "verify":
            if result.get("ok"):
                self.state["verify_failures"] = 0
                self.state["suggest_revise"] = False
            else:
                self.state["verify_failures"] += 1
                self.state["suggest_revise"] = (
                    self.state["verify_failures"] >= self.settings.revise_threshold
                )
        elif action == "edit" and any(
            e.get("status") == "applied" for e in result.get("edits", [])
        ):
            self.state["verify_failures"] = 0
            self.state["suggest_revise"] = False

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
        if action == "revise":
            # A revise is a deliberate re-plan, not stagnation -- reset the
            # counter and return False so it is never masked as a stalled finish.
            self.state["stall"] = 0
            return False
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
            else:
                self.state["stall"] += 1
            return self.state["stall"] >= self.settings.max_stalls
        else:
            self.state["stall"] += 1
        return self.state["stall"] >= self.settings.max_stalls
