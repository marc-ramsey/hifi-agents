"""The coding loop: goal -> plan -> act -> verify -> repeat.

The planner holds the goal and accumulated state, asks the LLM for the next
action each tick, and dispatches to the navigation (ast-bro), edit (gated
editor), or verify (pytest/ruff) primitives. It stops on ``finish`` or when the
step budget is exhausted.
"""

import json
import logging

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
        self.state = {"map": None, "history": [], "last_verify": None}

    async def seed(self, goal: str, root: str = "."):
        self.goal = goal
        self.state["map"] = await self.astbro.digest(paths=[root])
        return self.state["map"]

    async def _llm_response(self) -> dict:
        messages = [
            {"role": "system", "content": system_prompt(self.goal, self.state["map"])},
        ]
        messages.extend(self.state["history"][-6:])
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
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end != -1:
                return json.loads(content[start:end + 1])
            raise

    async def tick(self) -> dict:
        if self.step >= self.settings.max_steps:
            return {"action": "finish", "reason": "step budget exhausted"}

        self.step += 1
        decision = await self._llm_response()
        action = decision.get("action", "finish")
        result = await self._dispatch(action, decision)
        self.state["history"].append({
            "step": self.step, "action": action,
            "decision": decision, "result": result,
        })
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
        tool = decision.get("tool")
        args = decision.get("args", {})
        fn = getattr(self.astbro, tool, None)
        if fn is None:
            return await self.astbro.call(tool, args)
        return await fn(**args)

    async def _read(self, decision: dict) -> dict:
        return await self.astbro.show(
            path=decision.get("path"), **decision.get("args", {})
        )

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
