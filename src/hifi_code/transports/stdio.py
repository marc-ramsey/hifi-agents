"""stdio transport: run the loop and emit newline-delimited JSON."""

import json
import logging

logger = logging.getLogger(__name__)

from ..astbro import AstBro
from ..config import Settings
from ..editor import Editor
from ..llm import LLMClient
from ..planner import Planner
from ..verify import Verifier


async def _drive(settings: Settings, goal: str, root: str):
    astbro = AstBro(cmd=settings.server_cmd)
    llm = LLMClient(settings.llm)
    editor = Editor(propose_only=settings.propose_only)
    verifier = Verifier()
    planner = Planner(settings, llm, astbro, editor, verifier)

    await planner.seed(goal, root)
    try:
        while True:
            result = await planner.tick()
            yield result
            # A revise is terminal for the inner loop too: it hands control back
            # to the outer loop to re-draft the spec, which then re-seeds us.
            if result.get("action") in ("finish", "revise"):
                break
            # After edits or a plan (which ran parallel subtasks that may have
            # written files), run verify to confirm nothing broke.
            if "edits" in result or "subgoals" in result:
                yield await verifier.verify(settings.verify)
    finally:
        await astbro.close()


async def run(settings: Settings, goal: str, root: str = "."):
    async for result in _drive(settings, goal, root):
        print(json.dumps(result, ensure_ascii=False))
