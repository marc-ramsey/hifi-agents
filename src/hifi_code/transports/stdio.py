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
            if result.get("action") == "finish":
                break
            if "edits" in result:
                yield verifier.verify(settings.verify)
    finally:
        await astbro.close()


async def run(settings: Settings, goal: str, root: str = "."):
    async for result in _drive(settings, goal, root):
        print(json.dumps(result, ensure_ascii=False))
