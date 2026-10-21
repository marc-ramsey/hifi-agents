"""Surgical edits with a propose-diff gate.

The loop's write primitive. Each edit replaces one exact ``old`` block with a
``new`` block in a single file, optionally scoped to a ``region`` of the file
(used by symbol-scoped edits). In ``propose_only`` mode nothing is written to
disk; the cumulative diff is surfaced instead for review.
"""


import ast

import aiofiles


class Edit:
    def __init__(self, path: str, old: str, new: str):
        self.path = path
        self.old = old
        self.new = new

    def to_dict(self) -> dict:
        return {"path": self.path, "old": self.old, "new": self.new}


class Editor:
    def __init__(self, propose_only: bool = False):
        self._propose_only = propose_only
        self._applied: list[Edit] = []
        self._proposed: list[Edit] = []

    async def apply(self, edit: Edit, region: str | None = None) -> dict:
        if self._propose_only:
            self._proposed.append(edit)
            return {"status": "proposed", "edit": edit.to_dict()}

        try:
            async with aiofiles.open(edit.path, "r", encoding="utf-8") as fh:
                content = await fh.read()
        except FileNotFoundError:
            # ``old`` is empty: the file doesn't exist yet, so this edit creates
            # it. A non-empty ``old`` on a missing file is a stale edit -- the
            # model should have re-read or created the file first.
            if edit.old not in ("", None):
                return {"status": "rejected",
                        "reason": "file does not exist yet; use empty old to "
                                  "create it, then retry",
                        "edit": edit.to_dict()}
            content = ""

        # A ``region`` scopes the edit to a slice of the file. Symbol-scoped
        # edits pass the exact symbol source as the region so uniqueness is
        # enforced *within* the symbol: a common substring (``return``, ``self``)
        # that appears twice would otherwise be replaced everywhere it occurs.
        if region is not None:
            if region not in content:
                return {"status": "rejected",
                        "reason": "region (symbol source) not found in file; "
                                  "re-read the symbol via read, then retry",
                        "edit": edit.to_dict()}
            if content.count(region) > 1:
                return {"status": "rejected",
                        "reason": "region (symbol source) not unique in file; "
                                  "disambiguate the symbol, then retry",
                        "edit": edit.to_dict()}

        search = region if region is not None else content

        if edit.old == edit.new:
            return {"status": "noop", "reason": "old and new are identical",
                    "edit": edit.to_dict()}

        if edit.old not in search:
            return {"status": "rejected",
                    "reason": "old text not found; re-read to get the exact text, "
                              "then retry",
                    "edit": edit.to_dict()}

        # An empty ``old`` matches before every character, so the uniqueness guard
        # would reject a legitimate top-insert; only enforce it for a real substring.
        if edit.old != "" and search.count(edit.old) > 1:
            return {"status": "rejected",
                    "reason": f"old text not unique (found {search.count(edit.old)} times); "
                              "include surrounding lines to make it unique, then retry",
                    "edit": edit.to_dict()}

        original = content
        # An empty ``old`` with no region inserts at the top of the file (creating
        # it when it didn't exist); the generic replace would match the empty
        # string before every character instead.
        if edit.old == "" and region is None:
            new_content = edit.new + content
        else:
            new_content = self._replace_once(content, edit.old, edit.new, region)

        if (edit.path.endswith(".py")
                and self._is_valid_py(original)
                and not self._is_valid_py(new_content)):
            return {"status": "rejected",
                    "reason": "edit produced invalid Python; fix the syntax and retry",
                    "edit": edit.to_dict()}

        async with aiofiles.open(edit.path, "w", encoding="utf-8") as fh:
            await fh.write(new_content)

        self._applied.append(edit)
        return {"status": "applied", "edit": edit.to_dict()}

    @staticmethod
    def _is_valid_py(text: str) -> bool:
        """True if ``text`` parses as valid Python syntax."""
        try:
            ast.parse(text)
        except SyntaxError:
            return False
        return True

    @staticmethod
    def _replace_once(content: str, old: str, new: str, region: str | None) -> str:
        """Replace one occurrence of ``old`` with ``new``.

        When ``region`` is given the replacement is confined to that slice of
        ``content``; otherwise it behaves like ``str.replace`` over the whole
        string (the caller has already guaranteed a single match).
        """
        if region is None:
            return content.replace(old, new)
        start = content.index(region)
        end = start + len(region)
        return content[:start] + content[start:end].replace(old, new, 1) + content[end:]

    def diff(self) -> list[dict]:
        """Cumulative list of every proposed + applied edit."""
        return [e.to_dict() for e in self._proposed + self._applied]

    def state(self) -> dict:
        return {
            "applied": [e.to_dict() for e in self._applied],
            "proposed": [e.to_dict() for e in self._proposed],
        }
