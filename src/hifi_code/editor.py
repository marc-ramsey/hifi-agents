"""Surgical edits with a propose-diff gate.

The loop's write primitive. Each edit replaces one exact ``old`` block with a
``new`` block in a single file. In ``propose_only`` mode nothing is written to
disk; the cumulative diff is surfaced instead for review.
"""


import ast


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

    def apply(self, edit: Edit) -> dict:
        if self._propose_only:
            self._proposed.append(edit)
            return {"status": "proposed", "edit": edit.to_dict()}

        with open(edit.path, "r", encoding="utf-8") as fh:
            content = fh.read()

        if edit.old == edit.new:
            return {"status": "noop", "reason": "old and new are identical",
                    "edit": edit.to_dict()}

        if edit.old not in content:
            return {"status": "rejected", "reason": "old text not found",
                    "edit": edit.to_dict()}

        if content.count(edit.old) > 1:
            return {"status": "rejected",
                    "reason": f"old text not unique (found {content.count(edit.old)} times)",
                    "edit": edit.to_dict()}

        original = content
        content = content.replace(edit.old, edit.new)

        if (edit.path.endswith(".py")
                and self._is_valid_py(original)
                and not self._is_valid_py(content)):
            return {"status": "rejected",
                    "reason": "edit produced invalid Python",
                    "edit": edit.to_dict()}

        with open(edit.path, "w", encoding="utf-8") as fh:
            fh.write(content)

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

    def diff(self) -> list[dict]:
        """Cumulative list of every proposed + applied edit."""
        return [e.to_dict() for e in self._proposed + self._applied]

    def state(self) -> dict:
        return {
            "applied": [e.to_dict() for e in self._applied],
            "proposed": [e.to_dict() for e in self._proposed],
        }
