"""Edit sandbox: denylist enforcement + working-dir boundary.

A denylist entry blocks a file if the path equals the entry or lives beneath it
(as a directory). The working-dir boundary additionally blocks any edit outside
the project root. Together these protect the repo's "Do Not Touch" files and keep
the loop from writing anywhere else on the filesystem.
"""

import os


def _abs(path: str, base: str | None = None) -> str:
    """Normalize a path to an absolute, separator-suffixed form for containment.

    Expanduser handles ``~`` and normpath collapses ``..``/``.`` so a relative
    path like ``foo`` and an absolute one resolve to the same comparison key.
    A relative ``path`` is resolved against ``base`` (defaulting to the CWD), so
    denylist entries are always relative to the project root.
    """
    if base is not None and not os.path.isabs(path):
        path = os.path.join(base, path)
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(path))))


class Sandbox:
    def __init__(self, working_dir: str = ".", denylist=None, enabled: bool = True):
        self.working_dir = _abs(working_dir)
        self.denylist = [_abs(entry, base=self.working_dir) for entry in (denylist or [])]
        self.enabled = enabled

    def is_denied(self, path: str) -> bool:
        target = _abs(path)
        # Containment: the file must live inside the working directory.
        if not target.startswith(self.working_dir + os.sep):
            return True
        for entry in self.denylist:
            if target == entry or target.startswith(entry + os.sep):
                return True
        return False

    def check_edit(self, path: str) -> bool:
        """Return True if the edit is allowed, False if blocked."""
        if not self.enabled:
            return True
        return not self.is_denied(path)
