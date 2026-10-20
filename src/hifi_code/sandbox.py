"""Edit sandbox: denylist enforcement + working-dir boundary.

A denylist entry blocks a file if the path equals the entry or lives beneath it
(as a directory). This protects the repo's "Do Not Touch" files.
"""


class Sandbox:
    def __init__(self, working_dir: str = ".", denylist=None, enabled: bool = True):
        self.working_dir = working_dir
        self.denylist = list(denylist or [])
        self.enabled = enabled

    def is_denied(self, path: str) -> bool:
        for entry in self.denylist:
            if path == entry or path.startswith(entry.rstrip("/") + "/"):
                return True
        return False

    def check_edit(self, path: str) -> bool:
        """Return True if the edit is allowed, False if blocked."""
        if not self.enabled:
            return True
        return not self.is_denied(path)
