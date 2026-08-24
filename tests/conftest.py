"""Make the repo-root modules and the eval harness importable from tests/."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for entry in (str(REPO_ROOT), str(REPO_ROOT / "eval")):
    if entry not in sys.path:
        sys.path.insert(0, entry)
