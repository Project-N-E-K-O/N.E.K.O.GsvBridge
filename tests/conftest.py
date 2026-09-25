"""Make the api_neko package importable from a standalone checkout."""

import sys
from pathlib import Path


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

