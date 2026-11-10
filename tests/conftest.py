"""Bind imports to this checkout, regardless of the clone directory name."""

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "api_neko", REPO_ROOT / "__init__.py", submodule_search_locations=[str(REPO_ROOT)]
)
package = importlib.util.module_from_spec(spec)
sys.modules["api_neko"] = package
spec.loader.exec_module(package)

