"""Pytest configuration: ensure source packages are importable.

When running tests outside of a colcon-built workspace (e.g. directly
via ``python3 -m pytest test/``), the ROS2 Python packages are not on
sys.path.  This conftest adds the source directories so that imports
like ``from amr_perception.nodes...`` resolve correctly.
"""

import sys
from pathlib import Path

# Repository root (one level above test/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"

# Add each Python package's source directory to sys.path.
for pkg_dir in _SRC_DIR.iterdir():
    if pkg_dir.is_dir() and (pkg_dir / "setup.py").exists():
        if str(pkg_dir) not in sys.path:
            sys.path.insert(0, str(pkg_dir))
