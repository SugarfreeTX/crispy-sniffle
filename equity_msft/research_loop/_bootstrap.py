from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EQUITY_DIR = Path(__file__).resolve().parents[1]
LOOP_DIR = Path(__file__).resolve().parent


def ensure_import_paths() -> None:
    for path in (str(REPO_ROOT), str(EQUITY_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


ensure_import_paths()
