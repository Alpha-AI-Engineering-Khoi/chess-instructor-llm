"""Make the repo root importable so ``import src...`` / ``import config...`` work.

The project uses PEP-420 namespace packages with no top-level ``__init__.py``, so
tests must run with the repo root on ``sys.path``. Inserting it here means the
suite imports correctly no matter how pytest is invoked (``pytest``,
``python -m pytest``, from any cwd).
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
