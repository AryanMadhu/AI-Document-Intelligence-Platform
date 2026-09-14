"""Pytest configuration: ensure the workspace root is importable.

Adds the workspace root (this file's directory) to ``sys.path`` so that the
``src`` package is importable as ``src.<module>`` regardless of the directory
pytest is invoked from.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
