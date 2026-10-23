"""C++ value construction preserves payloads while making independent object values."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


def value_copy(value: Any) -> Any:
    """``T b = a;`` for an object: a copy of it, not the same object under a second name."""
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        return copy.deepcopy(value)
    except Exception:
        return copy.copy(value)

