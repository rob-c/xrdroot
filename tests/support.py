"""What the tests share that is not a fixture."""

from __future__ import annotations

from typing import Any

import numpy as np

from xrdroot import Jagged


def plain(value: Any) -> Any:
    """Arrays made lists all the way down, so that nested results compare with ``==``.

    NumPy compares element by element, which is what an array is for and not
    what an assertion about a dict holding one wants; this is the same value
    with every array and every run of rows turned into the lists they hold.
    """
    if isinstance(value, (np.ndarray, Jagged)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(plain(item) for item in value)
    return value
