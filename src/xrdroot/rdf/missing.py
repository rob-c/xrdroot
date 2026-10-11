"""Columns some entries have no value of: ROOT 6.32's ``DefaultValueFor`` and its filters.

A chain of trees that do not all have the same branches, or a friend tree
indexed to match only some entries, leaves entries with no value of a
column. Reading one is an error, as it is in ROOT; these say what to do
instead. ``DefaultValueFor(column, value)`` reads ``value`` there,
``FilterAvailable(column)`` keeps only the entries that have a value, and
``FilterMissing(column)`` only those that do not.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from .graph import Availability, Defined, Filter, RawColumn, SourceColumn

__all__ = ["default_value_for", "filter_available"]


def _checked(frame: Any, column: str, what: str) -> None:
    """Refuse a column that is not the data's: only those can be missing."""
    found = frame._columns.get(column)
    if found is None:
        raise KeyError(f"{what}({column!r}) names a column this frame has none called")
    if not isinstance(found, SourceColumn):
        raise ValueError(
            f"{what}({column!r}) is for a column of the data, which an entry may have no value "
            f"of; {column!r} is one made here, which every entry has"
        )


def default_value_for(frame: Any, column: str, value: Any) -> Any:
    """``DefaultValueFor``: the column as the data has it, and ``value`` where it has none."""
    _checked(frame, column, "DefaultValueFor")

    def filled(values: Any, have: Any) -> Any:
        if not isinstance(values, np.ndarray) or values.ndim != 1:
            raise UnsupportedFeatureError(
                f"DefaultValueFor({column!r}) gives a value to a column of one number per "
                f"entry; {column!r} holds more than one in each"
            )
        return np.where(have, values, np.asarray(value).astype(values.dtype))

    inputs = {"0": RawColumn(column), "1": Availability(column)}
    made = Defined(column, filled, inputs, frame._node)
    return frame._derived(columns={**frame._columns, column: made})


def filter_available(frame: Any, column: str, missing: bool = False) -> Any:
    """``FilterAvailable`` - or, ``missing``, ``FilterMissing``: the entries that have a
    value of the column, or those that do not."""
    _checked(frame, column, "FilterMissing" if missing else "FilterAvailable")

    def kept(have: Any) -> Any:
        return ~have if missing else have

    return frame._derived(node=Filter(frame._node, kept, {"0": Availability(column)}, ""))
