"""What a translation calls to write through a reference: the standard's, over :mod:`cppstd`."""

from __future__ import annotations

from typing import Any

from ...cppstd.references import IGNORE as IGNORE
from ...cppstd.references import Tie as Tie
from ...cppstd.references import _vector_assigned as _vector_assigned
from ...cppstd.references import assign_call as assign_call
from ...cppstd.references import assign_into as assign_into
from ...cppstd.references import assign_vector as assign_vector
from ...cppstd.references import increment_reference as increment_reference
from ...cppstd.references import store_reference as store_reference
from ...cppstd.references import store_through as store_through
from ...cppstd.references import tie as tie

__all__ = ["assign_call", "assign_method", "assign_into", "store_through", "assign_vector",
           "store_reference", "increment_reference", "Tie", "tie", "IGNORE"]


def assign_method(obj: Any, name: str, args: tuple[Any, ...], value: Any) -> Any:
    """``obj.name(*args) = value``: an assignment into the reference the method returns."""
    return assign_into(getattr(obj, name)(*args), value)
