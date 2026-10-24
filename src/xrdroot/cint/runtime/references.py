"""ROOT-style setters complement the generic C++ reference and assignment contracts."""

from __future__ import annotations

from typing import Any

from ...cppstd.references import _vector_assigned as _vector_assigned
from ...cppstd.references import assign_call as assign_call
from ...cppstd.references import assign_into as assign_into
from ...cppstd.references import assign_vector as assign_vector
from ...cppstd.references import increment_reference as increment_reference
from ...cppstd.references import store_reference as store_reference
from ...cppstd.references import store_through as store_through

__all__ = ["assign_call", "assign_method", "assign_into", "store_through", "assign_vector",
           "store_reference", "increment_reference"]


def assign_method(obj: Any, name: str, args: tuple[Any, ...], value: Any) -> Any:
    """``obj.name(*args) = value``: ROOT's ``Set<name>``, or an assignment into the result."""
    setter = getattr(obj, f"Set{name}", None)
    if not args and callable(setter):
        setter(value)
        return value
    return assign_into(getattr(obj, name)(*args), value)
