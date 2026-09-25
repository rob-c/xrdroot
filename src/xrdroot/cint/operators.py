"""C++'s overloadable operators as the methods Python calls for them.

A class's ``operator+`` is Python's ``__add__``, its ``operator()`` is
``__call__``, its ``operator double`` is ``__float__``. The same names are
used where a macro calls an operator by name, ``obj.operator()(1)`` or
``TVector3::operator+(a, b)``.
"""

from __future__ import annotations

__all__ = ["DUNDERS", "python_operator"]

#: Each operator as the method Python calls for it: with an argument, and without one.
DUNDERS = {
    "operator+": ("__add__", "__pos__"),
    "operator-": ("__sub__", "__neg__"),
    "operator*": ("__mul__", None),
    "operator/": ("__truediv__", None),
    "operator%": ("__mod__", None),
    "operator==": ("__eq__", None),
    "operator!=": ("__ne__", None),
    "operator<": ("__lt__", None),
    "operator>": ("__gt__", None),
    "operator<=": ("__le__", None),
    "operator>=": ("__ge__", None),
    "operator[]": ("__getitem__", None),
    "operator()": ("__call__", "__call__"),
    "operator+=": ("__iadd__", None),
    "operator-=": ("__isub__", None),
    "operator*=": ("__imul__", None),
    "operator/=": ("__itruediv__", None),
    "operator&": ("__and__", None),
    "operator|": ("__or__", None),
    "operator^": ("__xor__", None),
    "operator~": (None, "__invert__"),
    "operator<<": ("__lshift__", None),
    "operator>>": ("__rshift__", None),
    "operator=": ("_assign", None),
    "operator bool": (None, "__bool__"),
    "operator double": (None, "__float__"),
    "operator float": (None, "__float__"),
    "operator int": (None, "__int__"),
}


def python_operator(name: str) -> str:
    """The Python method for the C++ member ``name``: a dunder for an operator, else ``name``."""
    names = DUNDERS.get(name)
    if names is None:
        return name
    binary, unary = names
    return binary or unary or name
