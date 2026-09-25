"""``printf``'s formats, as ``Form`` and ``TString::Format`` read them.

Python's ``%`` operator is C's ``printf`` for every conversion ROOT's
tutorials use, less the length modifiers C needs and Python does not: ``%ld``,
``%lld``, ``%lu``, ``%lf`` and ``%zu`` are ``%d``, ``%u`` and ``%f`` once the
``l``, ``ll`` or ``z`` is dropped. A ``TString`` argument is its text, and a
``bool`` is the ``int`` C would have promoted it to.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["Form", "printf", "c_format"]

#: One conversion: flags, width, precision, then the length modifiers dropped.
CONVERSION = re.compile(
    r"%([-+ #0]*(?:\*|\d+)?(?:\.(?:\*|\d+))?)(?:hh|h|ll|l|L|q|j|z|t)?([a-zA-Z%])"
)


def _converted(found: re.Match[str]) -> str:
    """One conversion with its length modifier dropped, and ``%p`` as hexadecimal."""
    spec, letter = found.group(1), found.group(2)
    return f"%{spec}{'#x' if letter == 'p' else letter}"


def _argument(value: Any) -> Any:
    """What C would pass for a Python value: a boolean as an int, a string as its text."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return str(value)
    return value


def c_format(fmt: str, *args: Any) -> str:
    """``fmt`` filled from ``args`` as ``printf`` fills it."""
    text = CONVERSION.sub(_converted, str(fmt))
    values = tuple(_argument(value) for value in args)
    return text % values


def Form(fmt: str, *args: Any) -> str:
    """``Form``: the text ``printf`` would print, handed back rather than printed."""
    return c_format(fmt, *args)


def printf(fmt: str, *args: Any) -> None:
    """``printf``: print it, without a newline unless the format has one."""
    print(c_format(fmt, *args), end="")
