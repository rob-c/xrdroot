"""ROOT diagnostics retain their adapters over the shared C formatting implementation."""

from __future__ import annotations

import sys
from typing import Any

from ...cppstd.printf import BITS as BITS
from ...cppstd.printf import CONVERTERS as CONVERTERS
from ...cppstd.printf import SPEC as SPEC
from ...cppstd.printf import _Arguments as _Arguments
from ...cppstd.printf import _character as _character
from ...cppstd.printf import _conversion as _conversion
from ...cppstd.printf import _decimal as _decimal
from ...cppstd.printf import _integer as _integer
from ...cppstd.printf import _pointer as _pointer
from ...cppstd.printf import _signed as _signed
from ...cppstd.printf import _spec as _spec
from ...cppstd.printf import _text as _text
from ...cppstd.printf import _unsigned as _unsigned
from ...cppstd.printf import _value as _value
from ...cppstd.printf import cformat as cformat
from ...cppstd.printf import fprintf as fprintf
from ...cppstd.printf import printf as printf
from ...cppstd.printf import putchar as putchar
from ...cppstd.printf import puts as puts
from ...cppstd.printf import sprintf as sprintf

__all__ = [
    "cformat",
    "printf",
    "Printf",
    "sprintf",
    "Form",
    "fprintf",
    "puts",
    "putchar",
    "Info",
    "Warning",
    "Error",
    "Fatal",
    "Printf_",
]


def Printf(fmt: Any, *args: Any) -> None:
    """ROOT's ``Printf``: ``printf`` with a newline after."""
    sys.stdout.write(cformat(fmt, *args) + "\n")


def Form(fmt: Any, *args: Any) -> str:
    """ROOT's ``Form``: ``sprintf`` into a buffer of ROOT's, handed back."""
    return cformat(fmt, *args)


def _message(level: str, location: Any, fmt: Any, args: tuple[Any, ...]) -> None:
    where = f" in <{location}>" if location else ""
    sys.stderr.write(f"{level}{where}: {cformat(fmt, *args)}\n")


def Info(location: Any, fmt: Any, *args: Any) -> None:
    """ROOT's ``Info(where, fmt, ...)``: ``Info in <where>: message`` on standard error."""
    _message("Info", location, fmt, args)


def Warning(location: Any, fmt: Any, *args: Any) -> None:
    _message("Warning", location, fmt, args)


def Error(location: Any, fmt: Any, *args: Any) -> None:
    _message("Error", location, fmt, args)


def Fatal(location: Any, fmt: Any, *args: Any) -> None:
    _message("Fatal", location, fmt, args)
    raise SystemExit(1)


#: ``Printf`` under a name no macro's own ``Printf`` method can hide.
Printf_ = Printf
