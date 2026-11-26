"""The line cling prints for the value of an expression typed without its ``;``.

``(int) 4``, ``(double) 3.1400000``, ``(const char *) "abc"``, ``(bool)
true``, ``(std::string) "x"``, ``(TH1F *) 0x7f8e...`` for a pointer, and for
an object its type as a reference and what cling's ``printValue`` says of
it: ``(TH1F &) Name: h Title: h NbinsX: 10``, ``(TNamed &) Name: a Title:
b``, ``(TString &) "123ABC"[6]``, ``(std::vector<int> &) { 1, 2, 3 }`` and,
for a class it has nothing to say of, ``(Foo &) @0x7ffd...``. These are
ROOT 6's, as roottest's ``cling/other/execValuePrint`` records them.

The type is the translator's where it knows it - an ``int`` and a ``long
long`` are one Python ``int`` - and otherwise read off the Python value: a
number with a point is a ``double``, a ROOT object a pointer to its class,
which is what a call into ROOT returning one hands back.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["shown", "spelled_for"]

#: The integral types cling prints as plain numbers.
INTEGERS = frozenset(
    {"short", "int", "long", "unsigned", "signed", "wchar_t", "char16_t", "char32_t",
     "unsigned short", "unsigned int", "unsigned long", "long long", "unsigned long long"}
)  # fmt: skip

#: How a control character - or a quote, or a backslash - is spelled inside quotes.
CHARACTER_ESCAPES = {
    0x00: "\\0", 0x07: "\\a", 0x08: "\\b", 0x09: "\\t", 0x0A: "\\n", 0x0B: "\\v",
    0x0C: "\\f", 0x0D: "\\r", 0x27: "\\'", 0x5C: "\\\\",
}  # fmt: skip

#: The character types, whose values cling prints quoted.
CHARACTERS = frozenset({"char", "signed char", "unsigned char"})

#: The string classes, printed quoted as cling prints each.
STRINGS = frozenset({"std::string", "string", "TString"})


def _integer(value: Any) -> str:
    return str(int(value))


def _character(value: Any) -> str:
    """A ``char`` as object-like text: quoted when it is a character, its number when it is not.

    A byte outside printable ASCII - ``-128``, say - has no character to quote, and a line that
    is written back has to read as the value it was: ``char a = -128;`` is what a compiler takes,
    where ``'\\x80'`` reads as +128 in a signed char. The printable range keeps its quotes, so
    what cling echoes for an ordinary character is what this writes."""
    if isinstance(value, str):
        return "'" + value + "'"
    code = int(value)
    if code in CHARACTER_ESCAPES:  # a control character has its own spelling
        return "'" + CHARACTER_ESCAPES[code] + "'"
    printable = 0x20 <= code <= 0x7E and code not in (0x27, 0x5C)
    return "'" + chr(code) + "'" if printable else str(code)


def _bool(value: Any) -> str:
    return "true" if value else "false"


def _float(value: Any) -> str:
    return f"{float(value):#.6g}f"


def _double(value: Any) -> str:
    return f"{float(value):#.8g}"


def _string(value: Any) -> str:
    return f'"{value}"'


def _tstring(value: Any) -> str:
    text = str(value)
    return f'"{text}"[{len(text)}]'


#: How cling writes a value of each type it prints as a plain value.
FORMATS: dict[str, Callable[[Any], str]] = {
    **dict.fromkeys(INTEGERS, _integer),
    **dict.fromkeys(CHARACTERS, _character),
    "bool": _bool,
    "float": _float,
    "double": _double,
    "long double": _double,
    "std::string": _string,
    "string": _string,
    "TString": _tstring,
}


def _address(value: Any) -> str:
    return f"0x{id(value):x}"


def _pointer(value: Any, spelled: str) -> str:
    if value is None:
        return "nullptr"
    if isinstance(value, str) and spelled.replace("const ", "").startswith("char"):
        return _string(value)
    return _address(value)


def _element(value: Any) -> str:
    """One element of a container, as cling prints it inside ``{ ... }``."""
    if isinstance(value, str):
        return _string(value)
    write = FORMATS.get(spelled_for(value) or "")
    if write is not None:
        return write(value)
    return "nullptr" if value is None else _address(value)


def _histogram(value: Any) -> str:
    return f"Name: {value.GetName()} Title: {value.GetTitle()} NbinsX: {value.GetNbinsX()}"


def _named(value: Any) -> str:
    return f"Name: {value.GetName()} Title: {value.GetTitle()}"


def _braced(items: list[str]) -> str:
    return "{ " + ", ".join(items) + " }" if items else "{}"


def _mapping(value: Any) -> str:
    return _braced([f"{_element(key)} => {_element(item)}" for key, item in value.items()])


def _sequence(value: Any) -> str:
    return _braced([_element(item) for item in value])


def _printers() -> list[tuple[Callable[[Any], bool], Callable[[Any], str]]]:
    """Each kind of object cling's ``printValue`` knows, in the order it asks, with what
    it says of one: strings, ROOT's histograms and named objects, then containers."""
    from ... import pyroot

    return [
        (lambda value: isinstance(value, str), _string),
        (lambda value: isinstance(value, pyroot.TString), _tstring),
        (lambda value: isinstance(value, pyroot.TH1), _histogram),
        (lambda value: isinstance(value, pyroot.TObject), _named),
        (lambda value: hasattr(value, "items") and hasattr(value, "__len__"), _mapping),
        (lambda value: hasattr(value, "__len__") and hasattr(value, "__iter__"), _sequence),
    ]


def _object(value: Any) -> str:
    """What cling's ``printValue`` says of an object: ROOT's, a container's, or its address."""
    for applies, write in _printers():
        if applies(value):
            return write(value)
    return "@" + _address(value)


def spelled_for(value: Any) -> str | None:
    """The type cling would name for a value whose type the translator does not know."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, numbers.Integral):
        return "int"
    if isinstance(value, numbers.Real):
        return "double"
    if isinstance(value, str):
        return "const char *"
    if type(value).__name__ == "TString":
        return "TString"
    return f"{type(value).__name__} *"


def shown(value: Any, spelled: str | None, lvalue: bool = False) -> str | None:
    """The line cling prints for ``value`` of the type ``spelled`` (``None``: unknown).

    An object held by value is printed as a reference to it when the
    expression named one - ``h``, ``*p`` - as cling does; nothing is printed
    for a ``void`` call, which is a value of no known type that is ``None``.
    """
    spelled = spelled or spelled_for(value)
    if spelled is None or spelled == "void":
        return None
    if spelled.endswith("*"):
        return f"({spelled}) {_pointer(value, spelled)}"
    write = FORMATS.get(spelled)
    if write is not None and not (lvalue and spelled in STRINGS):
        return f"({spelled}) {write(value)}"
    return f"({spelled}{' &' if lvalue else ''}) {_object(value)}"
