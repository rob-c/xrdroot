"""TMVA's option strings: ``"!H:V:NTrees=850:MaxDepth=3:NSmoothSig[0]=20"``.

Every TMVA class is configured by one string of ``:``-separated settings, as
``Configurable::ParseOptions`` reads it: ``Name=value`` sets an option,
``Name`` alone switches a flag on and ``!Name`` off, and ``Name[i]=value``
sets one element of an array option - or, with no index, all of them.
Names are matched without regard to case, as TMVA matches them.

:class:`Options` holds what a string said; the class reading it asks for
each option it declares, with the default it has, and so an option a string
leaves out is the class's default exactly as in TMVA.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Options", "truth"]

#: The words TMVA's ``Option<bool>`` reads as true; anything else is false.
TRUE_WORDS = frozenset({"1", "t", "true", "ktrue", "y", "yes"})


def truth(text: Any) -> bool:
    """A flag's value as TMVA's ``Option<bool>::SetValue`` reads it."""
    if isinstance(text, bool):
        return text
    return str(text).strip().lower() in TRUE_WORDS


def _split(token: str) -> tuple[str, str | None, int | None]:
    """One setting as its lower-cased name, its value (``None`` for a bare flag) and its index."""
    name, equals, value = token.partition("=")
    index = None
    if "[" in name and name.endswith("]"):
        name, _, number = name[:-1].partition("[")
        index = int(number)
    return name.strip().lower(), (value.strip() if equals else None), index


class Options:
    """What an option string said, asked for by name with the asker's default.

    >>> o = Options("!H:V:NTrees=850:NSmoothSig[0]=20")
    >>> o.flag("H", True), o.flag("V", False), o.integer("NTrees", 800)
    (False, True, 850)
    >>> o.array("NSmoothSig", 2, 5)
    [20, 5]
    """

    def __init__(self, text: Any = "") -> None:
        self.text = str(text)
        self._values: dict[str, Any] = {}
        self._arrays: dict[str, dict[int, str]] = {}
        #: Every option given twice, and the value it had before, as ``ParseOptions`` warns.
        self.repeated: list[tuple[str, Any]] = []
        for token in self.text.split(":"):
            self._take(token.strip().lstrip("~"))

    def _take(self, token: str) -> None:
        if not token:
            return
        if "=" not in token:
            negated = token.startswith("!")
            self._values[token.lstrip("!").lower()] = not negated
            return
        name, value, index = _split(token)
        if index is None:
            if name in self._values:
                self.repeated.append((name, self._values[name]))
            self._values[name] = value
        else:
            self._arrays.setdefault(name, {})[index] = str(value)

    def given(self, name: str) -> bool:
        """Did the string say anything about ``name``?"""
        key = name.lower()
        return key in self._values or key in self._arrays

    def text_of(self, name: str, default: str = "") -> str:
        """A string option, as it was written."""
        value = self._values.get(name.lower())
        return default if value is None else str(value)

    def flag(self, name: str, default: bool = False) -> bool:
        """A boolean option: ``Name``, ``!Name`` or ``Name=True``."""
        value = self._values.get(name.lower())
        return default if value is None else truth(value)

    def number(self, name: str, default: float = 0.0) -> float:
        """A floating-point option."""
        value = self._values.get(name.lower())
        return default if value is None else float(str(value).rstrip("%"))

    def integer(self, name: str, default: int = 0) -> int:
        """An integer option; ``1e3`` and ``2.0`` are read as C++'s stream reads them, whole."""
        value = self._values.get(name.lower())
        return default if value is None else int(float(str(value)))

    def array(self, name: str, size: int, default: Any) -> list[Any]:
        """An array option: the whole-array value or ``default``, then each index given."""
        whole = self._values.get(name.lower())
        base = default if whole is None else _like(default, whole)
        values = [base] * size
        for index, value in self._arrays.get(name.lower(), {}).items():
            if 0 <= index < size:
                values[index] = _like(default, value)
        return values

    def __repr__(self) -> str:
        return f"Options({self.text!r})"


def _like(default: Any, value: Any) -> Any:
    """``value`` read as the type of ``default``."""
    if isinstance(default, bool):
        return truth(value)
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    return str(value)
