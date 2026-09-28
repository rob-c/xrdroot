"""``TMVA::Tools`` and ``TMVA::gConfig()``: the handful of TMVA's utilities macros call.

``TMVA::Tools::Instance()`` is how every TMVA macro starts - it loads the
library in ROOT, and here it is the one :class:`Tools` - and
``gTools().SplitString`` is what the tutorials split their method lists
with.
"""

from __future__ import annotations

from typing import Any

from .log import CONFIG, Config

__all__ = ["Tools", "gConfig", "gTools"]


class CxxVector(list):  # type: ignore[type-arg]
    """A ``std::vector`` TMVA hands back: a list, with ``size()``, ``at()`` and iterators."""

    def size(self) -> int:
        return len(self)

    def at(self, index: int) -> Any:
        return self[index]

    def push_back(self, value: Any) -> None:
        self.append(value)

    def empty(self) -> bool:
        return not self

    def begin(self) -> Iterator:
        return Iterator(self, 0)

    def end(self) -> Iterator:
        return Iterator(self, len(self))


class Iterator:
    """An iterator into a :class:`CxxVector`: ``*it`` is the iterator itself, as a macro reads it."""

    def __init__(self, owner: list[Any], position: int) -> None:
        self.owner, self.position = owner, position

    def __deref__(self) -> Any:
        return self.owner[self.position]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__deref__(), name)

    def __float__(self) -> float:
        return float(self.__deref__())

    def __iadd__(self, step: int) -> Iterator:
        self.position += int(step)
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Iterator):
            return NotImplemented
        return self.owner is other.owner and self.position == other.position

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __lt__(self, other: Iterator) -> bool:
        return self.position < other.position

    __hash__ = None  # type: ignore[assignment]


#: ``std::vector<TString>``, as ``SplitString`` hands one back.
StringVector = CxxVector


class Tools:
    """``TMVA::Tools``: the one instance, and the utilities on it."""

    _instance: Tools | None = None

    @classmethod
    def Instance(cls) -> Tools:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def SplitString(text: Any, separator: Any) -> StringVector:
        """``SplitString(s, ',')``: the pieces between each separator, empty ones dropped."""
        sep = chr(separator) if isinstance(separator, int) else str(separator)
        return StringVector(piece for piece in str(text).split(sep) if piece)

    @staticmethod
    def Color(name: Any) -> str:
        from .log import color

        return color(str(name))


def gTools() -> Tools:
    """``TMVA::gTools()``."""
    return Tools.Instance()


def gConfig() -> Config:
    """``TMVA::gConfig()``: the colour, silence and progress bar every TMVA class shares."""
    return CONFIG
