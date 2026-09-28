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


class StringVector(list):  # type: ignore[type-arg]
    """``std::vector<TString>``, as ``SplitString`` hands one back: a list with ``size()``."""

    def size(self) -> int:
        return len(self)

    def at(self, index: int) -> Any:
        return self[index]

    def push_back(self, value: Any) -> None:
        self.append(value)


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
