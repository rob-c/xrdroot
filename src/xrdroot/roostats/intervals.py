"""``ConfInterval`` and ``SimpleInterval``: what every RooStats interval answers.

An interval knows the parameters it was made for, its confidence level, and
whether a point of those parameters is in it; a simple one is ``[a, b]`` in
one parameter. :class:`Named` is the ``TNamed`` every RooStats result is.
"""

from __future__ import annotations

from typing import Any

from ..roofit.collections import RooArgSet, as_list
from ..roofit.messages import ERROR, log

__all__ = ["ConfInterval", "Named", "SimpleInterval", "same_parameters"]


class Named:
    """``TNamed``: a name, a title, and the class name ROOT gives - ``RooStats::...``."""

    def __init__(self, name: Any = "", title: Any = None) -> None:
        self._name = "" if name is None else str(name)
        self._title = self._name if title is None else str(title)

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        return self._title

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def SetTitle(self, title: str) -> None:
        self._title = str(title)

    def ClassName(self) -> str:
        return f"RooStats::{type(self).__name__}"

    def InheritsFrom(self, name: Any) -> bool:
        wanted = str(name).replace("RooStats::", "")
        return any(klass.__name__ == wanted for klass in type(self).__mro__)


def same_parameters(point: Any, parameters: Any) -> bool:
    """``CheckParameters``: whether ``point`` names the interval's parameters, said if not."""
    given, own = RooArgSet(as_list(point)), RooArgSet(as_list(parameters))
    if len(given) != len(own):
        log(None, ERROR, "InputArguments", "size is wrong, parameters don't match")
        return False
    if sorted(given.names()) != sorted(own.names()):
        log(None, ERROR, "InputArguments", "size is ok, but parameters don't match")
        return False
    return True


class ConfInterval(Named):
    """A confidence (or credible) interval: its parameters, level, and membership."""

    def __init__(self, name: Any = "") -> None:
        super().__init__(name)
        self._parameters = RooArgSet()
        self._cl = 0.95

    def SetConfidenceLevel(self, cl: float) -> None:
        self._cl = float(cl)

    def ConfidenceLevel(self) -> float:
        return self._cl

    def GetParameters(self) -> RooArgSet:
        return RooArgSet(list(self._parameters))

    def CheckParameters(self, point: Any) -> bool:
        return same_parameters(point, self._parameters)


class SimpleInterval(ConfInterval):
    """``[lower, upper]`` in one parameter."""

    def __init__(
        self, name: Any = "", var: Any = None, lower: float = 0.0, upper: float = 0.0,
        cl: float = 0.0,
    ) -> None:  # fmt: skip
        super().__init__(name)
        self._parameters = RooArgSet([var] if var is not None else [])
        self._lower, self._upper, self._cl = float(lower), float(upper), float(cl)

    def IsInInterval(self, point: Any) -> bool:
        if not self.CheckParameters(point) or len(RooArgSet(as_list(point))) != 1:
            return False
        value = as_list(point)[0].getVal()
        return bool(self._lower <= value <= self._upper)

    def LowerLimit(self, *args: Any) -> float:
        return self._lower

    def UpperLimit(self, *args: Any) -> float:
        return self._upper
