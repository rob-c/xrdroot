"""``RooCmdArg``: the named options every RooFit call takes, and PyROOT's keywords for them.

``pdf.fitTo(data, Save(), PrintLevel(-1))`` passes its options as a list of
command arguments, each a name and the values it was made with, and each
method picks out the ones it understands. PyROOT lets the same be said as
``pdf.fitTo(data, Save=True, PrintLevel=-1)``: a keyword is the command of
its name, made from its value - a tuple's elements as its arguments, and
``True`` for a command that takes none. :func:`commands` turns either form
into one :class:`Commands`, which is what the methods here read.

A command made with fewer arguments than ROOT's function has is given
ROOT's defaults for the rest (:data:`DEFAULTS`), so ``Save()`` is
``Save(true)`` and ``AutoBinning()`` is ``AutoBinning(100, 0.1)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = ["DEFAULTS", "FLAGS", "Commands", "RooCmdArg", "RooLinkedList", "commands", "make"]

#: The default arguments of RooFit's command functions, from ``RooGlobalFunc.h``.
DEFAULTS: dict[str, tuple[Any, ...]] = {
    "Save": (True,), "Extended": (True,), "Verbose": (True,), "Timer": (True,),
    "Warnings": (True,), "InitialHesse": (True,), "Hesse": (True,), "Minos": (True,),
    "SplitRange": (True,), "ShowConstants": (True,), "AutoBinned": (True,),
    "ExpectedData": (True,), "Asimov": (True,), "IntrinsicBinning": (True,),
    "Silence": (True,), "Binned": (True,), "FitGauss": (True,), "TLatexStyle": (True,),
    "LatexStyle": (True,), "LatexTableStyle": (True,), "VerbatimName": (True,),
    "RecycleConflictNodes": (True,), "Embedded": (True,), "NoRecursion": (True,),
    "Invisible": (True,), "Offset": (True,), "Optimize": (2,), "AutoPrecision": (2,),
    "FixedPrecision": (2,), "AutoBinning": (100, 0.1), "AutoSymBinning": (100, 0.1),
    "ModularL": (False,), "ShowProgress": (), "MoveToBack": (), "ShiftToZero": (),
    "VLines": (), "Prefix": (True,), "ShowValue": (True,), "ShowError": (True,),
    "ShowName": (True,), "ShowUnit": (True,), "ShowAsymError": (True,),
}  # fmt: skip
#: Commands whose keyword form takes only ``True`` (made) or ``False`` (left out).
FLAGS = frozenset({"MoveToBack", "ShiftToZero", "VLines", "ShowProgress"})


#: ``TColorNumber(std::string)``: matplotlib's colour letters, and the two enumerated names.
COLOUR_WORDS = {"r": 632, "b": 600, "g": 416, "y": 400, "w": 0, "k": 1, "m": 616, "c": 432,
                "kWhite": 0, "kBlack": 1}  # fmt: skip
#: ``interpretLineStyleString``.
LINE_STYLES = {"-": 1, "--": 2, ":": 3, "-.": 4}
#: ``RooAbsData::errorTypeFromString``.
ERROR_TYPES = {"Poisson": 0, "SumW2": 1, "None": 2, "Expected": 3, "Auto": 4}
#: The commands whose first argument ROOT also takes as a string, and how it reads it.
STRING_FORMS: dict[str, dict[str, int]] = {
    "LineColor": COLOUR_WORDS, "FillColor": COLOUR_WORDS, "MarkerColor": COLOUR_WORDS,
    "Color": COLOUR_WORDS, "LineStyle": LINE_STYLES, "FillStyle": {}, "MarkerStyle": {},
    "DataError": ERROR_TYPES,
}  # fmt: skip


def _read_string(name: str, value: str) -> int:
    """A string where ROOT's command takes a number: a colour, a style, an error type."""
    table = STRING_FORMS[name]
    if value in table:
        return table[value]
    from .names import named_constant

    return named_constant(value)


class RooCmdArg:
    """One named option and the values it was made with: ``RooFit::Save(true)``."""

    def __init__(self, name: str = "", *args: Any) -> None:
        self.name = str(name)
        self.args = tuple(args) if args else DEFAULTS.get(self.name, ())
        if self.name in STRING_FORMS and self.args and isinstance(self.args[0], str):
            self.args = (_read_string(self.name, self.args[0]), *self.args[1:])
        if self.name == "DataError" and self.args and self.args[0] is None:
            self.args = (ERROR_TYPES["None"], *self.args[1:])  # PyROOT's DataError=None

    def value(self, index: int = 0, default: Any = None) -> Any:
        """Its ``index``-th value, or ``default`` if it was made with fewer."""
        return self.args[index] if index < len(self.args) else default

    def GetName(self) -> str:
        return self.name

    @staticmethod
    def none() -> RooCmdArg:
        """``RooCmdArg::none()``: the option that says nothing."""
        return RooCmdArg()

    def __repr__(self) -> str:
        return f"RooFit::{self.name}{self.args!r}"


class RooLinkedList(list):  # type: ignore[type-arg]
    """``RooLinkedList``: a list of command arguments, as ``fitTo(data, list)`` takes one."""

    def Add(self, obj: Any) -> None:
        self.append(obj)

    def GetSize(self) -> int:
        return len(self)


def make(name: str, value: Any) -> RooCmdArg:
    """The command a PyROOT keyword stands for: ``Save=True`` is ``Save(True)``."""
    if name in FLAGS:
        return RooCmdArg(name) if value else RooCmdArg()
    if isinstance(value, (tuple, list)):
        return RooCmdArg(name, *value)
    if isinstance(value, dict):  # YVar=dict(var=y, Binning=50): the command made from keywords
        given = dict(value)
        first = [given.pop(key) for key in ("var", "what") if key in given]
        return RooCmdArg(name, *first, *(make(key, one) for key, one in given.items()))
    return RooCmdArg(name, value)


class Commands:
    """The options a call was given, by name: the last of each name, as ``RooCmdConfig`` takes it."""

    def __init__(self, given: Iterable[RooCmdArg]) -> None:
        self.given = [one for one in given if one.name]
        self._first: dict[str, RooCmdArg] = {}
        for one in self.given:
            self._first[one.name] = one

    def warn_duplicates(self, context: str) -> None:
        """``RooCmdConfig::process``'s warning for each option given more than once."""
        from .messages import WARNING, log

        seen: set[str] = set()
        for one in self.given:
            if one.name in seen:
                log(None, WARNING, "InputArguments",
                    f"{context} WARNING: argument {one.name} is duplicated")  # fmt: skip
            seen.add(one.name)

    def __contains__(self, name: str) -> bool:
        return name in self._first

    def get(self, name: str, index: int = 0, default: Any = None) -> Any:
        """The ``index``-th value of the option ``name``, or ``default`` if it was not given."""
        found = self._first.get(name)
        return default if found is None else found.value(index, default)

    def args(self, name: str) -> tuple[Any, ...]:
        found = self._first.get(name)
        return () if found is None else found.args

    def every(self, name: str) -> list[RooCmdArg]:
        """Each option called ``name``, in the order given: ``Slice`` and ``Cut`` can repeat."""
        return [one for one in self.given if one.name == name]

    def names(self) -> list[str]:
        return list(self._first)


def commands(args: Iterable[Any], kwargs: dict[str, Any] | None = None) -> Commands:
    """The options in ``args`` - command arguments or lists of them - and ``kwargs``, as one."""
    found: list[RooCmdArg] = []
    for arg in args:
        if isinstance(arg, RooCmdArg):
            found.append(arg)
        elif isinstance(arg, (list, tuple)) and all(isinstance(one, RooCmdArg) for one in arg):
            found.extend(arg)
        else:
            raise TypeError(f"{arg!r} is not a RooFit command argument, such as RooFit.Save()")
    found.extend(make(name, value) for name, value in (kwargs or {}).items())
    return Commands(found)
