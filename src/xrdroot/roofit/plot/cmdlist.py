"""The list of options a ``plotOn`` passes down, as RooFit passes it: one list, added to on the way.

RooFit's ``RooAbsPdf::plotOn`` and ``RooAbsReal::plotOn`` hand one
``RooLinkedList`` of options down to each other - and, for an error band,
to themselves again - each adding options for the duration of its call
(the fit range, the scale, a curve-name suffix) and each warning of the
options it then finds twice. What a tutorial prints while plotting follows
from that list, so :class:`CmdList` is it: shared, added to for a scope,
read as ``RooCmdConfig`` reads it - the last of a name wins.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..cmdargs import Commands, RooCmdArg, commands
from ..messages import WARNING, log

__all__ = ["CmdList", "canonical"]


def canonical(arg: RooCmdArg) -> RooCmdArg:
    """A command under the opcode ROOT's function gives it: ``Range("a")`` is ``RangeWithName``."""
    if arg.name == "Range" and arg.args and isinstance(arg.args[0], str):
        return RooCmdArg("RangeWithName", *arg.args)
    if arg.name == "Components":
        kind = "SelectCompSpec" if isinstance(arg.value(0), str) else "SelectCompSet"
        return RooCmdArg(kind, *arg.args)
    return arg


class CmdList:
    """Commands in order, some of them only for the call that added them."""

    def __init__(self, given: Any = ()) -> None:
        self.items: list[RooCmdArg] = list(given)

    @classmethod
    def of(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> CmdList:
        return cls(canonical(one) for one in commands(args, kwargs).given)

    def copy(self) -> CmdList:
        return CmdList(self.items)

    def find(self, name: str) -> RooCmdArg | None:
        return next((one for one in self.items if one.name == name), None)

    def has(self, name: str) -> bool:
        return self.find(name) is not None

    def options(self) -> Commands:
        """The commands as ``RooCmdConfig`` reads them: by name, the last of each."""
        return Commands(self.items)

    def process(self, context: str) -> Commands:
        """``RooCmdConfig::process``: a warning for each command given twice, then the options."""
        seen: set[str] = set()
        for one in self.items:
            if one.name in seen:
                log(
                    None,
                    WARNING,
                    "InputArguments",
                    f"{context} WARNING: argument {one.name} is duplicated",
                )
            seen.add(one.name)
        return self.options()

    def strip(self, *names: str) -> None:
        """``stripCmdList``: take the commands of these names out for good."""
        self.items = [one for one in self.items if one.name not in names]

    def strip_one(self, arg: RooCmdArg) -> None:
        self.items = [one for one in self.items if one is not arg]

    def replace_or_add(self, arg: RooCmdArg) -> RooCmdArg | None:
        """``replaceOrAdd``: put ``arg`` where the first of its name is, else at the end."""
        for index, one in enumerate(self.items):
            if one.name == arg.name:
                self.items[index] = arg
                return one
        self.items.append(arg)
        return None

    @contextmanager
    def adding(self, *args: RooCmdArg) -> Iterator[None]:
        """These commands in the list for the duration of a ``with``, as a call's own locals are."""
        self.items.extend(args)
        try:
            yield
        finally:
            self.items[:] = [item for item in self.items if all(item is not one for one in args)]
