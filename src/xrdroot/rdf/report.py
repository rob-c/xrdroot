"""What ``Report`` and ``Display`` give back: a cut flow, and a table of the first entries.

A cut flow is ROOT's ``RCutFlowReport``: for every named filter, how many
entries reached it, how many it passed, and the fraction of all entries that
got that far, printed exactly as ROOT prints one. A display is ROOT's
``RDisplay``: a box of the first few entries, a collection's elements one to
a line down its cell.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

__all__ = ["CutInfo", "CutFlowReport", "Display"]


class CutInfo:
    """One named filter's line of a cut flow."""

    __slots__ = ("name", "passed", "all")

    def __init__(self, name: str, passed: int, all: int) -> None:
        self.name = name
        self.passed = passed
        self.all = all

    def __repr__(self) -> str:
        return f"<CutInfo {self.name!r}: {self.passed} of {self.all}>"

    @property
    def efficiency(self) -> float:
        """The percentage of the entries that reached the filter that it passed."""
        return 100.0 * self.passed / self.all if self.all else float("nan")

    def GetName(self) -> str:
        return self.name

    def GetPass(self) -> int:
        return self.passed

    def GetAll(self) -> int:
        return self.all

    def GetEff(self) -> float:
        return self.efficiency


def _percent(value: float) -> str:
    """``%3.2f`` as C's printf writes it, ``nan`` included."""
    return "nan" if value != value else f"{value:3.2f}"


class CutFlowReport:
    """``RCutFlowReport``: every named filter's counts, in the order the entries meet them.

    >>> print(df.Report().GetValue())                  # doctest: +SKIP
    pt_cut    : pass=39         all=100        -- eff=39.00 % cumulative eff=39.00 %
    """

    def __init__(self, cuts: Sequence[CutInfo]) -> None:
        self.cuts = list(cuts)

    def __iter__(self) -> Iterator[CutInfo]:
        return iter(self.cuts)

    def __len__(self) -> int:
        return len(self.cuts)

    def __getitem__(self, key: int | str) -> CutInfo:
        if isinstance(key, str):
            return self.At(key)
        return self.cuts[key]

    def At(self, name: str) -> CutInfo:
        """The line of the filter called ``name``."""
        for cut in self.cuts:
            if cut.name == name:
                return cut
        raise KeyError(
            f"no filter called {name!r} is in this report; there is "
            + (", ".join(repr(cut.name) for cut in self.cuts) or "none")
        )

    def __str__(self) -> str:
        first = self.cuts[0].all if self.cuts else 0
        lines = []
        for cut in self.cuts:
            cumulative = 100.0 * cut.passed / first if first else float("nan")
            lines.append(
                f"{cut.name:<10}: pass={cut.passed:<10} all={cut.all:<10} -- "
                f"eff={_percent(cut.efficiency)} % cumulative eff={_percent(cumulative)} %"
            )
        return "\n".join(lines)

    def Print(self) -> None:
        print(self)

    at = At
    print = Print


def _at(values: Any, row: int) -> Any:
    """One entry of a column; of what Combinations gives, each member's row, as a list."""
    if isinstance(values, tuple):
        return tuple(member[row].tolist() for member in values)
    return values[row]


def _cell(value: Any, limit: int) -> list[str]:
    """A value as the lines of its cell: one for a number, one per element of a collection."""
    if isinstance(value, (np.ndarray, list, tuple)) and not isinstance(value, str):
        items = [str(item) for item in list(value)[:limit]]
        if len(value) > limit:
            items.append("...")
        return items or [""]
    return [str(value)]


class Display:
    """``RDisplay``: the first entries of some columns, as a box of text.

    >>> print(df.Display(["x", "pt"], 3).GetValue())    # doctest: +SKIP
    """

    def __init__(self, columns: dict[str, Any], rows: int, elements: int, more: bool) -> None:
        self.columns = columns
        self.rows = rows
        self.elements = elements
        self.more = more

    def _cells(self) -> list[list[list[str]]]:
        return [
            [[str(row)]]
            + [_cell(_at(values, row), self.elements) for values in self.columns.values()]
            for row in range(self.rows)
        ]

    def AsString(self) -> str:
        """The box, as ROOT's ``RDisplay::AsString`` draws it."""
        header = ["Row", *self.columns]
        table = self._cells()
        widths = _widths(header, table)
        rule = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
        lines = [rule, _line(header, widths), rule]
        for cells in table:
            lines += _block(cells, widths)
            lines.append(rule)
        if self.more:
            lines.append("...")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.AsString()

    def Print(self) -> None:
        print(self)

    as_string = AsString
    print = Print


def _widths(header: list[str], table: list[list[list[str]]]) -> list[int]:
    """How wide each column of the box is: its widest line, heading included."""
    widths = [len(text) for text in header]
    for cells in table:
        widths = [max(width, *map(len, cell)) for width, cell in zip(widths, cells)]
    return widths


def _block(cells: list[list[str]], widths: list[int]) -> list[str]:
    """One entry's lines: as many as its longest collection has elements shown."""
    depth = max(len(cell) for cell in cells)
    padded = [cell + [""] * (depth - len(cell)) for cell in cells]
    return [_line([cell[at] for cell in padded], widths) for at in range(depth)]


def _line(texts: Sequence[str], widths: Sequence[int]) -> str:
    return "|" + "|".join(f" {text:<{width}} " for text, width in zip(texts, widths)) + "|"
