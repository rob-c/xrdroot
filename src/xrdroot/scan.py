"""``TTree::Scan``: a tree's values as a table of text, laid out exactly as ROOT lays it.

    >>> print(tree.scan("Int32:Float64", "Int32 < 3"))        # doctest: +SKIP
    ************************************
    *    Row   *     Int32 *   Float64 *
    ************************************
    *        0 *         0 *         0 *
    *        1 *         1 *         1 *
    *        2 *         2 *         2 *
    ************************************
    ==> 3 selected entries

Each column is an expression, as ``Draw``'s are, and each is printed the
way ``TTreeFormula::PrintValue`` prints it: a number through C's ``%9.9g``,
trimmed of digits before its exponent when that makes it too wide, and
anything too wide for its column cut off at the column's edge; a string as
it is. A column is as wide as its name, from nine characters to twenty.

An expression that loops over a collection prints a row per element under an
``Instance`` column. The columns go down together only when the selection
loops too - then all of them share its loop, as ``TTreeFormulaManager`` makes
them - and otherwise each goes as far as it has elements, the row count of an
entry being the most any of them has and the columns that run out left blank.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import IO, Any, NamedTuple

from .drawspec import split_names
from .formula import Formula, compile_formula
from .formula.select import TreeNames, _Read, _shaped, _underlying
from .formula.together import evaluate_together
from .tree import Jagged

__all__ = ["scan"]

Array = Any

#: ``TTreePlayer::Scan``'s width and precision of a column, unless told otherwise.
COLUMN = 9
#: The widest a column grows to fit its name.
WIDEST = 20
#: The most digits ``colsize=`` gives the precision.
MOST_PRECISE = 18
#: Entries read at a time.
STEP = 10_000
#: The columns an empty expression prints: ROOT's first eight leaves.
FIRST = 8


class Layout(NamedTuple):
    """The widths a table is printed with, and whether it has an ``Instance`` column."""

    names: list[str]
    sizes: list[int]
    column: int
    precision: int
    instances: bool

    def border(self) -> str:
        """A line of stars as wide as the table."""
        start = "*" * (22 if self.instances else 11)
        return start + "".join("*" * (size + 3) for size in self.sizes) + "*"

    def header(self) -> str:
        """The names of the columns, each cut to ``...`` if it is wider than its column."""
        cells = []
        for name, size in zip(self.names, self.sizes):
            if len(name) > size:
                name = name[: max(1, size - 3)] + "..."
            cells.append(f"* {name[:size]:>{size}} ")
        return "*    Row   " + ("* Instance " if self.instances else "") + "".join(cells) + "*"

    def cell(self, value: Any, size: int) -> str:
        """One value as ``PrintValue`` prints it, then cut to its column."""
        return f"* {self.text(value)[:size]:>{size}} "

    def text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return str(value)
        text = f"{float(value):{self.column}.{self.precision}g}"
        exponent = text.find("e")
        if exponent >= 0 and len(text) > self.column:
            cut = len(text) - self.column  # trim the digits before the exponent
            text = text[: exponent - cut] + text[exponent:]
        return text


def _columns(source: Any, varexp: str) -> list[str]:
    if varexp == "*":
        return list(source.readable())
    if not varexp:
        return list(source.readable())[:FIRST]
    return [name.strip() for name in split_names(varexp)]


def _layout(
    names: list[str], width: int | None, precision: int | None, instances: bool
) -> Layout:
    column = COLUMN if width is None else width
    widest = WIDEST if width is None else width
    if precision is None:
        precision = COLUMN if width is None else min(width, MOST_PRECISE)
    if column < 1 or precision < 0:
        raise ValueError("a column is at least one character wide, and precision is not negative")
    sizes = [min(max(len(name), column), max(widest, column)) for name in names]
    return Layout(names, sizes, column, precision, instances)


class Result(NamedTuple):
    """One formula's values for a batch: per entry, or rows of elements."""

    values: Any
    valid: Any

    @property
    def rows(self) -> bool:
        return isinstance(self.values, Jagged)

    def count(self, entry: int) -> int:
        """How many values the entry has: its elements, or one."""
        if not self.rows:
            return 1
        return int(self.values.offsets[entry + 1] - self.values.offsets[entry])

    def at(self, entry: int, instance: int) -> Any:
        """The value of one instance of one entry, or ``None`` where there is none."""
        if not self.rows:
            return self.values[entry] if self.valid[entry] else None
        if instance >= self.count(entry):
            return None
        place = int(self.values.offsets[entry]) + instance
        return self.values.content[place] if self.valid.content[place] else None


def _passes(value: Any) -> bool:
    return value is not None and bool(value != 0)


class Table:
    """The rows of a scan, an entry at a time."""

    def __init__(
        self, layout: Layout, formulas: list[Formula], selection: Formula | None
    ) -> None:
        self.layout = layout
        self.formulas = formulas
        self.selection = selection
        #: With a selection that loops, every column goes down its loop.
        self.together = selection is not None and bool(selection.per_element)
        self.selected = 0

    def results(self, columns: Mapping[str, Any], read: Any, total: int) -> list[Result]:
        everything = self.formulas + ([self.selection] if self.selection else [])
        given = {
            "entries": total,
            "rows": len(read.numbers),
            "entry_numbers": read.numbers,
            "local_entries": read.local,
        }
        if self.together:
            found = evaluate_together(everything, columns, **given)
        else:
            found = [evaluate_together([f], columns, **given)[0] for f in everything]
        return [Result(*pair) for pair in found]

    def _count(self, results: Sequence[Result], entry: int) -> int:
        """How many rows an entry prints before the selection is asked."""
        if self.together:
            return results[-1].count(entry)
        chosen = results[: len(self.formulas)]
        count = max([1, *(r.count(entry) for r in chosen)]) if self.layout.instances else 1
        if self.selection is not None and results[-1].at(entry, 0) is None:
            return 0  # the selection has no value, so nothing is selected
        return count

    def rows(self, results: Sequence[Result], numbers: Array) -> Iterator[str]:
        for entry, number in enumerate(numbers.tolist()):
            for instance in range(self._count(results, entry)):
                if self.selection is not None and not _passes(results[-1].at(entry, instance)):
                    continue
                self.selected += 1
                yield self._row(results, entry, instance, number)

    def _row(self, results: Sequence[Result], entry: int, instance: int, number: int) -> str:
        line = f"* {number:8d} " + (f"* {instance:8d} " if self.layout.instances else "")
        for result, size in zip(results, self.layout.sizes):
            line += self.layout.cell(result.at(entry, instance), size)
        return line + "*"


def _compiled(
    source: Any, names: list[str], selection: str, aliases: Mapping[str, str] | None
) -> tuple[list[Formula], Formula | None]:
    known = TreeNames(source)
    formulas = [compile_formula(name, known, aliases=aliases) for name in names]
    cut = compile_formula(selection, known, aliases=aliases) if selection else None
    return formulas, cut


def _batches(
    source: Any, table: Table, first: int, entries: int | None, step: int
) -> Iterator[list[str]]:
    total = len(source)
    stop = total if entries is None else min(total, first + entries)
    everything = table.formulas + ([table.selection] if table.selection else [])
    branches = list(dict.fromkeys(name for f in everything for name in f.branches))
    for start in range(first, stop, step):
        read = _Read(source, start, min(start + step, stop), None)
        columns = {n: _shaped(_underlying(source[n]), read.column(n)) for n in branches}
        yield list(table.rows(table.results(columns, read, total), read.numbers))


def _checked(first_entry: int, entries: int | None, step: int) -> None:
    if first_entry < 0 or (entries is not None and entries < 0) or step < 1:
        raise ValueError(
            "first_entry and entries count entries and cannot be negative, and step is at "
            "least one"
        )


class Output:
    """The lines of a scan, kept, and written to ``file`` as they come."""

    def __init__(self, file: IO[str] | None) -> None:
        self.file = file
        self.lines: list[str] = []

    def emit(self, more: list[str]) -> None:
        self.lines.extend(more)
        if self.file is not None and more:
            self.file.write("\n".join(more) + "\n")


def scan(
    source: Any,
    varexp: str = "*",
    selection: str = "",
    *,
    entries: int | None = None,
    first_entry: int = 0,
    width: int | None = None,
    precision: int | None = None,
    file: IO[str] | None = None,
    step: int = STEP,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """``TTree::Scan`` on a tree, a chain or anything that reads like one; see ``TTree.scan``."""
    _checked(first_entry, entries, step)
    names = _columns(source, varexp)
    formulas, cut = _compiled(source, names, selection, aliases)
    looping = any(bool(f.per_element) for f in [*formulas, *([cut] if cut else [])])
    layout = _layout(names, width, precision, looping)
    table = Table(layout, formulas, cut)
    out = Output(file)
    out.emit([layout.border(), layout.header(), layout.border()])
    for rows in _batches(source, table, first_entry, entries, step):
        out.emit(rows)
    out.emit([layout.border()])
    if selection:
        out.emit([f"==> {table.selected} selected {'entry' if table.selected == 1 else 'entries'}"])
    return "\n".join(out.lines) + "\n"
