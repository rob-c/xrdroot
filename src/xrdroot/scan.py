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
anything too wide for its column cut off at the column's edge - a name that
is too wide for its column is cut off the same way; a string as it is. A
column is nine characters wide unless ``colsize=`` or a column's own
``printf`` conversion says otherwise.

An expression that loops over a collection prints a row per element under an
``Instance`` column. The columns go down together only when the selection
loops too - then all of them share its loop, as ``TTreeFormulaManager`` makes
them - and otherwise each goes as far as it has elements, the row count of an
entry being the most any of them has and the columns that run out left blank.
"""

from __future__ import annotations

import re
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
#: The most digits ``colsize=`` gives the precision.
MOST_PRECISE = 18
#: Entries read at a time.
STEP = 10_000
#: The columns an empty expression prints: ROOT's first eight leaves.
FIRST = 8

#: A ``printf`` conversion without its ``%``: the flags, width and precision, the length
#: that casts the value, and the letter that names its type - ``9.9``, ``#x``, ``5lld``.
CONVERSION = re.compile(r"([hlLjzt]*)([cdioxXufFeEgG])$")
#: The digits a conversion starts with: its width, which is also what it keeps of a number.
_DIGITS = re.compile(r"\d*")
#: What a conversion's type letter casts to: the widths ROOT's ``PrintValue`` uses.
SIGNED = {"": 32, "h": 16, "l": 64, "ll": 64, "L": 64, "j": 64, "z": 64, "t": 64}


class Field(NamedTuple):
    """One column: how wide it is, and the ``printf`` conversion its values go through."""

    size: int
    #: ROOT's ``decform``, the part of the conversion after the ``%``.
    conversion: str


class Layout(NamedTuple):
    """The widths a table is printed with, and whether it has an ``Instance`` column."""

    names: list[str]
    fields: list[Field]
    instances: bool

    def border(self) -> str:
        """A line of stars as wide as the table."""
        start = "*" * (22 if self.instances else 11)
        return start + "".join("*" * (field.size + 3) for field in self.fields) + "*"

    def header(self) -> str:
        """The names of the columns, each cut off at its column's edge, as ``%9.9s`` cuts."""
        cells = []
        for name, field in zip(self.names, self.fields, strict=False):
            cells.append(f"* {name[: field.size]:>{field.size}} ")
        return "*    Row   " + ("* Instance " if self.instances else "") + "".join(cells) + "*"

    def cell(self, value: Any, field: Field) -> str:
        """One value as ``PrintValue`` prints it, then cut to its column."""
        return f"* {_printed(value, field)[: field.size]:>{field.size}} "


def _cast(value: Any, length: str, signed: bool) -> int:
    """``value`` as the C type a conversion's length names, wrapping as C's cast does."""
    bits = SIGNED.get(length.lower(), 64)
    number = int(value)
    if signed and bits == 64:
        return number
    masked = number & ((1 << bits) - 1)
    if signed and masked >> (bits - 1):
        masked -= 1 << bits
    return masked


def _text(spec: str, value: int | float) -> str:
    """``value`` through ``spec``: Python's ``%`` takes every conversion C's does, less

    the length modifier, which only chooses the cast and is dropped here."""
    found = CONVERSION.search(spec)
    if found is None:
        return f"{float(value):{spec}g}"
    return ("%" + spec[: found.start(1)] + found.group(2)) % value


def _printed(value: Any, field: Field) -> str:
    """One value as ``TTreeFormula::PrintValue`` prints it under the column's conversion.

    A string is printed as it is, whatever the conversion: ROOT's ``PrintValue`` returns
    a string column's value before it looks at one. A number goes through
    :func:`_number`, and nothing - a column with no value this instance - is blank.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return str(value)
    return _number(value, field.conversion)


def _number(value: Any, conversion: str) -> str:
    """A number through the conversion, cast to the type its last letter names - ``%d`` as
    an ``Int_t``, ``%x`` as a ``UInt_t``, ``%c`` as the ``int`` C prints a byte of - and a
    conversion that names no type at all is a ``%g``, as ``PrintValue``'s default is."""
    found = CONVERSION.search(conversion)
    letter, length = (found.group(2), found.group(1)) if found is not None else ("", "")
    if letter == "c":
        return _text(conversion, _cast(value, length, signed=True) & 0xFF)
    if letter in ("d", "i"):
        return _text(conversion, _cast(value, length, signed=True))
    if letter in ("o", "x", "X", "u"):
        return _text(conversion, _cast(value, length, signed=False))
    return _trimmed(_text(conversion, float(value)), conversion)


def _leading(conversion: str) -> int:
    """The number a conversion starts with, which is both its width and its digit count."""
    found = _DIGITS.match(conversion)
    return int(found.group(0) or 0) if found is not None else 0


def _trimmed(text: str, conversion: str) -> str:
    """``text`` with its digits before the exponent dropped, when it is wider than the
    conversion asked for: ``%9.9g`` of a large number loses everything but ``e+20``'s few."""
    exponent = text.find("e")
    digits = _leading(conversion)
    if exponent < 0 or len(text) <= digits:
        return text
    cut = len(text) - digits
    return text[: exponent - cut] + text[exponent:]


def _width(conversion: str, column: int) -> int:
    """A column's width: the number its own conversion starts with, else ROOT's default."""
    return _leading(conversion) or column


def _columns(source: Any, varexp: str) -> list[str]:
    if varexp == "*":
        return list(source.readable())
    if not varexp:
        return list(source.readable())[:FIRST]
    return [name.strip() for name in split_names(varexp)]


def _layout(
    names: list[str],
    width: int | None,
    precision: int | None,
    conversions: Sequence[str] | None,
    instances: bool,
) -> Layout:
    column = COLUMN if width is None else width
    if precision is None:
        precision = COLUMN if width is None else min(width, MOST_PRECISE)
    if column < 1 or precision < 0:
        raise ValueError("a column is at least one character wide, and precision is not negative")
    default = f"{column}.{precision}"
    wanted = list(conversions or [])
    return Layout(names, [_field(at, wanted, default, column) for at in range(len(names))],
                  instances)


def _field(at: int, wanted: list[str], default: str, column: int) -> Field:
    """The column at ``at``: its own conversion if the list reached it and it is not empty -
    ROOT reads an empty ``col=`` entry as the default - and its width out of that."""
    given = wanted[at] if at < len(wanted) else ""
    conversion = given or default
    return Field(_width(conversion, column), conversion)


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
        for result, field in zip(results, self.layout.fields, strict=False):
            line += self.layout.cell(result.at(entry, instance), field)
        return line + "*"


def _whole_of(source: Any, branches: Sequence[str]) -> Any:
    """The members the tree's whole objects offer, or ``None`` when every name is a branch."""
    return None if all(name in source for name in branches) else TreeNames(source).whole


def _column_of(source: Any, name: str, whole: Any) -> Any:
    """What a name is read from: a branch, or the whole object a member of it is."""
    return source[whole.branch(name)] if whole is not None and name in whole else source[name]


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
    # A scan may name a member of an object the tree holds whole - ``Scan("fTemperature")``
    # of a branch of Events - which is no branch at all: the members are what the tree's
    # objects offer, learned the first time an expression reads inside one of them.
    whole = _whole_of(source, branches)
    for start in range(first, stop, step):
        read = _Read(source, start, min(start + step, stop), None, whole, borrowed=True)
        columns = {n: _shaped(_underlying(_column_of(source, n, whole)), read.column(n))
                   for n in branches}  # fmt: skip
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
    conversions: Sequence[str] | None = None,
    file: IO[str] | None = None,
    step: int = STEP,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """``TTree::Scan`` on a tree, a chain or anything that reads like one; see ``TTree.scan``."""
    _checked(first_entry, entries, step)
    names = _columns(source, varexp)
    formulas, cut = _compiled(source, names, selection, aliases)
    looping = any(bool(f.per_element) for f in [*formulas, *([cut] if cut else [])])
    layout = _layout(names, width, precision, conversions, looping)
    table = Table(layout, formulas, cut)
    out = Output(file)
    out.emit([layout.border(), layout.header(), layout.border()])
    for rows in _batches(source, table, first_entry, entries, step):
        out.emit(rows)
    out.emit([layout.border()])
    if selection:
        out.emit([f"==> {table.selected} selected {'entry' if table.selected == 1 else 'entries'}"])
    return "\n".join(out.lines) + "\n"
