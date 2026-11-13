"""``xrdroot ls``: ROOT's ``rootls``, flag for flag and space for space.

    $ xrdroot ls test.root                      # dir  hprof  hpx  hpxpy  tof
    $ xrdroot ls -1 test.root:tof               # one name a line
    $ xrdroot ls -l 'test.root:h*'              # class, date, name;cycle and title
    $ xrdroot ls -t simpleTree.root             # and each tree's branches and clusters
    $ xrdroot ls -R RNTuple.root                # and each RNTuple's fields
    $ xrdroot ls -r subdirs.root                # into every directory

Each FILE is a file and, after a colon, a path in it whose every part may be
a shell pattern - ``test.root:tof/h0_*``, ``f.root:tree;2`` - read as
:mod:`.sources` reads it. The listing is ``rootls``'s: names in columns as
wide as an 80-column terminal allows, or one a line with ``-1``; ``-c``
shows every cycle; ``-l`` a line per key with its class, the date its key
carries and its title, ``-t`` the same with each tree's branches, their
sizes and its cluster ranges, ``-R`` with each RNTuple's fields; ``-r``
goes into every directory. A pattern that names a directory lists what is
in it; several files, or several things in one, are each listed under a
``name :`` heading. What will not open, or matches nothing, is said as
``rootls`` says it, and the status is then 1.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import ROOTError
from .sources import Matches, Node, matching
from .tooling import Flag, Log, OptionError, Options, raw_parser

__all__ = ["add_parser", "run", "USAGE", "HELP"]

#: What ``rootls`` says when its command line is wrong, on standard error.
USAGE = "usage: rootls [-1hltr] FILE [FILE ...]\n"

#: The rest of ``rootls -h``, after the usage line.
HELP = """
Display ROOT files contents in the terminal.

positional arguments:
  FILE                    Input file

options:
  -h, --help              show this help message and exit
  -1, --oneColumn         Print content in one column
  -c, --allCycles         Force display of all objects' cycles
  -l, --longListing       Use a long listing format.
  -t, --treeListing       Print tree recursively and use a long listing format.
  -R, --rntupleListing    Print RNTuples recursively and use a long listing format.
  -r, --recursiveListing  Traverse file recursively entering any TDirectory.

Examples:
- rootls example.root
  Display contents of the ROOT file 'example.root'.

- rootls example.root:dir
  Display contents of the directory 'dir' from the ROOT file 'example.root'.

- rootls example.root:*
  Display contents of the ROOT file 'example.root' and his subdirectories.

- rootls file1.root file2.root
  Display contents of ROOT files 'file1.root' and 'file2.root'.

- rootls *.root
  Display contents of ROOT files whose name ends with '.root'.

- rootls -1 example.root
  Display contents of the ROOT file 'example.root' in one column.

- rootls -l example.root
  Display contents of the ROOT file 'example.root' and use a long listing format.

- rootls -t example.root
  Display contents of the ROOT file 'example.root', use a long listing format and print trees recursively.

- rootls -r example.root
  Display contents of the ROOT file 'example.root', traversing recursively any TDirectory.
"""  # noqa: E501

#: ``rootls``'s flags.
FLAGS = (
    Flag(("-h", "--help")),
    Flag(("-1", "--oneColumn")),
    Flag(("-c", "--allCycles")),
    Flag(("-l", "--longListing")),
    Flag(("-t", "--treeListing")),
    Flag(("-R", "--rntupleListing")),
    Flag(("-r", "--recursiveListing")),
)

#: The spaces between columns of names, and a level of indentation.
SEPARATION = INDENT = 2
#: How wide a terminal is taken to be when none says.
COLUMNS = 80
#: The months as ``TDatime`` dates are printed.
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
#: The classes whose keys ``-t`` and ``-R`` open and describe.
TREES = ("TTree", "TNtuple", "TNtupleD")
RNTUPLE = "ROOT::RNTuple"


@dataclass(frozen=True)
class Listing:
    """What the flags asked for."""

    one_column: bool
    cycles: bool
    long: bool
    trees: bool
    ntuples: bool
    recursive: bool

    @property
    def detailed(self) -> bool:
        return self.long or self.trees or self.ntuples


def add_parser(subparsers: Any) -> None:
    raw_parser(subparsers, "ls", "list what ROOT files hold, as rootls does")


def _width() -> int:
    """The terminal's width as ``rootls`` finds it: standard input's, output's, error's."""
    for fd in (0, 1, 2):
        try:
            return os.get_terminal_size(fd).columns
        except OSError:
            continue
    return COLUMNS


def run(args: argparse.Namespace) -> int:
    log = Log("rootls")
    try:
        options = Options(FLAGS).parse(args.argv)
    except OptionError as why:
        log.error(str(why))
        sys.stderr.write(USAGE)
        return 1
    if options.has("h"):
        sys.stderr.write(USAGE + HELP)
        return 0
    if not options.args:
        sys.stderr.write(USAGE)
        return 1
    listing = Listing(*(options.has(name) for name in "1cltRr"))
    return _list(options.args, listing, log)


def _list(given: Sequence[str], listing: Listing, log: Log) -> int:
    """Every source matched first, its complaints said, then each listed in name order."""
    try:
        found = [_said(matching(text, listing.recursive), log) for text in given]
    except ValueError as why:
        log.error(str(why))
        return 1
    failed = any(matched.errors for matched in found)
    printer = _Printer(listing, _width())
    try:
        for matched in sorted(found, key=lambda each: each.filename):
            printer.source(matched, many=len(found) > 1)
    finally:
        for matched in found:
            matched.close()
    sys.stdout.write("".join(printer.out))
    return 1 if failed else 0


def _said(matched: Matches, log: Log) -> Matches:
    """A source's complaints, said as ROOT and rootls say them."""
    for line in matched.said:
        sys.stderr.write(line + "\n")
    for message in matched.errors:
        log.error(message)
    return matched


class _Printer:
    """``rootls``'s output, built up line by line."""

    def __init__(self, listing: Listing, width: int) -> None:
        self.listing = listing
        self.width = width
        self.out: list[str] = []

    def source(self, matched: Matches, many: bool) -> None:
        indent = 0
        if many:
            self.out.append(f"{matched.filename} :\n")
            indent = INDENT
        items = len(matched.dirs) + len(matched.leaves)
        if items > 1:
            self.group(matched, matched.leaves, indent)
            for index in matched.dirs:
                self.out.append(f"{' ' * indent}{matched.path(index)} :\n")
                self.group(matched, matched.nodes[index].children, indent + INDENT)
        elif matched.dirs:
            self.group(matched, matched.nodes[matched.dirs[0]].children, indent)
        else:
            self.group(matched, matched.leaves, indent)

    def group(self, matched: Matches, indices: list[int], indent: int) -> None:
        if not indices:
            return
        if self.listing.detailed:
            self._long(matched, indices, indent)
        else:
            self._short(matched, indices, indent)

    def _expands(self, matched: Matches, index: int) -> bool:
        node = matched.nodes[index]
        return self.listing.recursive and node.is_directory and bool(node.children)

    # -- names in columns ------------------------------------------------------------

    def _label(self, node: Node) -> str:
        return f"{node.name};{node.cycle}" if self.listing.cycles else node.name

    def _unique(self, matched: Matches, indices: list[int]) -> list[int]:
        """The names to show: every cycle with ``-c``, else each name's first - its newest."""
        if self.listing.cycles:
            return list(indices)
        names = [matched.nodes[index].name for index in indices]
        return [index for at, index in enumerate(indices) if not at or names[at] != names[at - 1]]

    def _short(self, matched: Matches, indices: list[int], indent: int) -> None:
        unique = self._unique(matched, indices)
        nodes = [matched.nodes[index] for index in indices]
        widths = self._columns(nodes, len(unique), indent)
        column = 0
        for at, index in enumerate(unique):
            if column == 0:
                self.out.append(" " * indent)
            label = self._label(matched.nodes[index])
            expands = self._expands(matched, index)
            if self._ends_row(matched, unique, at, column + 1, len(widths)) or expands:
                self.out.append(label + "\n")
                column = 0
            else:
                self.out.append(label.ljust(widths[column]))
                column += 1
            if expands:
                self.group(matched, matched.nodes[index].children, indent + INDENT)

    def _ends_row(
        self, matched: Matches, unique: list[int], at: int, filled: int, columns: int
    ) -> bool:
        """Whether a name ends its row: the row full, the names done, or a directory next."""
        if filled % columns == 0 or at == len(unique) - 1:
            return True
        return self._expands(matched, unique[at + 1])

    def _columns(self, nodes: list[Node], count: int, indent: int) -> list[int]:
        """Each column's width: the most columns whose widest names fit the terminal.

        As ``rootls`` does, the widths are of the first ``count`` names as
        given, before cycles past the newest were dropped.
        """
        room = self.width - indent
        sizes = [len(self._label(node)) + SEPARATION for node in nodes[:count]]
        if self.listing.one_column or max(sizes) > room:
            return [0]
        columns = min(count, room // min(sizes))
        while True:
            widths = [max(sizes[column::columns]) for column in range(columns)]
            if sum(widths) <= room:
                return widths
            columns -= 1

    # -- a line per key --------------------------------------------------------------

    def _long(
        self, matched: Matches, indices: list[int], indent: int, least: tuple[int, int] = (0, 0)
    ) -> None:
        """A line per key; a directory opened under one lines its columns up at least as far
        as the one around it, less the indentation."""
        nodes = [matched.nodes[index] for index in indices]
        class_width = max(least[0], max(len(_class(node)) for node in nodes) + SEPARATION)
        name_width = max(least[1], max(len(node.name) for node in nodes) + SEPARATION)
        inner = (class_width - INDENT, name_width - INDENT)
        for index, node in zip(indices, nodes, strict=True):
            assert node.key is not None
            label = f"{node.name};{node.cycle}".ljust(name_width)
            self.out.append(f"{' ' * indent}{_class(node).ljust(class_width)}"
                            f"{_date(node.key.datime)}{label} \"{node.key.title}\" "
                            f"{_cycle_note(matched, index)}\n")  # fmt: skip
            self._details(matched, index, indent + INDENT, inner)

    def _details(
        self, matched: Matches, index: int, indent: int, least: tuple[int, int]
    ) -> None:
        """What goes under a key's line: a tree's branches, an RNTuple's fields, a directory."""
        node = matched.nodes[index]
        if self.listing.trees and node.classname in TREES:
            self.out.extend(_tree_lines(matched, node, indent))
        elif self.listing.ntuples and node.classname == RNTUPLE:
            self.out.extend(_ntuple_lines(matched, node, indent))
        elif self._expands(matched, index):
            self._long(matched, node.children, indent, least)


def _class(node: Node) -> str:
    """A key's class as ROOT reads it: an old ``TDirectory`` key is a ``TDirectoryFile``."""
    return "TDirectoryFile" if node.classname == "TDirectory" else node.classname


def _date(datime: int) -> str:
    """A key's date as ``rootls`` prints it - the minutes, as it happens, unpadded."""
    year, month, day = (datime >> 26) + 1995, (datime >> 22) & 0xF, (datime >> 17) & 0x1F
    hour, minute = (datime >> 12) & 0x1F, (datime >> 6) & 0x3F
    named = MONTHS[month - 1] if 1 <= month <= len(MONTHS) else "???"
    return f"{named} {day:02d} {hour:02d}:{minute} {year} "


def _cycle_note(matched: Matches, index: int) -> str:
    """``[current cycle]`` or ``[backup cycle]``, by the nodes either side, as rootls has it."""
    nodes = matched.nodes
    name = nodes[index].name
    if nodes[index - 1].name == name:
        return "[backup cycle]"
    if index + 1 < len(nodes) and nodes[index + 1].name == name:
        return "[current cycle]"
    return ""


def _opened(matched: Matches, node: Node) -> Any:
    parent = matched.nodes[node.parent].directory
    assert parent is not None
    return parent[f"{node.name};{node.cycle}"]


def _tree_lines(matched: Matches, node: Node, indent: int) -> list[str]:
    """A tree's branches with their titles and sizes, then its clusters, as rootls lists them."""
    try:
        tree = _opened(matched, node)
    except (ROOTError, KeyError, ValueError):
        return []
    lines = _branch_lines(tree.records, indent)
    ranges = tree.clustering.ranges()
    pad = " " * indent
    lines.append(f"{pad}Cluster INCLUSIVE ranges:\n")
    lines.extend(f"{pad} - # {at}: [{start}, {stop - 1}]\n" for at, (start, stop) in
                 enumerate(ranges))  # fmt: skip
    lines.append(f"{pad}The total number of clusters is {len(ranges)}\n")
    return lines


def _branch_lines(records: list[Any], indent: int) -> list[str]:
    if not records:
        return []
    name_width = max(len(record.name) for record in records) + SEPARATION
    title_width = max(len(record.title) for record in records) + 2 + SEPARATION
    lines = []
    for record in records:
        title = f'"{record.title}"'
        lines.append(f"{' ' * indent}{record.name.ljust(name_width)}{title.ljust(title_width)}"
                     f"{record.tot_bytes}\n")  # fmt: skip
        lines.extend(_branch_lines(record.branches, indent + INDENT))
    return lines


def _ntuple_lines(matched: Matches, node: Node, indent: int) -> list[str]:
    """An RNTuple's fields, sorted, each level's names dotted out to the widest."""
    ntuple = _opened(matched, node)
    schema = ntuple._store.schema
    return _field_lines(schema, [field.id for field in schema.tops()], indent, 0, 0, " ")


def _field_lines(
    schema: Any, ids: list[int], indent: int, least_name: int, least_type: int, fill: str
) -> list[str]:
    fields = sorted((schema.fields[each] for each in ids), key=lambda field: field.name.encode())
    if not fields:
        return []
    name_width = max(least_name, max(len(field.name) for field in fields) + SEPARATION)
    type_width = max(least_type, max(len(field.type) for field in fields) + 2 * SEPARATION)
    lines = []
    for field in fields:
        about = f'"{field.description}"' if field.description else ""
        lines.append(f"{' ' * indent}{field.name.ljust(name_width, fill)}"
                     f"{field.type.ljust(type_width)}{about}\n")  # fmt: skip
        least = (name_width - SEPARATION, type_width - SEPARATION)
        lines.extend(_field_lines(schema, field.children, indent + INDENT, *least, "."))
    return lines
