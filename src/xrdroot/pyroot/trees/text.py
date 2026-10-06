"""``TTree::ReadFile``: a file of columns of text, read into a tree a line at a time.

The branches are the descriptor's - a leaf list, ``x/F:y:z`` - or, when it
is empty, the file's first line's; a tree that has branches already (a
``TNtuple``) takes each line's values into them in order. Lines that are
empty or start with ``#`` are passed over, as ROOT passes them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .leaflist import parse

__all__ = ["read_file"]


def _values(line: str, delimiter: str) -> list[str]:
    return line.split() if delimiter == " " else [part.strip() for part in line.split(delimiter)]


def _branches(tree: Any, descriptor: str) -> list[Any]:
    """The addresses a line's values go into: the tree's own, or new ones for the descriptor."""
    store = tree._writable("ReadFile")
    if store.slots and not descriptor:
        return [slot.address for slot in store.slots.values()]
    made = []
    for leaf in parse(descriptor):
        holder: Any = bytearray(256) if leaf.code == "C" else np.zeros(1, dtype=leaf.code)
        tree.Branch(leaf.name, holder, f"{leaf.title}/{_letter(leaf.code)}")
        made.append(store.slots[leaf.name].address)
    return made


def _letter(code: str) -> str:
    from .leaflist import CODE_LETTERS

    return CODE_LETTERS[code]


def read_file(tree: Any, filename: str, descriptor: str, delimiter: str) -> int:
    """Fill ``tree`` from the lines of ``filename``; how many were read comes back."""
    with open(filename) as text:
        wanted = [line.rstrip("\n") for line in text if _wanted(line)]
    if not descriptor and not tree._writable("ReadFile").slots and wanted:
        descriptor, wanted = wanted[0].strip(), wanted[1:]
    addresses = _branches(tree, descriptor)
    for line in wanted:
        for address, value in zip(addresses, _values(line, delimiter), strict=False):
            address.put(value if address.text else float(value))
        tree.Fill()
    return len(wanted)


def _wanted(line: str) -> bool:
    """Whether a line holds values: not empty, and not a comment."""
    return bool(line.strip()) and not line.lstrip().startswith("#")
