"""``"x/D"``, ``"x[n]/F"``, ``"a/I:b:c/F"``: ROOT's leaf lists, read.

A leaf list is how ``TTree::Branch`` is told what an address holds: leaves
separated by colons, each a name, the sizes of its dimensions in brackets and
a slash and a letter for its type. A size is a number, or the name of a leaf
that says how many there are in each entry - which must come before it. A
leaf with no type takes the one before it, and the first leaf with no type
is a ``Float_t``, exactly as ``TTree::Branch``'s documentation has it.
"""

from __future__ import annotations

import re
from typing import NamedTuple

__all__ = ["Leaf", "parse", "LETTERS", "LEAF_CLASSES", "TYPE_NAMES"]

#: What each of ROOT's type letters is, as an :mod:`array` type code.
LETTERS: dict[str, str] = {
    "C": "C",
    "B": "b",
    "b": "B",
    "S": "h",
    "s": "H",
    "I": "i",
    "i": "I",
    "F": "f",
    "f": "f",
    "D": "d",
    "d": "d",
    "L": "q",
    "l": "Q",
    "G": "q",
    "g": "Q",
    "O": "?",
}

#: The leaf class ROOT keeps a leaf of each type code in.
LEAF_CLASSES: dict[str, str] = {
    "C": "TLeafC",
    "b": "TLeafB",
    "B": "TLeafB",
    "h": "TLeafS",
    "H": "TLeafS",
    "i": "TLeafI",
    "I": "TLeafI",
    "f": "TLeafF",
    "d": "TLeafD",
    "q": "TLeafL",
    "Q": "TLeafL",
    "?": "TLeafO",
}

#: The type name ``TLeaf::GetTypeName`` gives for each type code.
TYPE_NAMES: dict[str, str] = {
    "C": "Char_t",
    "b": "Char_t",
    "B": "UChar_t",
    "h": "Short_t",
    "H": "UShort_t",
    "i": "Int_t",
    "I": "UInt_t",
    "f": "Float_t",
    "d": "Double_t",
    "q": "Long64_t",
    "Q": "ULong64_t",
    "?": "Bool_t",
}

#: The letter a branch title spells each type code with, as ROOT writes it.
CODE_LETTERS: dict[str, str] = {
    "C": "C",
    "b": "B",
    "B": "b",
    "h": "S",
    "H": "s",
    "i": "I",
    "I": "i",
    "f": "F",
    "d": "D",
    "q": "L",
    "Q": "l",
    "?": "O",
}

#: One leaf: a name, then any number of bracketed sizes, then maybe a type.
LEAF = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[[^\[\]]+\])*)(?:/(.))?$")
SIZES = re.compile(r"\[([^\[\]]+)\]")


class Leaf(NamedTuple):
    """One leaf of a leaf list, as it was declared.

    ``code`` is the :mod:`array` type code, or ``C`` for a string; ``size``
    is how many values an entry holds when that is fixed - the product of
    the numbered dimensions - and ``counter`` the leaf whose value says how
    many times that many there are, when it is not.
    """

    name: str
    code: str
    size: int
    counter: str | None
    #: How the leaf's title is written: its name and its brackets.
    title: str


def _dimensions(name: str, sizes: str, text: str) -> tuple[int, str | None]:
    """A leaf's fixed size, and its counter: ``[n][3]`` is 3 values, ``n`` times."""
    found = SIZES.findall(sizes)
    counter = None
    if found and not found[0].isdigit():
        counter = found.pop(0)
    if not all(size.isdigit() and int(size) > 0 for size in found):
        raise ValueError(
            f"{text!r} gives {name!r} the sizes {sizes}; a leaf's first size may be the "
            f"name of the leaf counting it, and every size after that is a number"
        )
    size = 1
    for each in found:
        size *= int(each)
    return size, counter


def _leaf(part: str, before: str, text: str) -> Leaf:
    found = LEAF.match(part.strip())
    if found is None:
        raise ValueError(
            f"{part!r} in the leaf list {text!r} is not a leaf; a leaf is a name, any "
            f"sizes in brackets, and a slash and a type letter, as in x[n]/F"
        )
    name, sizes, letter = found.groups()
    if letter is not None and letter not in LETTERS:
        raise ValueError(
            f"{letter!r} is not a type ROOT has a letter for, in {text!r}; the letters are "
            f"{' '.join(LETTERS)}"
        )
    size, counter = _dimensions(name, sizes, text)
    code = LETTERS[letter] if letter is not None else before
    return Leaf(name, code, size, counter, f"{name}{sizes}")


def parse(text: str) -> list[Leaf]:
    """Every leaf a leaf list declares, in order, with the types it inherits filled in."""
    leaves: list[Leaf] = []
    before = "f"
    for part in text.split(":"):
        leaf = _leaf(part, before, text)
        leaves.append(leaf)
        before = leaf.code
    names = [leaf.name for leaf in leaves]
    if len(set(names)) != len(names):
        raise ValueError(f"the leaf list {text!r} names one leaf twice")
    return leaves
