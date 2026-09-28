"""``RooPrintable``: how every RooFit object prints itself, in ROOT's words and widths.

``Print()`` on a RooFit object picks a style from its option - one line,
``"s"`` standard, ``"v"`` verbose, ``"t"`` the tree of what it depends on -
and a set of contents - its class, name, arguments, value, extras - and
prints them in a fixed order with fixed separators. Each class says what
its value and extras are (:meth:`RooPrintable.printValue`...) and the rest is
this module, so every object's line reads as ROOT's does: ``RooRealVar::mean
= 1.01746 +/- 0.0300144  L(-10 - 10)``.

Numbers are printed as a C++ ``std::ostream`` prints a ``double`` by
default, six significant figures (:func:`g`).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from . import cout

__all__ = [
    "kName",
    "kClassName",
    "kValue",
    "kArgs",
    "kExtras",
    "kAddress",
    "kTitle",
    "kCollectionHeader",
    "kInline",
    "kSingleLine",
    "kStandard",
    "kVerbose",
    "kTreeStructure",
    "RooPrintable",
    "PRECISION",
    "g",
    "address",
    "style_of",
]

#: ``RooPrintable::ContentsOption``.
kName, kClassName, kValue, kArgs, kExtras, kAddress, kTitle, kCollectionHeader = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
)
#: ``RooPrintable::StyleOption``.
kInline, kSingleLine, kStandard, kVerbose, kTreeStructure = range(1, 6)


#: The precision ``std::cout`` prints a ``double`` with: six, until something changes it -
#: RooFit's verbose minimisation leaves it at four for the rest of the process.
PRECISION = [6]


def g(value: Any, digits: int | None = None) -> str:
    """``os << value`` for a ``double``: ``%g`` with ``std::cout``'s precision, as C++ prints it."""
    digits = PRECISION[0] if digits is None else digits
    number = float(value)
    if math.isnan(number):
        return "nan" if math.copysign(1.0, number) > 0 else "-nan"
    return format(number, f".{digits}g")


def address(obj: Any) -> str:
    """What ``os << this`` prints: a pointer, in hexadecimal - which a comparison ignores."""
    return hex(id(obj))


def style_of(option: Any) -> int:
    """``RooPrintable::defaultPrintStyle``: the style an option string asks for."""
    text = str(option or "").lower()
    for letter, style in (("v", kVerbose), ("s", kStandard), ("i", kInline), ("t", kTreeStructure)):
        if letter in text:
            return style
    return kSingleLine


class RooPrintable:
    """The printing every RooFit object shares: ``Print``, ``printStream`` and their parts."""

    #: ``RooPrintable::_nameLength``: the width names are printed in, when nonzero.
    name_length: ClassVar[list[int]] = [0]

    def printName(self) -> str:
        return ""

    def printTitle(self) -> str:
        return ""

    def printClassName(self) -> str:
        return ""

    def printArgs(self) -> str:
        return ""

    def printValue(self) -> str:
        return ""

    def printExtras(self) -> str:
        return ""

    def printAddress(self) -> str:
        return address(self)

    def printMultiline(self, contents: int, verbose: bool, indent: str) -> str:
        return ""

    def printTree(self, indent: str) -> str:
        return f"Tree structure printing not implement for class {type(self).__name__}\n"

    def defaultPrintContents(self, option: Any) -> int:
        return kName | kValue

    def defaultPrintStyle(self, option: Any) -> int:
        return style_of(option)

    def printStream(self, contents: int, style: int, indent: str = "") -> str:
        """``RooPrintable::printStream``: the text ``contents`` in ``style`` makes."""
        if style in (kVerbose, kStandard):
            return self.printMultiline(contents, style == kVerbose, indent)
        if style == kTreeStructure:
            return self.printTree(indent)
        text = "" if style == kInline else indent
        text += self._parts(contents)
        return text if style == kInline else text + "\n"

    def _parts(self, contents: int) -> str:
        text = ""
        if contents & kAddress:
            text += self.printAddress() + (" " if contents != kAddress else "")
        if contents & kClassName:
            text += self.printClassName() + ("::" if contents != kClassName else "")
        if contents & kName:
            width = self.name_length[0]
            text += self.printName().rjust(width) if width else self.printName()
        if contents & kArgs:
            text += self.printArgs()
        if contents & kValue:
            text += (" = " if contents & kName else "") + self.printValue()
        return text + self._tail(contents)

    def _tail(self, contents: int) -> str:
        text = ""
        if contents & kExtras:
            text += (" " if contents != kExtras else "") + self.printExtras()
        if contents & kTitle:
            title = self.printTitle()
            text += title if contents == kTitle else f' "{title}"'
        return text

    def Print(self, option: str = "") -> None:
        """Print to standard output as ROOT's ``Print(option)`` does."""
        cout.write(
            self.printStream(self.defaultPrintContents(option), self.defaultPrintStyle(option))
        )

    def __str__(self) -> str:
        return self.printStream(self.defaultPrintContents("I"), kInline)

    def cxx_ostream(self) -> str:
        """``std::cout << obj``: RooFit's ``operator<<``, the inline print - ``(a,b,c)`` for a
        set, whatever Python's ``print`` shows of it."""
        return RooPrintable.__str__(self)
