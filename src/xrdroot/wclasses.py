"""The classes a tree's objects are written as: their members, version and checksum.

A branch of objects is only readable by what the file says the class looks
like - its ``TStreamerInfo``, member by member - and a split branch names
each member's place in that description. A class ROOT has is described as
the donor files under ``tests/data`` describe it (:data:`~.winfo.INFOS`);
a class a macro declares is described from its declaration, as ROOT's
interpreter would: each data member in order, its type spelled as written
(``Double32_t``, not the ``double`` it is in memory), its trailing comment
as its title, and the class's version from its ``ClassDef`` - 1 without one.

The checksum is ``TClass::GetCheckSum``'s, worked out from the elements:
the class name, then each base's name and checksum, then each member's
name, type and array dimensions, and whatever its title holds between its
first brackets - which reproduces every checksum the donors carry.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .errors import UnsupportedFeatureError
from .winfo import INFOS, Element

__all__ = ["Layout", "Member", "checksum", "declared", "harvested"]

#: The integer types ``TClass::GetCheckSum`` does not count as an enum.
INTEGERS = ("int", "Int_t", "unsigned int", "UInt_t")

#: Each plain C++ type a member can be, as written: its streamer type, the name
#: the streamer spells it with, and how many bytes it takes in memory.
TYPES: dict[str, tuple[int, str, int]] = {
    **dict.fromkeys(("double", "Double_t"), (8, "double", 8)),
    **dict.fromkeys(("float", "Float_t"), (5, "float", 4)),
    **dict.fromkeys(("int", "Int_t"), (3, "int", 4)),
    **dict.fromkeys(("unsigned int", "UInt_t"), (13, "unsigned int", 4)),
    **dict.fromkeys(("short", "Short_t"), (2, "short", 2)),
    **dict.fromkeys(("unsigned short", "UShort_t"), (12, "unsigned short", 2)),
    **dict.fromkeys(("char", "Char_t"), (1, "char", 1)),
    **dict.fromkeys(("unsigned char", "UChar_t"), (11, "unsigned char", 1)),
    **dict.fromkeys(("bool", "Bool_t"), (18, "bool", 1)),
    **dict.fromkeys(("long", "Long_t"), (4, "long", 8)),
    **dict.fromkeys(("unsigned long", "ULong_t"), (14, "unsigned long", 8)),
    **dict.fromkeys(("long long", "Long64_t"), (16, "Long64_t", 8)),
    **dict.fromkeys(("unsigned long long", "ULong64_t"), (17, "ULong64_t", 8)),
    "Double32_t": (9, "Double32_t", 8),
    "Float16_t": (19, "Float16_t", 4),
}


class Member(NamedTuple):
    """One data member: its name, streamer type and type name, title and size."""

    name: str
    stype: int
    typename: str
    title: str
    size: int


class Layout(NamedTuple):
    """One class as a file describes it: name, version, checksum and members."""

    name: str
    version: int
    checksum: int
    members: tuple[Member, ...]

    def elements(self) -> tuple[Element, ...]:
        """The ``TStreamerElement`` of each member, as :data:`~.winfo.INFOS` holds them."""
        return tuple(
            ("TStreamerBasicType", m.name, m.title, m.stype, m.size, 0, 0, (0,) * 5, m.typename, ())
            for m in self.members
        )


def checksum(name: str, elements: tuple[Element, ...]) -> int:
    """``TClass::GetCheckSum`` over a class's streamer elements."""
    found = _mixed(0, name)
    for kind, member, title, stype, _size, _length, dims, maxima, typename, _extra in elements:
        if kind == "TStreamerBase":
            found = (_mixed(found, member) * 3 + (maxima[1] & 0xFFFFFFFF)) & 0xFFFFFFFF
            continue
        if stype == 3 and typename not in INTEGERS:  # an enum counts once more before it
            found = (found * 3 + 1) & 0xFFFFFFFF
        found = _mixed(_mixed(found, member), typename)
        for dimension in maxima[:dims]:
            found = (found * 3 + dimension) & 0xFFFFFFFF
        found = _mixed(found, _bracketed(title))
    return found


def _mixed(found: int, text: str) -> int:
    for byte in text.encode():
        found = (found * 3 + byte) & 0xFFFFFFFF
    return found


def _bracketed(title: str) -> str:
    """What a title holds between its first ``[`` and the ``]`` after it, or nothing."""
    left = title.find("[")
    right = title.find("]", left) if left >= 0 else -1
    return title[left + 1 : right] if right >= 0 else ""


#: The elements a class can be split by here: numbers, and objects held by value.
SPLITTABLE = ("TStreamerBasicType", "TStreamerObjectAny")


def harvested(name: str) -> Layout | None:
    """A class ROOT has, as the donor files describe it - if each member is a number or an
    object held by value."""
    found = INFOS.get(name)
    if found is None:
        return None
    stored, version, elements = found
    if any(element[0] not in SPLITTABLE for element in elements):
        return None
    members = tuple(Member(e[1], e[3], e[8], e[2], e[4]) for e in elements)
    return Layout(name, version, stored, members)


def declared(cls: Any) -> Layout:
    """A class a macro declared, from what its translation says of its members."""
    name, bases, fields = cls._cxx_layout_
    if bases:
        raise UnsupportedFeatureError(
            f"{name} derives from {', '.join(bases)}, and a tree writes a macro's class "
            f"member by member only when it stands alone; give the branch a class of "
            f"plain members"
        )
    members = tuple(_member(name, *field) for field in fields)
    version_of = getattr(cls, "Class_Version", None)
    version = int(version_of()) if callable(version_of) else 1
    made = Layout(name, version, 0, members)
    return made._replace(checksum=checksum(name, made.elements()))


def _member(owner: str, name: str, spelled: str, title: str, dims: tuple[Any, ...]) -> Member:
    found = TYPES.get(spelled)
    if found is None or dims:
        what = f"{spelled}[{']['.join(map(str, dims))}]" if dims else spelled
        raise UnsupportedFeatureError(
            f"{owner}::{name} is a {what}, and a macro's class is written to a tree here "
            f"when each of its members is a single number; keep this one out of the class "
            f"the branch holds"
        )
    stype, typename, size = found
    return Member(name, stype, typename, title, size)
