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

from functools import cache
from typing import Any, NamedTuple, cast

from .errors import UnsupportedFeatureError
from .winfo import INFOS, Element

__all__ = ["TREF", "Layout", "Member", "checksum", "declared", "harvested"]

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


class Base(NamedTuple):
    """A base class: as one of the donors describes the element that names it - the
    streamer type of the base itself, its title, its checksum and its version."""

    name: str
    stype: int
    title: str
    checksum: int
    version: int


class Layout(NamedTuple):
    """One class as a file describes it: name, version, checksum, members and bases.

    A macro's class that derives from one of ROOT's carries that base first in
    what the file says of it, so a ``TRef`` member's owner - the class that
    declares it - is named for each member and not only for the class.
    """

    name: str
    version: int
    checksum: int
    members: tuple[Member, ...]
    bases: tuple[Base, ...] = ()

    def elements(self) -> tuple[Element, ...]:
        """The ``TStreamerElement`` of each base and member, as :data:`~.winfo.INFOS` holds
        them: a base's own checksum and version, a member's streamer type and title."""
        bases = tuple(
            ("TStreamerBase", b.name, b.title, b.stype, 0, 0, 0, (0, b.checksum, 0, 0, 0), "BASE",
             (b.version,))
            for b in self.bases
        )  # fmt: skip
        kinds = {OBJECT: "TStreamerObject"}
        return bases + tuple(
            (kinds.get(m.stype, "TStreamerBasicType"), m.name, m.title, m.stype, m.size, 0, 0,
             (0,) * 5, m.typename, ())
            for m in self.members
        )  # fmt: skip


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


#: ``kObject``: a member that is a whole object of its own instead of a number, and how many
#: bytes each class of such a member takes in memory - ``sizeof(TRef)``, as ROOT writes it.
OBJECT = 61
OBJECT_TYPES: dict[str, int] = {"TRef": 24}


@cache
def _bases() -> dict[str, Base]:
    """ROOT's base classes, as the donors describe the elements naming them."""
    found: dict[str, Base] = {}
    for _checksum, _version, elements in INFOS.values():
        for element in elements:
            kind, name, title, stype, _size, _a, _d, maxima, _t, extra = element
            if kind == "TStreamerBase" and name not in found:
                found[name] = _element_base(name, stype, title, maxima, extra)
    return found


def _element_base(name: str, stype: int, title: str, maxima: Any, extra: Any) -> Base:
    """The base an element names: its checksum is the second maximum, its version the extra."""
    version = cast("tuple[int, ...]", extra)
    return Base(name, stype, title, cast("int", maxima[1]), version[0] if version else 1)


def _base(name: str) -> Base:
    """The base ``name``, which has to be one of ROOT's for a macro's class to be split."""
    found = _bases().get(name)
    if found is None:
        raise UnsupportedFeatureError(
            f"{name} is a base class this writer has no description of, so a macro's class "
            f"deriving from it cannot be described member by member; declare the class "
            f"without that base"
        )
    return found


def _tref() -> Layout:
    """ROOT's description of ``TRef``, its checksum worked out from the base as ROOT does."""
    found = Layout("TRef", 1, 0, (), (_base("TObject"),))
    return found._replace(checksum=checksum("TRef", found.elements()))


#: ``TRef`` as a file describes it, which is as ROOT's own does: no member of its own - the
#: process id ``fPID`` is transient, and the id of the object referred to travels in
#: ``TObject``'s ``fUniqueID`` - so nothing but the ``TObject`` base. A whole-object member
#: of a split collection is written into a tree entry as this class's record, and ROOT reads
#: that record back by this description; ``roottest``'s ``io/references/lotsRef`` holds one.
#: The version and the base are ROOT 6.40's, and :func:`checksum` gives back ROOT's
#: ``0x91757901`` for them, which is what proves the description is the one ROOT writes.
TREF: Layout = _tref()


def declared(cls: Any) -> Layout:
    """A class a macro declared, from what its translation says of its members."""
    name, bases, fields = cls._cxx_layout_
    members = tuple(_member(name, *field) for field in fields)
    version_of = getattr(cls, "Class_Version", None)
    version = int(version_of()) if callable(version_of) else 1
    made = Layout(name, version, 0, members, tuple(_base(base) for base in bases))
    return made._replace(checksum=checksum(name, made.elements()))


def _member(owner: str, name: str, spelled: str, title: str, dims: tuple[Any, ...]) -> Member:
    found = TYPES.get(spelled)
    if dims or (found is None and spelled not in OBJECT_TYPES):
        what = f"{spelled}[{']['.join(map(str, dims))}]" if dims else spelled
        raise UnsupportedFeatureError(
            f"{owner}::{name} is a {what}, and a macro's class is written to a tree here "
            f"when each of its members is a single number; keep this one out of the class "
            f"the branch holds"
        )
    if found is None:
        return Member(name, OBJECT, spelled, title, OBJECT_TYPES[spelled])
    stype, typename, size = found
    return Member(name, stype, typename, title, size)
