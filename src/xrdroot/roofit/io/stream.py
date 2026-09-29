"""The bytes of a RooFit object graph, walked by the file's streamer information.

Every object is a :class:`Streamed` record: its class and its members by
name, each base class's members under the base's name. A pointer to an
object is the object itself, read where it first appears and the same
Python object wherever it appears again - even inside itself, since a
node's clients point back at it - because it is registered before its
members are read, as ROOT registers it. The classes whose streamers RooFit
writes by hand are read by hand (:data:`CUSTOM`, filled in by
:mod:`.custom`); every other is read member by member as the file
describes it, and a class the file does not describe is refused by name.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import Any

from ...buffer import Buffer
from ...errors import FormatError, UnsupportedFeatureError

__all__ = ["CUSTOM", "Reader", "Streamed"]

#: ``kMapOffset``, ``kNewClassTag``, ``kClassMask``, ``kByteCountMask``.
MAP_OFFSET, NEW_CLASS, CLASS_MASK, BYTE_COUNT = 2, 0xFFFFFFFF, 0x80000000, 0x40000000
#: A container written field by field rather than element by element.
MEMBER_WISE = 0x4000
#: The fundamental types by streamer number: their struct letter.
BASIC = {1: "b", 2: "h", 3: "i", 4: "q", 5: "f", 6: "i", 8: "d", 9: "f", 11: "B", 12: "H",
         13: "I", 14: "Q", 15: "I", 16: "q", 17: "Q", 18: "?", 19: "f"}  # fmt: skip
#: The fundamental types by C++ name, as a container's template names them.
NAMED = {"char": "b", "short": "h", "int": "i", "long": "q", "float": "f", "double": "d",
         "unsigned char": "B", "unsigned short": "H", "unsigned int": "I", "unsigned long": "Q",
         "Long64_t": "q", "ULong64_t": "Q", "bool": "?", "Int_t": "i", "Double_t": "d",
         "UInt_t": "I", "Bool_t": "?", "Float_t": "f", "long long": "q"}  # fmt: skip
#: A reader by hand, for a class whose streamer RooFit writes itself: ``(reader, class)``.
CUSTOM: dict[str, Callable[[Reader, str], Any]] = {}


class Streamed:
    """One object as the file holds it: its class, and its members by name."""

    __slots__ = ("cls", "m")

    def __init__(self, cls: str) -> None:
        self.cls = cls
        self.m: dict[str, Any] = {}

    def get(self, name: str, default: Any = None) -> Any:
        """A member of this class or - searched in declaration order - of one of its bases."""
        if name in self.m:
            return self.m[name]
        for value in self.m.values():
            if isinstance(value, Streamed) and value.cls == "":  # a base's members
                found = value.get(name, _MISSING)
                if found is not _MISSING:
                    return found
        return default

    def __repr__(self) -> str:
        return f"<Streamed {self.cls} {self.get('fName', '')!r}>"


_MISSING = object()


def base(cls: str) -> Streamed:
    """A base class's members, kept as a record of no class of its own."""
    made = Streamed("")
    made.m["__class__"] = cls
    return made


class Reader:
    """A cursor over one key's bytes, reading RooFit's objects by the file's descriptions."""

    def __init__(self, buf: Buffer, infos: dict[str, Any]) -> None:
        self.buf = buf
        self.infos = infos

    # -- fundamentals -------------------------------------------------------------

    def number(self, letter: str) -> Any:
        """One number, by its ``struct`` letter, big-endian as ROOT writes it."""
        form = struct.Struct(">" + letter)
        return form.unpack_from(self.buf.data, self.buf._span(form.size))[0]

    def numbers(self, letter: str, count: int) -> list[Any]:
        form = struct.Struct(f">{count}{letter}")
        return list(form.unpack_from(self.buf.data, self.buf._span(form.size)))

    def tobject(self) -> dict[str, int]:
        unique, bits = self.buf.tobject()
        return {"fUniqueID": unique, "fBits": bits}

    def tnamed(self) -> dict[str, Any]:
        name, title = self.buf.named()
        return {"fName": name, "fTitle": title}

    # -- objects ------------------------------------------------------------------

    def pointer(self) -> Any:
        """``ReadObjectAny``: ``None``, an object read before, or one written here."""
        buf = self.buf
        start = buf.pos
        count = buf.u32()
        if count & BYTE_COUNT and count != NEW_CLASS:
            end: int | None = start + 4 + (count & ~BYTE_COUNT)
            after, tag = buf.pos, buf.u32()
        else:
            end, after, tag = None, buf.pos, count
        if not tag & CLASS_MASK:
            return None if tag == 0 else buf.refs.get(tag)
        if tag == NEW_CLASS:
            name = buf.cstring()
            buf.refs[after + MAP_OFFSET] = name
        else:
            name = str(buf.refs.get(tag & ~CLASS_MASK, ""))
            if not name:
                raise FormatError(f"a class reference at {start} points nowhere")
        made = Streamed(name)
        buf.refs[start + MAP_OFFSET] = made  # before its members: they may point back at it
        self.fill(made, name)
        if end is not None and buf.pos != end:
            raise FormatError(f"a {name} at {start} ended {end - buf.pos} bytes short of its end")
        return made

    def fill(self, made: Streamed, cls: str) -> Streamed:
        """``cls``'s streamer: by hand for RooFit's own, else its record and members."""
        custom = CUSTOM.get(cls)
        if custom is not None:
            custom(self, made)
            return made
        version, end = self.buf.header()
        if version == 0:
            self.buf.u32()  # a class with no version of its own writes a checksum instead
        self.members(made, cls)
        self.buf.resume(end)
        return made

    def object(self, cls: str) -> Streamed:
        """An object of ``cls`` written where it stands, by its own streamer."""
        return self.fill(Streamed(cls), cls)

    def members(self, made: Streamed, cls: str) -> None:
        """Every member ``cls`` is declared with, in order, into ``made``."""
        described = self.infos.get(cls)
        if described is None:
            raise UnsupportedFeatureError(
                f"this file does not describe the class {cls}, so an object of it cannot be "
                "read member by member"
            )
        for member in described.values():
            made.m[member.name] = self.member(member, made)

    def member(self, member: Any, made: Streamed) -> Any:
        """One member, by its streamer type - or ``None`` for a kind this does not read."""
        kind = member.stype
        if kind == 0:
            return self.fill(base(member.name), member.name)
        if kind == 66:
            return self.tobject()
        if kind == 67:
            return self.tnamed()
        if kind in BASIC:
            return self.number(BASIC[kind])
        if kind - 20 in BASIC:
            return self.numbers(BASIC[kind - 20], member.length)
        if kind - 40 in BASIC:  # x[n]: a marker byte, then the n its counter member holds
            count = int(made.get(member.count, 0) or 0)
            return self.numbers(BASIC[kind - 40], count) if self.buf.u8() else []
        if kind == 65:
            return self.buf.string()
        if kind in (61, 62, 63):  # an object held by value, or by a pointer never null
            return self.object(member.typename.rstrip("*"))
        if kind in (64, 69):
            return self.pointer()
        if kind in (500, 300):
            return self.container(member.typename)
        raise UnsupportedFeatureError(
            f"the member {member.name} of type {member.typename} is streamed as a kind "
            f"({kind}) this reader of RooFit's objects does not decode"
        )

    # -- containers ---------------------------------------------------------------

    def container(self, typename: str) -> Any:
        """A standard container - or string - member: its record, then its elements."""
        text = _clean(typename)
        if text in ("string", "TString"):
            _version, end = self.buf.header()
            found = self.buf.string()
            self.buf.resume(end)
            return found
        head, args = _template(text)
        version, end = self.buf.header()
        if head in ("vector", "list", "deque", "set", "unordered_set", "multiset"):
            found: Any = self._elements(args[0], self.buf.u32(), version)
        elif head in ("map", "unordered_map", "multimap"):
            found = self._mapping(args[0], args[1], version)
        else:
            raise UnsupportedFeatureError(f"a member of type {typename} is not a container "
                                          "this reader of RooFit's objects decodes")  # fmt: skip
        self.buf.resume(end)
        return found

    def _elements(self, item: str, count: int, version: int) -> list[Any]:
        if item in NAMED:
            return self.numbers(NAMED[item], count)
        if item in ("string", "TString"):
            return [self.buf.string() for _ in range(count)]
        if item.endswith("*"):
            return [self.pointer() for _ in range(count)]
        if version & MEMBER_WISE:
            raise UnsupportedFeatureError(f"a container of {item} written field by field is "
                                          "not a shape this reader decodes")  # fmt: skip
        return [self.object(item) for _ in range(count)]

    def _one(self, item: str) -> Any:
        if item in NAMED:
            return self.number(NAMED[item])
        if item in ("string", "TString"):
            return self.buf.string()
        if item.endswith("*"):
            return self.pointer()
        return self.object(item)

    def _mapping(self, key: str, value: str, version: int) -> dict[Any, Any]:
        if version & MEMBER_WISE:
            if self.buf.i16() <= 0:
                self.buf.u32()
            count = self.buf.u32()
            if not count:
                return {}
            keys = self._block(key, count)
            return dict(zip(keys, self._block(value, count)))
        count = self.buf.u32()
        pairs = [(self._one(key), self._one(value)) for _ in range(count)]
        return dict(pairs)


    def _block(self, item: str, count: int) -> list[Any]:
        """One member of every pair of a container written field by field: strings in a record
        of their own, numbers as they are, objects one after another."""
        if item in ("string", "TString"):
            _version, end = self.buf.header()
            found = [self.buf.string() for _ in range(count)]
            self.buf.resume(end)
            return found
        return [self._one(item) for _ in range(count)]


def _clean(typename: str) -> str:
    return typename.replace("std::", "").replace("const ", "").strip()


def _template(text: str) -> tuple[str, list[str]]:
    """``map<string,RooArgSet>`` as ``("map", ["string", "RooArgSet"])``."""
    head, _, rest = text.partition("<")
    inner = rest.rsplit(">", 1)[0]
    args, depth, current = [], 0, ""
    for char in inner:
        depth += {"<": 1, ">": -1}.get(char, 0)
        if char == "," and depth == 0:
            args.append(current.strip())
            current = ""
            continue
        current += char
    args.append(current.strip())
    return head.strip(), args
