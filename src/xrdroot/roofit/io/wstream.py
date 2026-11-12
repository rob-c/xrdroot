"""RooFit's object identities are mapped before their members can point back at them."""

from __future__ import annotations

import struct
from typing import Any

from ...writer import WBuffer
from .layouts import LAYOUTS
from .stream import BASIC, NAMED, Streamed, _clean, _template


class Writer:
    """One key's object graph, counting references from the key's header."""

    def __init__(self, keylen: int) -> None:
        self.buf = WBuffer()
        self.keylen = keylen
        self.objects: dict[int, int] = {}

    def number(self, letter: str, value: Any) -> None:
        self.buf.raw(struct.pack(">" + letter, value))

    def tobject(self, value: dict[str, int]) -> None:
        self.buf.u16(1)
        self.buf.u32(value.get("fUniqueID", 0))
        bits = value.get("fBits", 0)
        self.buf.u32(bits)
        if bits & (1 << 4):
            self.buf.u16(value.get("pid", 0))

    def tnamed(self, value: dict[str, Any]) -> None:
        at = self.buf.start(1)
        self.tobject(value.get("TObject", {}))
        self.buf.string(value["fName"])
        self.buf.string(value["fTitle"])
        self.buf.end(at)

    def pointer(self, value: Streamed | None) -> None:
        if value is None:
            self.buf.u32(0)
        elif id(value) in self.objects:
            self.buf.u32(self.objects[id(value)])
        else:
            at = self.buf.tag(value.cls)
            self.objects[id(value)] = at + self.keylen + 2
            self.object(value, value.cls)
            self.buf.end(at)

    def object(self, value: Streamed, classname: str) -> None:
        from .wcustom import CUSTOM, linked

        if classname == "RooLinkedList":
            linked(self, value)
            return
        at = self.buf.start(LAYOUTS[classname][0])
        custom = CUSTOM.get(classname)
        if custom is not None:
            custom(self, value)
        else:
            for name, kind, typename, _length in LAYOUTS[classname][1]:
                self.member(name, kind, typename, value.m[name])
        self.buf.end(at)

    def member(self, name: str, kind: int, typename: str, value: Any) -> None:
        if kind == 0:
            self.object(value, name)
        elif kind == 66:
            self.tobject(value)
        elif kind == 67:
            self.tnamed(value)
        elif kind in BASIC:
            self.number(BASIC[kind], value)
        elif kind - 20 in BASIC:
            for number in value:
                self.number(BASIC[kind - 20], number)
        else:
            self._held_member(kind, typename, value)

    def _held_member(self, kind: int, typename: str, value: Any) -> None:
        if kind == 65:
            self.buf.string(value)
        elif kind in (61, 62, 63):
            self.object(value, typename.rstrip("*"))
        elif kind in (64, 69):
            self.pointer(value)
        else:
            self.container(typename, value)

    def container(self, typename: str, value: Any) -> None:
        text = _clean(typename)
        at = self.buf.start(10)
        if text in ("string", "TString"):
            self.buf.string(value)
        else:
            head, args = _template(text)
            self.buf.u32(len(value))
            if head in ("map", "unordered_map", "multimap"):
                for key, item in value.items():
                    self.one(args[0], key)
                    self.one(args[1], item)
            else:
                for item in value:
                    self.one(args[0], item)
        self.buf.end(at)

    def one(self, typename: str, value: Any) -> None:
        if typename in NAMED:
            self.number(NAMED[typename], value)
        elif typename in ("string", "TString"):
            self.buf.string(value)
        elif typename.endswith("*"):
            self.pointer(value)
        else:
            self.object(value, typename)
