"""``TBuffer`` and ``TBufferFile``: objects streamed into bytes in memory, and back.

A ``TBufferFile`` is the buffer ROOT streams an object into before it goes into
a file - and that a program may use on its own, writing objects of its classes
with ``WriteObjectAny`` or ``TClass::Streamer`` and reading them back. The
bytes are a file's: an object's record and members, as :mod:`xrdroot.wdobjects`
writes a key's, so what is read back is read by the same descriptions - and,
as ROOT does, into the class asked for, ``vector<int>`` written and
``vector<float>`` read.
"""

from __future__ import annotations

import struct
from typing import Any

from ...buffer import Buffer as Cursor
from ...rdobjects import flat_reader_for, reader_for
from ...wdeclared import Catalogue, parse_type
from ...wdobjects import write_value
from ...writer import WBuffer
from .declared import (
    _plan,
    catalogue_for,
    converted,
    flat_bytes,
    new_value,
    plain_value,
)
from .objects import TObject

__all__ = ["TBuffer", "TBufferFile"]

#: Each fundamental type a buffer reads and writes on its own, by the name its methods carry.
BASIC = {"Bool": ">?", "Char": ">b", "UChar": ">B", "Short": ">h", "UShort": ">H", "Int": ">i",
         "UInt": ">I", "Long": ">q", "ULong": ">Q", "Long64": ">q", "ULong64": ">Q",
         "Float": ">f", "Double": ">d"}  # fmt: skip


def _type_of(cl: Any) -> str:
    """The class a ``TClass`` - or a name - stands for."""
    return str(cl.GetName() if hasattr(cl, "GetName") else cl)


class TBuffer(TObject):
    """``TBuffer``: bytes an object is streamed into or out of, and which way."""

    kRead, kWrite = 0, 1
    kInitialSize = 1024

    def __init__(self, mode: int = 1, bufsize: int = 1024, *rest: Any) -> None:
        super().__init__()
        self._mode = int(mode)
        self._data = bytearray()
        self._pos = 0

    # -- which way, and where ----------------------------------------------------------

    def IsReading(self) -> bool:
        return self._mode == self.kRead

    def IsWriting(self) -> bool:
        return self._mode == self.kWrite

    def SetReadMode(self) -> None:
        self._mode = self.kRead

    def SetWriteMode(self) -> None:
        self._mode = self.kWrite

    def Reset(self) -> None:
        """``Reset``: back to the start - what is written next overwrites what was there."""
        self._pos = 0

    def Length(self) -> int:
        return self._pos

    def BufferSize(self) -> int:
        return max(len(self._data), self.kInitialSize)

    def Buffer(self) -> bytes:
        return bytes(self._data)

    def SetBufferOffset(self, offset: int = 0) -> None:
        self._pos = int(offset)

    def GetBufferOffset(self) -> int:
        """Not ROOT's name, but what ``Length`` is: how far in the next byte goes."""
        return self._pos

    # -- bytes -------------------------------------------------------------------------

    def _put(self, chunk: bytes) -> None:
        """Bytes written at the cursor, over anything there and past the end."""
        self._data[self._pos : self._pos + len(chunk)] = chunk
        self._pos += len(chunk)

    def _cursor(self) -> Cursor:
        buf = Cursor(bytes(self._data))
        buf.pos = self._pos
        return buf

    def _write(self, form: str, value: Any) -> None:
        self._put(struct.pack(form, value))

    def _read(self, form: str) -> Any:
        size = struct.calcsize(form)
        found = struct.unpack_from(form, self._data, self._pos)[0]
        self._pos += size
        return found

    def __getattr__(self, name: str) -> Any:
        """``WriteInt(i)``, ``ReadDouble()`` and the like, for each fundamental type."""
        written = BASIC.get(name[5:]) if name.startswith("Write") else None
        if written is not None:
            return lambda value: self._write(written, value)
        read = BASIC.get(name[4:]) if name.startswith("Read") else None
        if read is not None:
            return lambda: self._read(read)
        raise AttributeError(f"ROOT's TBuffer has {name}, or does not; xrdroot.pyroot's does not")

    def WriteString(self, text: Any) -> None:
        raw = str(text).encode("utf-8", "surrogateescape")
        self._put(bytes([len(raw)]) if len(raw) < 255 else b"\xff" + struct.pack(">i", len(raw)))
        self._put(raw)

    def ReadString(self, *args: Any) -> str:
        buf = self._cursor()
        found = str(buf.string())
        self._pos = buf.pos
        return found

    WriteTString = WriteString
    ReadTString = ReadString

    # -- objects -------------------------------------------------------------------------

    def StreamObject(self, obj: Any, cl: Any) -> Any:
        """``StreamObject``: ``obj`` written, or read into, by the way the buffer goes."""
        name = _type_of(cl)
        if self.IsWriting():
            self._put(_written(obj, name, tagged=False))
            return obj
        return self._read_value(name, name, obj)

    def WriteObjectAny(self, obj: Any, cl: Any, cacheReuse: bool = True) -> int:
        """``WriteObjectAny``: ``obj`` as an object naming its class ``cl`` - a null pointer as
        the zero that says there is none."""
        self._put(_written(obj, _type_of(cl), tagged=True) if obj is not None else bytes(4))
        return 1

    def WriteObject(self, obj: Any, cacheReuse: bool = True) -> None:
        self.WriteObjectAny(obj, obj.ClassName())

    def ReadObjectAny(self, cl: Any) -> Any:
        """``ReadObjectAny``: the object written here, made into an object of ``cl``."""
        buf = self._cursor()
        if buf.u32() == 0:  # a null pointer: the zero alone
            self._pos = buf.pos
            return None
        buf.u32()  # the new-class tag, the class's name after it
        written = buf.cstring()
        self._pos = buf.pos
        return self._read_value(written, _type_of(cl), None)

    def ReadObject(self, cl: Any) -> Any:
        return self.ReadObjectAny(cl)

    def _read_value(self, written: str, wanted: str, into: Any) -> Any:
        """What was written as a ``written``, read into ``into`` - or a new ``wanted`` - as
        a member of that type is read, converted as schema evolution converts it."""
        buf = self._cursor()
        catalogue: Catalogue = catalogue_for(written)
        if into is not None and written == wanted:
            flat = _read_flat(buf, written, catalogue, into)
            if flat is not None:
                self._pos = buf.pos
                return flat  # a class of plain numbers, read straight into the object
        value = reader_for(written, catalogue)(buf)
        self._pos = buf.pos
        target = into if into is not None else new_value(wanted)
        return converted(value, parse_type(wanted), target)


class TBufferFile(TBuffer):
    """``TBufferFile``: the buffer ROOT streams objects into on their way to a file."""


def _read_flat(buf: Cursor, classname: str, catalogue: Catalogue, into: Any) -> Any:
    """``into`` given the record that reads as one struct of plain numbers - or ``None`` when
    the class is more than that, a class the catalogue does not declare, or a file whose
    members are not the ones the class declares now.

    The bulk loop of roottest's ``WriteBuffer`` reads ten million records of nine numbers,
    and each of them built a dictionary of them for the object to be given from.
    """
    if catalogue.declared(classname) is None:
        return None
    made = flat_reader_for(classname, catalogue)
    if made is None:
        return None
    read, labels = made
    plan = _plan(catalogue, classname)
    if plan.flat is None or tuple(field.name for field, _at, _found in plan.fields) != labels:
        return None  # a class that is more than numbers, or a file that has evolved
    for attribute, value in zip(plan.flat, read(buf), strict=True):
        setattr(into, attribute, value)
    return into


def _written(obj: Any, classname: str, tagged: bool) -> bytes:
    """The bytes of ``obj`` as a ``classname`` - naming its class first when ``tagged``."""
    catalogue: Catalogue = catalogue_for(classname)
    if not tagged:
        flat = flat_bytes(obj, classname, catalogue)
        if flat is not None:
            return flat  # a class of plain numbers, written straight from the object
    out = WBuffer()
    at = out.tag(classname) if tagged else None
    write_value(out, classname, plain_value(obj, classname, catalogue), catalogue)
    if at is not None:
        out.end(at)
    return bytes(out.data)
