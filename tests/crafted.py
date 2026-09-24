"""Small ROOT files made byte by byte, for classes no file in ``tests/data`` holds.

No ROOT file under ``tests/data`` holds a ``TEntryList``, a ``TEventList``, a
``THnSparse``, a ``THStack`` or a ``TProfile3D``, and without ROOT there is no
way to make one that did. So these are assembled here: a header, a directory,
one key per object, and a streamer information record describing every class
written - the classes of ROOT's own kit as ``gauss-h1.root`` describes them,
and the rest from the members their C++ headers declare, member for member.

The objects are then written by walking those same descriptions, which is
the way ROOT writes any class it has no streamer of its own for. That makes
these files a test of reading a layout faithfully rather than of the layout
itself, and that is said wherever they are used.
"""

from __future__ import annotations

import pathlib
import struct
from typing import Any

import numpy as np

from xrdroot import open_root
from xrdroot.buffer import BYTE_COUNT_MASK, NEW_CLASS_TAG
from xrdroot.interp import ARRAYS, BASIC
from xrdroot.streamers import Member

DATA = pathlib.Path(__file__).parent / "data"

#: The file whose description of ROOT's own classes the crafted files carry.
DONOR = DATA / "gauss-h1.root"


def member(name: str, stype: int, typename: str, *, count: str = "", length: int = 1) -> Member:
    return Member(name, "", stype, typename, length, count)


def base(name: str, stype: int = 0) -> Member:
    return member(name, stype, "BASE")


#: The classes the crafted files hold, as ROOT's headers declare their
#: persistent members, in declaration order.
LAYOUTS: dict[str, list[Member]] = {
    "TEntryList": [
        base("TNamed", 67),
        member("fLists", 64, "TList*"),
        member("fNBlocks", 3, "int"),
        member("fBlocks", 64, "TObjArray*"),
        member("fN", 16, "Long64_t"),
        member("fEntriesToProcess", 16, "Long64_t"),
        member("fTreeName", 65, "TString"),
        member("fFileName", 65, "TString"),
        member("fReapply", 18, "bool"),
    ],
    "TEntryListBlock": [
        base("TObject", 66),
        member("fNPassed", 3, "int"),
        member("fN", 6, "int"),
        member("fIndices", 52, "UShort_t*", count="fN"),
        member("fType", 3, "int"),
        member("fPassing", 18, "bool"),
    ],
    "TEventList": [
        base("TNamed", 67),
        member("fN", 6, "int"),
        member("fSize", 3, "int"),
        member("fDelta", 3, "int"),
        member("fReapply", 18, "bool"),
        member("fList", 56, "Long64_t*", count="fN"),
    ],
    "THStack": [
        base("TNamed", 67),
        member("fHists", 64, "TList*"),
        member("fHistogram", 64, "TH1*"),
        member("fMaximum", 8, "double"),
        member("fMinimum", 8, "double"),
    ],
    "THnBase": [
        base("TNamed", 67),
        member("fNdimensions", 3, "int"),
        member("fAxes", 61, "TObjArray"),
        member("fEntries", 8, "double"),
        member("fTsumw", 8, "double"),
        member("fTsumw2", 8, "double"),
        member("fTsumwx", 62, "TArrayD"),
        member("fTsumwx2", 62, "TArrayD"),
    ],
    "THnSparse": [
        base("THnBase"),
        member("fChunkSize", 3, "int"),
        member("fFilledBins", 16, "Long64_t"),
        member("fBinContent", 61, "TObjArray"),
    ],
    "THnSparseT<TArrayD>": [base("THnSparse")],
    "THnSparseArrayChunk": [
        base("TObject", 66),
        member("fSingleCoordinateSize", 3, "int"),
        member("fCoordinatesSize", 6, "int"),
        member("fCoordinates", 41, "char*", count="fCoordinatesSize"),
        member("fContent", 64, "TArray*"),
        member("fSumw2", 64, "TArrayD*"),
    ],
    "TProfile3D": [
        base("TH3D"),
        member("fBinEntries", 62, "TArrayD"),
        member("fErrorMode", 3, "EErrorType"),
        member("fTmin", 8, "double"),
        member("fTmax", 8, "double"),
        member("fTsumwt", 8, "double"),
        member("fTsumwt2", 8, "double"),
        member("fBinSumw2", 62, "TArrayD"),
    ],
    "TH3D": [base("TH3"), base("TArrayD")],
    "TH3": [
        base("TH1"),
        base("TAtt3D"),
        member("fTsumwy", 8, "double"),
        member("fTsumwy2", 8, "double"),
        member("fTsumwxy", 8, "double"),
        member("fTsumwz", 8, "double"),
        member("fTsumwz2", 8, "double"),
        member("fTsumwxz", 8, "double"),
        member("fTsumwyz", 8, "double"),
    ],
    "TAtt3D": [],
}


class Out:
    """Bytes being written, with ROOT's byte counts filled in afterwards."""

    def __init__(self) -> None:
        self.data = bytearray()

    def pack(self, form: str, *values: Any) -> None:
        self.data += struct.pack(">" + form, *values)

    def string(self, text: str) -> None:
        raw = text.encode()
        self.pack("B", len(raw))
        self.data += raw

    def start(self, version: int) -> int:
        at = len(self.data)
        self.pack("IH", 0, version)
        return at

    def tag(self, classname: str) -> int:
        at = len(self.data)
        self.pack("II", 0, NEW_CLASS_TAG)
        self.data += classname.encode() + b"\x00"
        return at

    def end(self, at: int) -> None:
        struct.pack_into(">I", self.data, at, (len(self.data) - at - 4) | BYTE_COUNT_MASK)

    def tobject(self, value: Any = None) -> None:
        row = value if isinstance(value, dict) else {}
        self.pack("HII", 1, row.get("fUniqueID", 0), row.get("fBits", 0x03000000) & ~(1 << 4))


class Writer:
    """Objects written the way their descriptions say, which is how ROOT writes most."""

    def __init__(self, layouts: dict[str, dict[str, Member]]) -> None:
        self.layouts = layouts

    def record(self, out: Out, classname: str, row: dict[str, Any]) -> None:
        at = out.start(1)
        for one in self.layouts[classname].values():
            self.member(out, one, row)
        out.end(at)

    def member(self, out: Out, one: Member, row: dict[str, Any]) -> None:
        value = row.get(one.name)
        if one.stype == 66:
            out.tobject(value)
        elif one.stype == 67:
            named = value or {}
            at = out.start(1)
            out.tobject(named)
            out.string(named.get("fName", ""))
            out.string(named.get("fTitle", ""))
            out.end(at)
        elif one.stype == 0:
            self.held(out, one.name, value if value is not None else {})
        elif one.stype in (61, 62, 63, 68):
            self.held(out, one.typename.rstrip("*"), value)
        elif one.stype == 64:
            self.pointed(out, one.typename.rstrip("*"), value)
        elif one.stype == 65:
            out.string(value or "")
        else:
            self.numbers(out, one, value, row)

    def numbers(self, out: Out, one: Member, value: Any, row: dict[str, Any]) -> None:
        if one.stype in BASIC:
            code = BASIC[one.stype].typecode
            out.pack(code, value if value is not None else 0)
            return
        if one.stype - 20 in BASIC:  # a fixed-size array, x[10]
            fixed = list(value if value is not None else [0] * one.length)
            out.pack(f"{one.length}{BASIC[one.stype - 20].typecode}", *fixed)
            return
        code = BASIC[one.stype - 40].typecode
        values = list(value if value is not None else ())[: int(row.get(one.count, 0))]
        out.pack("B", 1)
        out.pack(f"{len(values)}{code}", *values)

    def held(self, out: Out, classname: str, value: Any) -> None:
        """An object written where it stands: an array, a collection, or a record."""
        if classname in ARRAYS:
            prim = ARRAYS[classname]
            values = np.asarray(value if value is not None else (), dtype=prim.typename)
            out.pack("i", len(values))
            out.pack(f"{len(values)}{ARRAYS[classname].typecode}", *values.tolist())
        elif classname in ("TList", "TObjArray"):
            self.collection(out, classname, value or [])
        else:
            self.record(out, classname, value if value is not None else {})

    def pointed(self, out: Out, classname: str, value: Any) -> None:
        """A pointer: nothing, or an object naming its class - ``(class, value)`` to say another."""
        if value is None:
            out.pack("I", 0)
            return
        if isinstance(value, tuple):
            classname, value = value
        at = out.tag(classname)
        self.held(out, classname, value)
        out.end(at)

    def collection(self, out: Out, classname: str, items: list[tuple[str, Any]]) -> None:
        at = out.start(5 if classname == "TList" else 3)
        out.tobject()
        out.string("")
        out.pack("i", len(items))
        if classname == "TObjArray":
            out.pack("i", 0)
        for held, value in items:
            self.pointed(out, held, (held, value))
            if classname == "TList":
                out.string("")
        out.end(at)


def _element(out: Out, one: Member) -> None:
    """One ``TStreamerElement``, under the subclass its streamer type calls for."""
    counted = 40 < one.stype < 60
    kind = "TStreamerBasicPointer" if counted else "TStreamerElement"
    tag = out.tag(kind)
    sub = out.start(2)
    shared = out.start(4)
    named = out.start(1)
    out.tobject()
    out.string(one.name)
    out.string(one.title)
    out.end(named)
    out.pack("iiii", one.stype, 0, one.length, 0)
    out.pack("5i", 0, 0, 0, 0, 0)
    out.string(one.typename)
    out.end(shared)
    if counted:
        out.pack("i", 1)
        out.string(one.count)
        out.string("")
    out.end(sub)
    out.end(tag)


def _infos(layouts: dict[str, dict[str, Member]]) -> bytes:
    out = Out()
    at = out.start(5)
    out.tobject()
    out.string("")
    out.pack("i", len(layouts))
    for name, members in layouts.items():
        tag = out.tag("TStreamerInfo")
        info = out.start(9)
        named = out.start(1)
        out.tobject()
        out.string(name)
        out.string("")
        out.end(named)
        out.pack("Ii", 0, 1)
        held = out.tag("TObjArray")
        array = out.start(3)
        out.tobject()
        out.string("")
        out.pack("ii", len(members), 0)
        for one in members.values():
            _element(out, one)
        out.end(array)
        out.end(held)
        out.end(info)
        out.end(tag)
        out.string("")
    out.end(at)
    return bytes(out.data)


def layouts() -> dict[str, dict[str, Member]]:
    """Every class a crafted file describes: the donor's, then the ones declared here."""
    with open_root(str(DONOR)) as donor:
        known = dict(donor._source.streamers())
    for name, members in LAYOUTS.items():
        known[name] = {one.name: one for one in members}
    return known


def _key(classname: str, name: str, seek: int, payload: bytes) -> bytes:
    strings = Out()
    for text in (classname, name, ""):
        strings.string(text)
    keylen = 26 + len(strings.data)
    out = Out()
    out.pack("iHiIhhii", keylen + len(payload), 4, len(payload), 0, keylen, 1, seek, 100)
    return bytes(out.data + strings.data)


def craft(path: pathlib.Path, objects: list[tuple[str, str, dict[str, Any]]]) -> pathlib.Path:
    """A ROOT file at ``path`` holding each ``(class, name, members)`` as a key."""
    known = layouts()
    writer = Writer(known)
    body = bytearray(b"\x00" * 160)  # the header, and the directory after it
    keys = []
    for classname, name, row in objects:
        out = Out()
        writer.record(out, classname, row)
        header = _key(classname, name, len(body), bytes(out.data))
        keys.append(header)
        body += header + out.data
    info = _infos(known)
    info_at = len(body)
    info_key = _key("TList", "StreamerInfo", info_at, info)
    body += info_key + info
    listing = Out()
    listing.pack("i", len(keys))
    listed = b"".join(keys)
    keys_at = len(body)
    body += _key("TFile", "", keys_at, bytes(listing.data) + listed) + listing.data + listed
    head = Out()
    head.data += b"root"
    head.pack("ii", 62006, 100)
    head.pack("iiiiiBiii", len(body), 0, 0, 0, 0, 1, 0, info_at, len(info_key) + len(info))
    body[: len(head.data)] = head.data
    directory = Out()
    directory.pack("HIIiiiii", 5, 0, 0, len(body) - keys_at, 0, 100, 0, keys_at)
    body[100 : 100 + len(directory.data)] = directory.data
    path.write_bytes(bytes(body))
    return path
