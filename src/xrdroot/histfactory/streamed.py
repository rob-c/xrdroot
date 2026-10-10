"""A ``Measurement`` as ROOT's streamer writes it, byte for byte, for ROOT to read back.

A ``RooStats::HistFactory::Measurement`` is a ``TNamed`` with members of
every kind ROOT streams: ``std::string`` (a counted record of version 10
round a ``TString``), ``vector<string>``, ``map<string, double>`` (a
member-wise vector of pairs), vectors of HistFactory's own classes
(member-wise: each element's values for every object in turn, a nested
vector's objects counted inside), members of classes without a
``ClassDef`` (version 0, then the class's checksum), and ``TH1*`` pointers
(the histogram tagged with its class the first time, referred back to by
the tag's place in the key after that, as ``TBufferFile`` keeps its map).
The checksums are the ones ROOT 6.40 computes for its dictionaries, taken
from a file it wrote.
"""

from __future__ import annotations

import struct
from typing import Any

from ..buffer import CLASS_MASK, MAP_OFFSET, NEW_CLASS_TAG
from ..hist import Histogram
from ..winfo import INFOS, Element
from ..writer import WBuffer, info_entry, object_payload
from .layouts import CHECKSUMS, LAYOUTS, full, kind_type

__all__ = ["stream", "streamed"]

MEASUREMENT_VERSION, MEASUREMENT_CHECKSUM = 3, 0x11A8DD71
#: The class version ROOT's dictionary gives a class without a ``ClassDef``.
FOREIGN_VERSION = 1
#: The version ROOT writes for its STL collections and strings, and the member-wise bit.
STL_VERSION, MEMBER_WISE = 10, 0x4000


#: What each member is read from our model by, where the getter is the same in every class.
GETTERS: dict[str, str] = {
    "fName": "GetName", "fInputFile": "GetInputFile", "fHistoName": "GetHistoName",
    "fHistoPath": "GetHistoPath", "fLow": "GetLow", "fHigh": "GetHigh", "fVal": "GetVal",
    "fChannelName": "GetChannelName", "fNormalizeByTheory": "GetNormalizeByTheory",
    "fStatErrorActivate": "HasStatError", "fhNominal": "GetHisto", "fhData": "GetHisto",
    "fOverallSysList": "GetOverallSysList", "fNormFactorList": "GetNormFactorList",
    "fHistoSysList": "GetHistoSysList", "fHistoFactorList": "GetHistoFactorList",
    "fShapeSysList": "GetShapeSysList", "fShapeFactorList": "GetShapeFactorList",
    "fStatError": "GetStatError", "fData": "GetData", "fAdditionalData": "GetAdditionalData",
    "fStatErrorConfig": "GetStatErrorConfig", "fSamples": "GetSamples",
    "fRelErrorThreshold": "GetRelErrorThreshold", "fConstraintType": "GetConstraintType",
    "fActivate": "GetActivate", "fUseHisto": "GetUseHisto", "fConstant": "IsConstant",
    "fHasInitialShape": "HasInitialShape", "fExpression": "GetExpression",
    "fDependents": "GetDependents", "fParamsToFix": "GetParamsToFix",
    "fParamValsToSet": "GetParamsToSet", "fOutputFilePrefix": "GetOutputFilePrefix",
    "fPOI": "GetPOIList", "fLumi": "GetLumi", "fLumiRelErr": "GetLumiRelErr",
    "fBinLow": "GetBinLow", "fBinHigh": "GetBinHigh",
    "fInterpolationScheme": "GetInterpolationScheme", "fChannels": "GetChannels",
    "fConstantParams": "GetConstantParams", "fParamValues": "GetParamValues",
    "fFunctionObjects": "GetFunctionObjects", "fAsimovDatasets": "GetAsimovDatasets",
    "fGammaSyst": "GetGammaSyst", "fUniformSyst": "GetUniformSyst",
    "fLogNormSyst": "GetLogNormSyst", "fNoSyst": "GetNoSyst",
    "fInputFileLow": "GetInputFileLow", "fInputFileHigh": "GetInputFileHigh",
    "fHistoNameLow": "GetHistoNameLow", "fHistoNameHigh": "GetHistoNameHigh",
    "fHistoPathLow": "GetHistoPathLow", "fHistoPathHigh": "GetHistoPathHigh",
    "fhLow": "GetHistoLow", "fhHigh": "GetHistoHigh",
}  # fmt: skip
#: Members ROOT keeps that our model does not: what they are written as.
FIXED: dict[str, Any] = {"fExportOnly": True, "fhCountingHist": None}


def _value(obj: Any, member: str) -> Any:
    """One member's value from our model's object, by its getter: a systematic of one
    histogram answers for the base class's "high" slots and leaves the "low" ones empty."""
    if member in FIXED:
        return FIXED[member]
    return getattr(obj, GETTERS[member])()


class _Stream:
    """The bytes of one key's object, and the classes tagged in it so far."""

    def __init__(self, keylen: int) -> None:
        self.buf = WBuffer()
        self.keylen = keylen
        self.tags: dict[str, int] = {}
        self.used: dict[str, None] = {}
        #: Where each ``TH1*`` member's bytes lie, for whoever compares them with ROOT's.
        self.histograms: list[tuple[int, int]] = []

    def strings(self, items: Any) -> None:
        """``vector<string>``: a record of version 10, how many, and each as a ``TString``."""
        at = self.buf.start(STL_VERSION)
        self.buf.i32(len(items))
        for item in items:
            self.buf.string(str(item))
        self.buf.end(at)

    def _collection(self, classname: str) -> int:
        """A member-wise collection's start: version 10 with the bit, its class's 0 and checksum."""
        at = self.buf.start(STL_VERSION | MEMBER_WISE)
        self.buf.u16(0)
        self.buf.u32(CHECKSUMS[classname])
        return at

    def column(self, texts: list[Any]) -> None:
        """Every object's ``std::string`` member at once, member-wise: one record of version 10
        round them all - and nothing at all for no objects."""
        if texts:
            at = self.buf.start(STL_VERSION)
            for text in texts:
                self.buf.string(str(text))
            self.buf.end(at)

    def mappings(self, kind: str, groups: list[Any]) -> None:
        """``map<string, T>`` of each of ``groups`` (the objects holding one, in turn): one
        header for a member-wise vector of pairs, then each map's count, its keys in the map's
        own order as one column, and its values."""
        pair, value = {"map": ("pair<string,double>", "double"),
                       "mapbool": ("pair<string,bool>", "bool")}[kind]  # fmt: skip
        at = self._collection(pair)
        for items in groups:
            keys = sorted(items, key=lambda key: str(key).encode())
            self.buf.i32(len(keys))
            self.column(keys)
            for key in keys:
                self.number(value, items[key])
        self.buf.end(at)

    def number(self, kind: str, value: Any) -> None:
        if kind == "double":
            self.buf.raw(struct.pack(">d", float(value)))
        elif kind == "int":
            self.buf.i32(int(value))
        else:
            self.buf.u8(int(bool(value)))

    def histogram(self, hist: Any) -> None:
        """``TH1*``: nothing for none; else the histogram, tagged with its class the first time
        and referred back to the tag's place in the key after that."""
        if hist is None:
            self.buf.i32(0)
            return
        core = getattr(hist, "_xrd", hist)  # the histogram itself, under any wrapper
        classname, payload, used = object_payload(
            core if isinstance(core, Histogram) else Histogram.of(core))
        self.used.update(dict.fromkeys(used))
        at = len(self.buf.data)
        self.buf.raw(b"\0\0\0\0")
        if classname in self.tags:
            self.buf.u32(CLASS_MASK | (self.tags[classname] + self.keylen + MAP_OFFSET))
        else:
            self.tags[classname] = len(self.buf.data)
            self.buf.u32(NEW_CLASS_TAG)
            self.buf.cstring(classname)
        self.buf.raw(payload)
        self.buf.end(at)
        self.histograms.append((at, len(self.buf.data)))

    def record(self, classname: str, items: list[Any]) -> None:
        """A class without a ``ClassDef`` written whole: version 0 and its checksum, then its
        members - of the one object, or of a base class's objects each in turn."""
        at = self.buf.start(0)
        self.buf.u32(CHECKSUMS[classname])
        self.members(classname, items, wise=False)
        self.buf.end(at)

    def vectors(self, classname: str, groups: list[Any]) -> None:
        """``vector<Class>`` of each of ``groups`` (the objects holding one, in turn), member-wise:
        one header, then each group's count and its objects' members element by element."""
        at = self._collection(classname)
        for items in groups:
            self.buf.i32(len(items))
            self.members(classname, list(items), wise=True)
        self.buf.end(at)

    def members(self, classname: str, items: list[Any], wise: bool) -> None:
        """Each member of the class for every object in turn; one object is the object's own,
        and none - an empty vector's - is nothing at all. A base class is a record of its own
        round one object's, and member-wise (``wise``) just its members among the rest."""
        if not items:
            return
        for member, kind in LAYOUTS[classname]:
            if kind == "base" and wise:
                self.members(member, items, wise=True)
            elif kind == "base":
                self.record(member, items)
            else:
                self.element(kind, [_value(obj, member) for obj in items])

    def element(self, kind: str, values: list[Any]) -> None:
        """One member of every object in turn, as its kind is streamed."""
        if kind.startswith("["):
            self.vectors(kind[1:-1], [list(value) for value in values])
        elif kind in ("map", "mapbool"):
            self.mappings(kind, [dict(value) for value in values])
        elif kind == "string":
            self.column(values)
        else:
            each = self._each(kind)
            for value in values:
                each(value)

    def _each(self, kind: str) -> Any:
        """How one object's member of ``kind`` is written."""
        if kind in ("double", "int", "bool"):
            return lambda value: self.number(kind, value)
        if kind == "strings":
            return lambda value: self.strings(list(value))
        if kind == "TH1*":
            return self.histogram
        if kind == "HistRef":
            return self.reference
        return lambda value: self.record(kind, [value])

    def reference(self, hist: Any) -> None:
        """A ``HistRef``: the class's record round the histogram's pointer."""
        at = self.buf.start(0)
        self.buf.u32(CHECKSUMS["HistRef"])
        self.histogram(hist)
        self.buf.end(at)


def stream(measurement: Any, keylen: int) -> _Stream:
    """The measurement streamed for a key whose header is ``keylen`` long."""
    made = _Stream(keylen)
    at = made.buf.start(MEASUREMENT_VERSION)
    # ROOT 6.40's TObject::Streamer keeps the heap bits to itself: a measurement's are none.
    made.buf.named(measurement.GetName(), measurement.GetTitle(), bits=0)
    made.members("Measurement", [measurement], wise=False)
    made.buf.end(at)
    return made


def streamed(measurement: Any, keylen: int) -> tuple[bytes, tuple[str, ...], dict[Any, bytes]]:
    """The measurement's bytes for a key whose header is ``keylen`` long, the classes of the
    library's own it streamed, and the descriptions of HistFactory's to carry in the file."""
    made = stream(measurement, keylen)
    return bytes(made.buf.data), tuple(made.used), descriptions()


def _signed(checksum: int) -> int:
    """A checksum as the signed integer a ``TStreamerBase`` keeps it as."""
    return checksum - (1 << 32) if checksum >= 1 << 31 else checksum


#: ``sizeof`` of each class a member may be, as ROOT's dictionary measures it.
SIZES = {"Data": 104, "HistRef": 8, "StatErrorConfig": 16, "StatError": 192}


#: What a ``TStreamerSTL`` says besides its type: the container's kind and what it holds.
STL_EXTRAS = {"string": (365, 365), "strings": (1, 365), "map": (4, 61), "mapbool": (4, 61)}


def _element(member: str, kind: str) -> Element:
    """One ``TStreamerElement`` of a HistFactory class, in the writer's description."""
    none = (0, 0, 0, 0, 0)
    stype, typename = kind_type(member, kind)
    if kind == "base":
        return ("TStreamerBase", full(member), "", 0, 0, 0, 0,
                (0, _signed(CHECKSUMS[member]), 0, 0, 0), "BASE", (0,))  # fmt: skip
    if kind in ("double", "int", "bool"):
        size = {"double": 8, "int": 4, "bool": 1}[kind]
        return ("TStreamerBasicType", member, "", stype, size, 0, 0, none, typename, ())
    if kind == "TH1*":
        return ("TStreamerObjectPointer", member, "", stype, 8, 0, 0, none, typename, ())
    if kind.startswith("[") or kind in STL_EXTRAS:
        extras = STL_EXTRAS.get(kind, (1, 61))
        return ("TStreamerSTL", member, "", stype, 24, 0, 0, none, typename, extras)
    return ("TStreamerObjectAny", member, "", stype, SIZES.get(kind, 8), 0, 0, none, typename, ())


def descriptions() -> dict[tuple[str, int], bytes]:
    """Every HistFactory class's ``TStreamerInfo`` entry, keyed as the file carries them."""
    found: dict[tuple[str, int], bytes] = {}
    named = next(e for e in INFOS["TH1"][2] if e[0] == "TStreamerBase" and e[1] == "TNamed")
    for classname, layout in LAYOUTS.items():
        elements = tuple(_element(member, kind) for member, kind in layout)
        if classname == "Measurement":
            found[(full(classname), MEASUREMENT_VERSION)] = info_entry(
                full(classname), MEASUREMENT_CHECKSUM, MEASUREMENT_VERSION, (named, *elements))
        else:
            found[(full(classname), FOREIGN_VERSION)] = info_entry(
                full(classname), CHECKSUMS[classname], FOREIGN_VERSION, elements)
    return found
