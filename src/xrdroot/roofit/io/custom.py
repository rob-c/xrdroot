"""The streamers RooFit writes by hand, read as their ``Streamer`` functions write them.

Each is ``RooFit``'s own code turned round: ``RooRealVar::Streamer`` writes
its base, its errors, a pointer to its binning and its shared properties;
``RooLinkedList::Streamer`` a version with no byte count, its ``TObject``
and its members as pointers; ``RooRefArray`` a ``TRefArray`` of the
proxies' unique identifiers; the code repository of a workspace three maps
of strings - and a repository that holds any is refused, since what it
holds is C++ to compile.
"""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from .stream import CUSTOM, Reader, Streamed, base

__all__ = ["install"]


def _code_repo(reader: Reader, made: Streamed) -> None:
    """``RooWorkspace::CodeRepo``: the classes' files, their relations and extra headers."""
    buf = reader.buf
    version, end = buf.header()
    files = [(buf.string(), buf.string(), buf.string(), buf.string()) for _ in range(buf.i32())]
    relations = [(buf.string(), buf.string(), buf.string()) for _ in range(buf.i32())]
    extras = [(buf.string(), buf.string(), buf.string()) for _ in range(buf.i32())] if (
        version == 2) else []  # fmt: skip
    buf.resume(end)
    made.m.update(files=files, relations=relations, extras=extras)


def _ref_array(reader: Reader, made: Streamed) -> None:
    """``RooRefArray``: a ``TRefArray`` in a record - the proxies, by unique identifier."""
    buf = reader.buf
    _version, end = buf.header()
    _inner, inner_end = buf.header()
    made.m["TObject"] = reader.tobject()
    made.m["fName"] = buf.string()
    count, made.m["fLowerBound"] = buf.i32(), buf.i32()
    made.m["pid"] = buf.u16()
    made.m["uids"] = [buf.u32() for _ in range(count)]
    buf.resume(inner_end)
    buf.resume(end)


def _linked_list(reader: Reader, made: Streamed) -> None:
    """``RooLinkedList``: a bare version, a ``TObject``, then its members as pointers."""
    buf = reader.buf
    version, end = buf.header()
    made.m["TObject"] = reader.tobject()
    made.m["items"] = [reader.pointer() for _ in range(buf.i32())]
    if 1 < version < 4:
        made.m["_name"] = buf.string()
    buf.resume(end)


def _abs_binning(reader: Reader, made: Streamed) -> None:
    """``RooAbsBinning``: a record, a ``TNamed`` - a ``TObject`` in version 1 - and a
    ``RooPrintable``."""
    buf = reader.buf
    version, end = buf.header()
    made.m["TNamed"] = reader.tobject() if version == 1 else reader.tnamed()
    _printable, printable_end = buf.header()
    buf.resume(printable_end)
    buf.resume(end)


def _real_var(reader: Reader, made: Streamed) -> None:
    """``RooRealVar::Streamer``: its base, its errors, its binning, its shared properties."""
    buf = reader.buf
    version, end = buf.header()
    made.m["RooAbsRealLValue"] = reader.fill(base("RooAbsRealLValue"), "RooAbsRealLValue")
    if version == 1:
        made.m["fit"] = (reader.number("d"), reader.number("d"), buf.i32())
    made.m["_error"] = reader.number("d")
    made.m["_asymErrLo"] = reader.number("d")
    made.m["_asymErrHi"] = reader.number("d")
    if version >= 2:
        made.m["_binning"] = reader.pointer()
    if version == 3:
        made.m["_sharedProp"] = reader.pointer()
    if version >= 4:
        made.m["_sharedProp"] = reader.object("RooRealVarSharedProperties")
    buf.resume(end)


def _collection(reader: Reader, made: Streamed) -> None:
    """``TList`` and ``THashList``: a record, a ``TObject``, a name, then each object and its
    option."""
    buf = reader.buf
    version, end = buf.header()
    if version > 3:
        made.m["TObject"] = reader.tobject()
        made.m["fName"] = buf.string()
    items, options = [], []
    for _ in range(buf.i32()):
        items.append(reader.pointer())
        options.append(buf.string() if version > 3 else "")
    made.m.update(items=items, options=options)
    buf.resume(end)


def _refused(reason: str) -> Any:
    def read(reader: Reader, made: Streamed) -> None:
        raise UnsupportedFeatureError(reason)

    return read


def install() -> None:
    """Put the readers by hand where :class:`~.stream.Reader` looks first."""
    CUSTOM.update(
        {
            "RooWorkspace::CodeRepo": _code_repo,
            "RooRefArray": _ref_array,
            "RooLinkedList": _linked_list,
            "RooAbsBinning": _abs_binning,
            "RooRealVar": _real_var,
            "TList": _collection,
            "THashList": _collection,
            "RooTreeDataStore": _refused(
                "this dataset is stored in a TTree (RooTreeDataStore); datasets are read here "
                "from the vector store RooFit writes by default"
            ),
        }
    )


install()
