"""ROOT's meanings of the runtime's helpers, where ROOT's objects mean more than plain C++'s.

A translation in the ROOT dialect imports these in place of the plain ones
of :mod:`xrdroot.pycxx.runtime`, under the same names: ``delete`` of a file
closes it, placement ``new`` into a ``TClonesArray`` is its ``AddAt``, an
``obj.Name() = v`` is ROOT's setter ``SetName(v)``, a ROOT object says its own
class to ``typeid``, ROOT 7's pad lengths are literals, and ROOT's ``Printf``,
``Form``, ``Info``, ``Warning``, ``Error`` and ``Fatal`` print as ROOT's do.
Loading this module also hands the standard library ROOT's numeric typedefs to
read with, and ROOT's ``RVec`` to give element-wise ``<cmath>`` results in.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ...cppstd import cformat as _cformat
from ...cppstd import cmath as _cmath
from ...cppstd.units import duration_literal
from ...pycxx.runtime import callrefs as _callrefs
from ...pycxx.runtime import objects as _objects
from ...pycxx.runtime import references as _references
from ...pycxx.runtime import typeinfo as _typeinfo
from ...pycxx.runtime.istreams import KINDS
from ...pycxx.runtime.printf import cformat
from .root import ROOT

__all__ = [
    "delete",
    "fscanf",
    "construct_at",
    "deref",
    "assign_method",
    "CallRef",
    "typeid_dynamic",
    "user_literal",
    "Printf",
    "Form",
    "Info",
    "Warning",
    "Error",
    "Fatal",
    "Printf_",
]

#: ROOT 7's suffixes, as the ``RPadLength`` member class each makes.
PAD_LENGTHS = {"_normal": "Normal", "_px": "Pixel", "_user": "User"}


def delete(obj: Any) -> None:
    """``delete p``: a user class's destructor runs; a ROOT file is closed, as its would be."""
    if _objects.delete(obj):
        return
    close = getattr(obj, "Close", None)
    if callable(close) and hasattr(obj, "IsOpen"):
        close()


def construct_at(container: Any, index: Any, obj: Any) -> Any:
    """``new (clones[i]) T(args)``: ``obj`` put in slot ``index`` - ROOT's ``AddAt``, or ``[i] =``.

    A ``TClonesArray`` hands placement new the memory of its slot ``i``;
    here the object is built first and the array given it, which is the
    same array of the same objects afterwards.
    """
    add = getattr(container, "AddAt", None)
    if callable(add):
        add(obj, int(index))
        return obj
    return _objects.construct_at(container, index, obj)


def deref(pointer: Any) -> Any:
    """``*p`` when the type of ``p`` was not known: a cell's value, an array's first element.

    A ROOT collection that is a list is not a C array: ``*list`` is the list.
    A library object that holds a value as a cell does - an RNTuple field's
    ``shared_ptr<int>`` - says so with a true ``_cint_cell``.
    """
    if isinstance(pointer, (np.ndarray, list)) and not hasattr(pointer, "ClassName"):
        return pointer[0]  # a C array; a ROOT collection that is a list, *list, is the list
    return _objects.held(pointer)


def assign_method(obj: Any, name: str, args: tuple[Any, ...], value: Any) -> Any:
    """``obj.name(*args) = value``: ROOT's ``Set<name>``, or an assignment into the result."""
    setter = getattr(obj, f"Set{name}", None)
    if not args and callable(setter):
        setter(value)
        return value
    return _references.assign_method(obj, name, args, value)


class CallRef(_callrefs.CallRef):
    """What ``obj.name(*args)`` returns a reference to, written back with ROOT's setter."""

    __slots__ = ()

    assigning = staticmethod(assign_method)


def typeid_dynamic(value: Any, static: str | None = None) -> Any:
    """``typeid(e)`` of an object whose class may be one derived from its static type.

    The macro's classes say what they are, and so do ROOT's, by ``ClassName()``.
    """
    own = getattr(value, "ClassName", None)
    declared = getattr(type(value), "_cxx_layout_", None) is not None
    if value is not None and not declared and callable(own):
        return _typeinfo.typeid_of(str(own()))
    return _typeinfo.typeid_dynamic(value, static)


def user_literal(suffix: str, value: Any) -> Any:
    """What the library's ``operator""`` for ``suffix`` makes of the number ``value``: a
    ``std::chrono`` duration, or one of ROOT 7's pad lengths - ``0.1_normal`` calls
    ``ROOT::Experimental::operator""_normal(0.1)``, an ``RPadLength::Normal``, looked up only
    when the literal is evaluated, the way every ROOT name a translation uses is."""
    duration = duration_literal(suffix, value)
    if duration is not None:
        return duration
    return getattr(ROOT.RPadLength, PAD_LENGTHS[suffix])(value)


def fscanf(file: Any, fmt: Any, *targets: Any) -> int:
    """``fscanf``: the fields of a ``FILE`` that ``fmt`` names, each stored where its target is.

    C's ``fscanf`` keeps its place in the stream between calls, so the scan of a file that
    has started is kept on the :class:`~xrdroot.cppstd.cstdio.FILE` itself: the first call
    reads the rest of the stream and every later one goes on from where the one before it
    stopped. A program that scans a file must not also read it another way - ``fgets`` after
    ``fscanf`` finds the stream at its end - because the text is held here as the scan reads
    it. The count is C's: the fields stored, or ``EOF`` when the text ran out before one was.
    """
    if file is None:
        raise TypeError(
            "fscanf of a null FILE *: fopen returned None, and C would read through a null "
            "pointer; check fp against None before scanning from it"
        )
    scan = getattr(file, "scan", None)
    if scan is None:
        # The scan's own state, so that a call goes on where the last one stopped: cformat's
        # scanning of a text, as sscanf uses it, without a fresh text for every call.
        text = file.stream.read().decode("latin-1")
        scan = file.scan = _cformat._Scan(text)
    given = iter(targets)
    stored = 0
    for kind, item in _cformat._tokens(str(fmt)):
        outcome = _cformat._STEPS[kind](scan, item, given)
        if outcome == _cformat.EXHAUSTED:
            file.eof = True
            return stored or _cformat.EOF
        if outcome == _cformat.MISMATCH:
            return stored
        stored += outcome
    return stored


def Printf(fmt: Any, *args: Any) -> None:
    """ROOT's ``Printf``: ``printf`` with a newline after."""
    sys.stdout.write(cformat(fmt, *args) + "\n")


def Form(fmt: Any, *args: Any) -> str:
    """ROOT's ``Form``: ``sprintf`` into a buffer of ROOT's, handed back."""
    return cformat(fmt, *args)


def _message(level: str, location: Any, fmt: Any, args: tuple[Any, ...]) -> None:
    where = f" in <{location}>" if location else ""
    sys.stderr.write(f"{level}{where}: {cformat(fmt, *args)}\n")


def Info(location: Any, fmt: Any, *args: Any) -> None:
    """ROOT's ``Info(where, fmt, ...)``: ``Info in <where>: message`` on standard error."""
    _message("Info", location, fmt, args)


def Warning(location: Any, fmt: Any, *args: Any) -> None:
    _message("Warning", location, fmt, args)


def Error(location: Any, fmt: Any, *args: Any) -> None:
    _message("Error", location, fmt, args)


def Fatal(location: Any, fmt: Any, *args: Any) -> None:
    _message("Fatal", location, fmt, args)
    raise SystemExit(1)


#: ``Printf`` under a name no macro's own ``Printf`` method can hide.
Printf_ = Printf


def _rvec(found: Any) -> Any:
    """Construct the same ROOT vector only when an array operation needs its result."""
    from ...pyroot.rdf.rvec import RVec

    return RVec[found.dtype](found)


#: ROOT's numeric typedefs, which ``std::istream >>`` reads as the types they stand for.
KINDS.update({
    **dict.fromkeys(("Int_t", "Long64_t", "UInt_t"), "integral"),
    **dict.fromkeys(("Double_t", "Float_t"), "floating"),
})
_cmath.set_vector_result(_rvec)
