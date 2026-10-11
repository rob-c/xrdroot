"""``ROOT.RDataFrame``: xrdroot's ``RDataFrame``, taking and giving back PyROOT's objects.

The frame itself is :class:`xrdroot.RDataFrame`, which already has ROOT's
methods and runs ROOT's C++ expressions over whole batches. What this adds is
the boundary: a frame is made from a ``TTree`` or ``TChain`` of
:mod:`xrdroot.pyroot.trees` as well as from files, a result's value is handed
back as the PyROOT object ``core`` wraps it in, ``AsNumpy`` gives its dict at
once as PyROOT's does - a NumPy array for each entry of a collection - and
``RDF.FromNumpy`` makes a frame of a dict of arrays.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

from ... import rdf as _rdf
from ...tree import Jagged
from ..stl import is_vector
from ..trees import hooks
from .datasources import RCsvDS, csv_source, lazy_source, sqlite_source
from .entrywise import per_entry
from .helpers import Not, PassAsVec
from .jitted import retried

__all__ = ["RDataFrame", "RResultPtr", "RDF"]


def _unwrapped(value: Any) -> Any:
    """What xrdroot takes for a PyROOT argument: a tree's own, a list for a vector of names."""
    if isinstance(value, (RDataFrame, RResultPtr)):
        return value._inner
    if hasattr(value, "_xrd"):
        return value._xrd
    if is_vector(value):
        return [str(each) for each in value]
    return value


def _wrapped(value: Any) -> Any:
    """What a PyROOT script is given back for what xrdroot returned."""
    if isinstance(value, _rdf.RDataFrame):
        return RDataFrame._of(value)
    if isinstance(value, _rdf.Result) or hasattr(value, "IsReady"):
        return RResultPtr(value)  # a booked result, or a booked Snapshot
    return value


class RResultPtr:
    """``RResultPtr``: a booked result; its value, when asked for, as a PyROOT object."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __repr__(self) -> str:
        return repr(self._inner)

    def GetValue(self) -> Any:
        return _wrapped(hooks.wrap(self._inner.GetValue()))

    GetPtr = GetValue
    GetSharedPtr = GetValue

    def IsReady(self) -> bool:
        return bool(self._inner.IsReady())

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.GetValue(), name)

    def __float__(self) -> float:
        return float(self.GetValue())

    def __int__(self) -> int:
        return int(self.GetValue())

    def __index__(self) -> int:
        return int(self.GetValue())

    def __iter__(self) -> Iterator[Any]:
        return iter(self.GetValue())

    def __len__(self) -> int:
        return len(self.GetValue())

    def __getitem__(self, key: Any) -> Any:
        return self.GetValue()[key]

    def __str__(self) -> str:
        return str(self.GetValue())

    # ``*count == n`` in a macro: the value compared, as C++ compares what it points at
    def __eq__(self, other: object) -> bool:
        return bool(self.GetValue() == other)

    def __ne__(self, other: object) -> bool:
        return bool(self.GetValue() != other)

    def __lt__(self, other: Any) -> bool:
        return bool(self.GetValue() < other)

    def __le__(self, other: Any) -> bool:
        return bool(self.GetValue() <= other)

    def __gt__(self, other: Any) -> bool:
        return bool(self.GetValue() > other)

    def __ge__(self, other: Any) -> bool:
        return bool(self.GetValue() >= other)

    __hash__ = None  # type: ignore[assignment]


def _column(values: Any) -> Any:
    """A column as ``AsNumpy`` gives it: an array, with an array per entry of a collection."""
    if isinstance(values, Jagged):
        made = np.empty(len(values), dtype=object)
        for at in range(len(values)):
            made[at] = np.asarray(values[at])
        return made
    return np.asarray(values)


class _Method:
    """A frame's method, called with PyROOT's objects; ``m['double']`` - a macro's C++ template
    arguments - is the same method, since a column's type is read from the data."""

    def __init__(self, name: str, found: Callable[..., Any]) -> None:
        self.name, self.found = name, found

    def __call__(self, *arguments: Any, **options: Any) -> Any:
        given = per_entry(self.name, [_unwrapped(each) for each in arguments])
        keywords = {key: _unwrapped(value) for key, value in options.items()}
        try:
            return _wrapped(self.found(*given, **keywords))
        except Exception as why:  # C++ the batch evaluator refuses: translated, if it can be
            again = retried(self.name, getattr(self.found, "__self__", None), given, why)
            if again is None:
                raise
        return _wrapped(self.found(*again, **keywords))

    def __getitem__(self, types: Any) -> _Method:
        return self


class RDataFrame:
    """``ROOT.RDataFrame``, and every node a transformation of it makes."""

    #: The frame of xrdroot's this one is.
    _inner: Any

    def __init__(self, *arguments: Any, **options: Any) -> None:
        if len(arguments) == 1 and isinstance(arguments[0], RDataFrame):  # RNode(node): itself
            self._inner = arguments[0]._inner
            return
        given = [_unwrapped(each) for each in arguments[:3]]
        self._inner = _rdf.RDataFrame(*given, **options)

    @classmethod
    def _of(cls, inner: Any) -> RDataFrame:
        made = cls.__new__(cls)
        made._inner = inner
        return made

    def __repr__(self) -> str:
        return repr(self._inner)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        found = getattr(self._inner, name)
        if not callable(found):
            return found

        return _Method(name, found)

    def Hist(self, *arguments: Any) -> Any:
        """``Hist``: a ROOT 7 ``RHist`` of columns, booked (see :mod:`..histv7.frames`)."""
        from ..histv7.frames import book

        return book(self, arguments)

    def AsNumpy(self, columns: Any = None, exclude: Any = None, lazy: bool = False) -> Any:
        """Every column asked for, read now, as a dict of NumPy arrays."""
        found = self._inner.AsNumpy(_unwrapped(columns), _unwrapped(exclude))
        return {name: _column(values) for name, values in found.GetValue().items()}


def _from_numpy(columns: dict[str, Any]) -> RDataFrame:
    """``RDF.FromNumpy``: a frame of a dict of arrays, a column per key."""
    from ... import create, open_root

    buffer = io.BytesIO()
    with create(buffer) as out:
        out["numpy"] = {str(name): np.asarray(values) for name, values in columns.items()}
    return RDataFrame(open_root(io.BytesIO(buffer.getvalue()))["numpy"])


def _trivial(size: int, skipEvenEntries: bool = False) -> RDataFrame:
    """``RDF.MakeTrivialDataFrame``: ROOT's demonstration source, one column ``col0`` of each
    entry's number - the odd entries only, when the even ones are skipped."""
    numbers = np.arange(int(size), dtype=np.uint64)
    return _from_numpy({"col0": numbers[1::2] if skipEvenEntries else numbers})


def _from_csv(*arguments: Any, **options: Any) -> RDataFrame:
    """``RDF.FromCSV``: a frame over a CSV file's columns (see :mod:`.datasources`)."""
    return RDataFrame(csv_source(*arguments, **options))


def _from_sqlite(fileName: Any, query: Any) -> RDataFrame:
    """``RDF.FromSqlite``: a frame over the rows an SQL query gives."""
    return RDataFrame(sqlite_source(fileName, query))


def _from_results(*pairs: Any) -> RDataFrame:
    """``RDF.MakeLazyDataFrame``: a frame over columns ``Take`` booked."""
    return RDataFrame(lazy_source(*pairs))


def _model(*parts: Any) -> tuple[Any, ...]:
    """``TH1DModel`` and its kin: the tuple xrdroot books from, in the constructor's order."""
    return tuple(list(part) if isinstance(part, np.ndarray) else part for part in parts)


def _run_graphs(results: Iterable[Any]) -> int:
    """``RunGraphs``: the results' loops run together - a booked ``RHist``'s are the columns it
    is filled from."""
    given = [_unwrapped(each) for each in results]
    return int(_rdf.RunGraphs([part for each in given for part in _parts(each)]))


def _parts(result: Any) -> list[Any]:
    """The frame's own results a result is: itself, or the columns a booked ``RHist`` takes."""
    if not hasattr(type(result), "booked"):
        return [result]
    return [_unwrapped(part) for part in result.booked()]


class _RDF:
    """``ROOT.RDF``: the functions and models that go with ``RDataFrame``."""

    RunGraphs = staticmethod(_run_graphs)
    FromNumpy = staticmethod(_from_numpy)
    MakeNumpyDataFrame = staticmethod(_from_numpy)
    FromCSV = staticmethod(_from_csv)
    MakeCsvDataFrame = staticmethod(_from_csv)
    FromSqlite = staticmethod(_from_sqlite)
    MakeLazyDataFrame = staticmethod(_from_results)
    MakeTrivialDataFrame = staticmethod(_trivial)
    MakeSqliteDataFrame = staticmethod(_from_sqlite)
    RCsvDS = RCsvDS
    TH1DModel = staticmethod(_model)
    TH2DModel = staticmethod(_model)
    TH3DModel = staticmethod(_model)
    TProfile1DModel = staticmethod(_model)
    TProfile2DModel = staticmethod(_model)
    RNode = RDataFrame
    RResultPtr = RResultPtr
    Not = staticmethod(Not)
    PassAsVec = PassAsVec

    @staticmethod
    def AsRNode(frame: Any) -> Any:
        """``AsRNode(node)``: any node of a frame as an ``RNode`` - which every node here is."""
        return frame

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"ROOT has RDF.{name}; xrdroot.pyroot does not yet")

    def __repr__(self) -> str:
        return "<namespace ROOT::RDF>"


#: ``ROOT.RDF``.
RDF = _RDF()
