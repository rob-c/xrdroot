"""``RHistEngine<T>`` and ``RHist<T>``: ROOT 7's histograms - their axes, a bin content of
type ``T`` for every bin, and for ``RHist`` the statistics of every fill.

The bins are kept in the order of their global index (:class:`.variant.RAxes`),
as ROOT keeps them. A fill outside every bin - past an axis without flow
bins - changes no bin, though an ``RHist`` counts it in its statistics, as
ROOT's does. A slice keeps a range of bins per dimension, regroups or sums
them, and moves the bins it does not keep into the flow bins.
"""

from __future__ import annotations

from typing import Any, Generic, NoReturn, TypeVar

from .axes import NUMBERS
from .contents import RBinWithError, kind_name, kind_of
from .index import RBinIndex, RBinIndexMultiDimRange, RBinIndexRange
from .slices import ROperationRebin, ROperationSum, RSliceBinIndexMapper, RSliceSpec, as_spec
from .stats import ATOMIC, RHistStats, split_weight
from .variant import AXES, RAxes, RAxisVariant

__all__ = ["ENGINE", "HIST", "RBinWithError", "RHist", "RHistEngine"]

#: A class a template makes one of for each bin content type.
T = TypeVar("T")
#: What ``Slice`` takes as one dimension's specification, rather than a list of them.
SPECS = (RSliceSpec, RBinIndexRange, ROperationRebin, ROperationSum)


def _axes_of(args: tuple[Any, ...]) -> list[Any]:
    """The axes a histogram is made with: each given, a list of them, or ``(n, {low, high})``."""
    from .axes import RRegularAxis

    if len(args) == 2 and isinstance(args[0], NUMBERS):
        return [RRegularAxis(args[0], args[1])]
    if len(args) == 1 and not isinstance(args[0], (*AXES, RAxisVariant)):
        return list(args[0])
    return list(args)


def _indices(args: tuple[Any, ...]) -> list[Any]:
    """A bin's indices: given one by one, or as one vector or array of them."""
    if len(args) == 1 and not isinstance(args[0], (RBinIndex, *NUMBERS)):
        return list(args[0])
    return list(args)


def _specs(args: tuple[Any, ...]) -> list[RSliceSpec]:
    """``Slice``'s specifications: one per dimension, or one vector of them."""
    given = list(args[0]) if len(args) == 1 and not isinstance(args[0], SPECS) else args
    return [as_spec(spec) for spec in given]


class _Converter:
    """``hist.Convert<U>()``: ``hist.Convert['U']()``, the histogram with bins of type ``U``."""

    def __init__(self, convert: Any) -> None:
        self._convert = convert

    def __getitem__(self, kind: Any) -> Any:
        return lambda: self._convert(kind)


class RHistEngine:
    """``RHistEngine<T>``: the axes and the bin contents, without statistics."""

    #: The bin content type, ``T``: set on each class ``RHistEngine[T]`` makes.
    content: Any = "double"

    def __init__(self, *args: Any) -> None:
        self._axes = RAxes(_axes_of(args))
        self._kind = kind_of(kind_name(self.content))
        self._bins = [self._kind.zero() for _ in range(self._axes.ComputeTotalNBins())]

    @property
    def SupportsWeightedFilling(self) -> bool:
        return not self._kind.integral

    def GetAxes(self) -> Any:
        return self._axes.Get()

    def GetNDimensions(self) -> int:
        return self._axes.GetNDimensions()

    def GetTotalNBins(self) -> int:
        return len(self._bins)

    def contents(self) -> list[Any]:
        """Every bin, in the order of the global index."""
        return self._bins

    def _where(self, indices: list[Any], name: str) -> int:
        if len(indices) != self._axes.GetNDimensions():
            raise ValueError(f"invalid number of indices passed to {name}")
        index, valid = self._axes.indices_index(indices)
        if not valid:
            raise ValueError(f"bin not found in {name}")
        return index

    def GetBinContent(self, *args: Any) -> Any:
        """``GetBinContent(indices...)``: by ``RBinIndex`` or a normal bin's number, or by a
        vector or array of them."""
        return self._bins[self._where(_indices(args), "GetBinContent")]

    def SetBinContent(self, *args: Any) -> None:
        """``SetBinContent(indices..., value)``, or ``SetBinContent(indices, value)``."""
        indices = _indices(args[:-1])
        self._bins[self._where(indices, "SetBinContent")] = self._kind.convert(args[-1])

    def GetFullMultiDimRange(self) -> RBinIndexMultiDimRange:
        return self._axes.GetFullMultiDimRange()

    def Fill(self, *args: Any) -> None:
        """``Fill(values...[, RWeight])`` or ``Fill(tuple[, RWeight])``: one entry."""
        values, weight = split_weight(args)
        if len(values) != self._axes.GetNDimensions():
            raise ValueError("invalid number of arguments to Fill")
        index, valid = self._axes.values_index(values)
        if valid:
            bins, kind = self._bins, self._kind
            found = bins[index]
            bins[index] = kind.inc(found) if weight is None else kind.add(found, weight)

    def FillAtomic(self, *args: Any) -> None:
        with ATOMIC:
            self.Fill(*args)

    def Add(self, other: RHistEngine, name: str = "Add") -> None:
        """``Add(other)``: another histogram's bins, on axes that must be the same."""
        if self._axes != other._axes:
            raise ValueError(f"axes configurations not identical in {name}")
        combine = self._kind.combine
        pairs = zip(self._bins, other._bins, strict=True)
        self._bins = [combine(mine, theirs) for mine, theirs in pairs]

    def AddAtomic(self, other: RHistEngine) -> None:
        with ATOMIC:
            self.Add(other, "AddAtomic")

    def Clear(self) -> None:
        self._bins = [self._kind.zero() for _ in self._bins]

    def _made(self, kind: Any, axes: list[RAxisVariant]) -> RHistEngine:
        return ENGINE[kind](axes)

    def Clone(self) -> RHistEngine:
        made = self._made(self.content, self._axes.variants())
        made._bins = [self._kind.copy(bin) for bin in self._bins]
        return made

    def _convert(self, kind: Any) -> RHistEngine:
        made = self._made(kind, self._axes.variants())
        made._bins = [made._kind.convert(bin) for bin in self._bins]
        return made

    @property
    def Convert(self) -> _Converter:
        return _Converter(self._convert)

    def Scale(self, factor: Any) -> None:
        scale, factor = self._kind.scale, float(factor)
        self._bins = [scale(bin, factor) for bin in self._bins]

    def sliced(self, specs: list[RSliceSpec]) -> tuple[RHistEngine, bool]:
        """The sliced histogram, and whether a bin was dropped by a summed range."""
        if len(specs) != self._axes.GetNDimensions():
            raise ValueError("invalid number of specifications passed to Slice")
        axes = self._axes.Slice(specs)
        if not axes:
            raise ValueError("summing across all dimensions is not supported")
        made, mapper, dropped = self._made(self.content, axes), RSliceBinIndexMapper(specs), False
        combine = self._kind.combine
        walk = self._axes.GetFullMultiDimRange().combinations()
        for bin, original in zip(self._bins, walk, strict=True):
            mapped = mapper.mapped(list(original))
            if mapped is None:
                dropped = True
                continue
            at = made._axes.indices_index(mapped)[0]
            made._bins[at] = combine(made._bins[at], bin)
        return made, dropped

    def Slice(self, *args: Any) -> RHistEngine:
        """``Slice(specs...)``: an ``RSliceSpec`` per dimension - or a range or an operation,
        which make one - or one vector of them."""
        return self.sliced(_specs(args))[0]

    def Streamer(self, *buffer: Any) -> NoReturn:
        """ROOT's streamer refuses to store a histogram engine; so does this."""
        raise RuntimeError("unable to store RHistEngine")

    #: Writing it into a file runs its streamer, which refuses, as ROOT's does.
    _write_into = Streamer


class RHist:
    """``RHist<T>``: an engine's bins, and the statistics of every fill."""

    #: The bin content type, ``T``: set on each class ``RHist[T]`` makes.
    content: Any = "double"
    _engine: RHistEngine
    _stats: RHistStats

    def __init__(self, *args: Any) -> None:
        if len(args) == 1 and isinstance(args[0], RHist):  # RHist(std::move(other))
            self._engine, self._stats = args[0]._engine, args[0]._stats
            return
        self._engine = ENGINE[self.content](*args)
        self._stats = RHistStats(self._engine.GetNDimensions())
        for at, axis in enumerate(self._engine._axes.variants()):
            if axis.GetCategoricalAxis() is not None:  # names have no mean
                self._stats.DisableDimension(at)

    @classmethod
    def wrapping(cls, engine: RHistEngine) -> RHist:
        """The histogram of an engine, with statistics of nothing yet: ROOT's private
        constructor, whose statistics are the caller's to set."""
        made = cls.__new__(cls)
        made._engine, made._stats = engine, RHistStats(engine.GetNDimensions())
        return made

    @property
    def SupportsWeightedFilling(self) -> bool:
        return self._engine.SupportsWeightedFilling

    def GetEngine(self) -> RHistEngine:
        return self._engine

    def GetStats(self) -> RHistStats:
        return self._stats

    def GetAxes(self) -> Any:
        return self._engine.GetAxes()

    def GetNDimensions(self) -> int:
        return self._engine.GetNDimensions()

    def GetTotalNBins(self) -> int:
        return self._engine.GetTotalNBins()

    def GetNEntries(self) -> int:
        return self._stats.GetNEntries()

    def ComputeNEffectiveEntries(self) -> float:
        return self._stats.ComputeNEffectiveEntries()

    def ComputeMean(self, dim: Any = 0) -> float:
        return self._stats.ComputeMean(dim)

    def ComputeStdDev(self, dim: Any = 0) -> float:
        return self._stats.ComputeStdDev(dim)

    def GetBinContent(self, *args: Any) -> Any:
        return self._engine.GetBinContent(*args)

    def GetFullMultiDimRange(self) -> RBinIndexMultiDimRange:
        return self._engine.GetFullMultiDimRange()

    def SetBinContent(self, *args: Any) -> None:
        """``SetBinContent``: the bin set, and the statistics tainted - no longer of the bins."""
        self._engine.SetBinContent(*args)
        self._stats.Taint()

    def Fill(self, *args: Any) -> None:
        self._engine.Fill(*args)
        self._stats.Fill(*args)

    def Add(self, other: RHist) -> None:
        self._engine.Add(other._engine)
        self._stats.Add(other._stats)

    def AddAtomic(self, other: RHist) -> None:
        with ATOMIC:
            self._engine.AddAtomic(other._engine)
            self._stats.AddAtomic(other._stats)

    def Clear(self) -> None:
        self._engine.Clear()
        self._stats.Clear()

    def Clone(self) -> RHist:
        made = type(self).wrapping(self._engine.Clone())
        made._stats = self._stats.copy()
        return made

    def _convert(self, kind: Any) -> RHist:
        made = HIST[kind].wrapping(self._engine.Convert[kind]())
        made._stats = self._stats.copy()
        return made

    @property
    def Convert(self) -> _Converter:
        return _Converter(self._convert)

    def Scale(self, factor: Any) -> None:
        self._engine.Scale(factor)
        self._stats.Scale(factor)

    def Slice(self, *args: Any) -> RHist:
        """``Slice(specs...)``: the sliced histogram, whose statistics are this one's - but for
        the dimensions summed over - unless bins were dropped, which taints them."""
        specs = _specs(args)
        engine, dropped = self._engine.sliced(specs)
        made = type(self).wrapping(engine)
        if dropped or self._stats.IsTainted():
            made._stats.Taint()
            return made
        made._stats.fNEntries = self._stats.fNEntries
        made._stats.fSumW, made._stats.fSumW2 = self._stats.fSumW, self._stats.fSumW2
        pairs = zip(self._stats.copy().fDimensionStats, specs, strict=True)
        kept = [found for found, spec in pairs if spec.GetOperationSum() is None]
        made._stats.fDimensionStats = kept
        return made

    def Streamer(self, *buffer: Any) -> NoReturn:
        """ROOT's streamer refuses to store a histogram; so does this."""
        raise RuntimeError("unable to store RHist")

    #: Writing it into a file runs its streamer, which refuses, as ROOT's does.
    _write_into = Streamer


class Template(Generic[T]):
    """``RHist<T>``: the class for each bin content type, by ``[...]``."""

    def __init__(self, base: type[T]) -> None:
        self._base = base
        self._made: dict[str, type[T]] = {}

    def __getitem__(self, kind: Any) -> type[T]:
        name = kind_name(kind)
        if name not in self._made:
            kind_of(name)
            self._made[name] = type(f"{self._base.__name__}<{name}>", (self._base,),
                                    {"content": name})  # fmt: skip
        return self._made[name]


#: ``RHistEngine<T>`` and ``RHist<T>``, by ``[T]``.
ENGINE, HIST = Template(RHistEngine), Template(RHist)
