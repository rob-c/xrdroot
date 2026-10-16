"""A callback-free ntuple loop can fill whole columns in their original entry order."""
from __future__ import annotations

import inspect
import math
import types
from typing import Any, cast

import numpy as np

from ...hist import Axis, Histogram, Traits
from ...pyroot.core.hists import TH1D, TH1F
from ...pyroot.core.rootns import TSeqI
from ...pyroot.trees.batch import Batch
from ...pyroot.trees.ntuple import TNtuple, TNtupleD
from ...roofit.fitting import hfnative
from ...tree import Branch, TTree

__all__ = ['fill_ntuple_histogram']


def _canonical(node: Any, name: str, definition: tuple[str, str]) -> bool:
    method = inspect.getattr_static(node, name, None)
    if type(method) is property:
        method = method.fget
    return (type(method) is types.FunctionType
            and type(method.__module__) is str and type(method.__qualname__) is str
            and (method.__module__, method.__qualname__) == definition)


def _axis(axis: Any) -> bool:
    if (type(axis) is not Axis or '__getattribute__' in Axis.__dict__
            or '__getattr__' in Axis.__dict__):
        return False
    if not all(type(value) is Traits or hfnative._attribute(value)
               for value in Axis.__dict__.values()):
        return False
    return all(_canonical(axis, name, ('xrdroot.hist', 'Axis.' + name))
               for name in ('find_bin', 'edges', 'even'))


def _definitions(tree: Any, histogram: Any) -> bool:
    expected = ((tree, 'GetArgs', ('xrdroot.pyroot.trees.ntuple', 'TNtuple.GetArgs')),
                (tree, 'GetEntry', ('xrdroot.pyroot.trees.reading', '_Reading.GetEntry')),
                (histogram, 'Fill', ('xrdroot.pyroot.core.histbins', 'Bins.Fill')),
                (histogram, '_fill_arrays', ('xrdroot.pyroot.core.histbins', 'Bins._fill_arrays')),
                (histogram._xrd, 'fill', ('xrdroot.hist', 'Histogram.fill')))
    return all(_canonical(node, name, definition) for node, name, definition in expected)


def _entries(histogram: Any, count: int) -> bool:
    core = histogram._xrd._core
    if type(core) is not dict:
        return False
    entries = core.get('fEntries')
    return (type(entries) is float and math.isfinite(entries) and entries.is_integer()
            and 0 <= entries <= 2**53 - count)


def _histogram(histogram: Any) -> bool:
    if not hfnative.instance(histogram, (TH1F, TH1D)):
        return False
    auto = histogram.__dict__.get('_auto_axes')
    if auto is not None and (type(auto) is not set or auto):
        return False
    held = histogram._xrd
    if not hfnative.instance(held, (Histogram,)):
        return False
    if type(held.axes) is not tuple or len(held.axes) != 1:
        return False
    return _valid_axis(held.axes[0])


def _valid_axis(axis: Any) -> bool:
    if not _axis(axis):
        return False
    return _axis_values(axis) and _axis_range(axis)


def _axis_values(axis: Any) -> bool:
    return (type(axis.nbins) is int and axis.nbins > 0
            and type(axis.low) is float and type(axis.high) is float
            and type(axis._edges) in (list, np.ndarray) and len(axis._edges) == 0)


def _axis_range(axis: Any) -> bool:
    return math.isfinite(axis.low) and math.isfinite(axis.high) and axis.low < axis.high


def _tree(tree: Any) -> bool:
    if not hfnative.instance(tree, (TNtuple, TNtupleD)):
        return False
    if tree._store is not None or tree._entry_list is not None:
        return False
    return _tree_state(tree) and _tree_backing(tree)


def _tree_state(tree: Any) -> bool:
    state = ((tree._addresses, dict), (tree._disabled, set),
             (tree._friends, list), (tree._aliases, dict))
    if not all(type(value) is kind and not value for value, kind in state):
        return False
    if type(tree._code) is not str or tree._code not in ('f', 'd'):
        return False
    return True


def _tree_backing(tree: Any) -> bool:
    if not hfnative.instance(tree._source, (TTree,)):
        return False
    if not hfnative.instance(tree._batch, (Batch,)):
        return False
    callback = tree._batch._source
    return (type(callback) is types.MethodType and callback.__self__ is tree
            and callback.__func__ is cast(Any, type(tree))._backing)


def _columns(tree: Any) -> list[str] | None:
    leaves = tree._leaves()
    if not leaves or any(leaf.typename not in ('Float_t', 'Double_t') or leaf.size != 1
                         or leaf.counter is not None for leaf in leaves):
        return None
    names = tree._columns()
    if not all(type(name) is str and hfnative.instance(tree._source[name], (Branch,))
               for name in names):
        return None
    return cast(list[str], names)


def _sequence(sequence: Any) -> range | None:
    if not hfnative.instance(sequence, (TSeqI,)) or type(sequence._range) is not range:
        return None
    return sequence._range


def _range(tree: Any, entries: range) -> bool:
    return entries.step == 1 and entries.start >= 0 and entries.stop <= tree.GetEntriesFast()


def _prepared(tree: Any, histogram: Any, entries: range, column: int) -> list[str] | None:
    if not _tree(tree) or not _histogram(histogram):
        return None
    if not _range(tree, entries):
        return None
    if not _definitions(tree, histogram) or not _entries(histogram, len(entries)):
        return None
    names = _columns(tree)
    if names is None:
        return None
    return _selected(names, column)


def _selected(names: list[str], column: int) -> list[str] | None:
    return names if type(column) is int and 0 <= column < len(names) else None


def fill_ntuple_histogram(tree: Any, histogram: Any, sequence: Any, column: int) -> bool:
    """Return false before a fill when the original scalar loop must run instead."""
    with hfnative.checking():
        entries = _sequence(sequence)
        if entries is None:
            return False
        if not entries:
            return True
        names = _prepared(tree, histogram, entries, column)
        if names is None:
            return False
    for batch in tree._source.iterate(names, entry_start=entries.start, entry_stop=entries.stop):
        histogram.Fill(np.asarray(batch[names[column]], dtype=tree._code))
    tree.GetEntry(entries.stop - 1)
    return True
