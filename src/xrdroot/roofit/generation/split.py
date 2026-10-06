"""RooFit's binned and split generator contexts: ``RooBinnedGenContext`` and
``RooSimSplitGenContext``.

``generate`` - unless ``AutoBinned(false)`` - draws a binned density's events
bin by bin, as ``generateBinned`` would, and hands them back as a weighted
dataset (``wu``) of the bin centres; a simultaneous density asked for its
category too, and allowed binned generation, draws each state's events from
that state's own context - its expected number, or Poisson-shared - and
joins them in one dataset (``hmaster``) indexed by the category.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..messages import ERROR, INFO, log
from ..rng import generator

__all__ = ["auto_binned", "binned_events", "split_events", "splits"]


def auto_binned(pdf: Any, names: frozenset[str], auto: bool, tag: str) -> bool:
    """``autoGenContext``'s choice of the binned context: a binned density, or one tagged."""
    if auto and names and getattr(pdf, "isBinnedDistribution", None) is not None:
        if pdf.isBinnedDistribution(names):
            return True
    return bool(tag) and (tag == "*" or pdf.getAttribute(tag))


def splits(pdf: Any, names: frozenset[str], auto: bool, tag: str) -> bool:
    """``RooSimultaneous::autoGenContext``'s choice: the category asked for, binned allowed."""
    index = getattr(pdf, "index", None)
    return hasattr(pdf, "channels") and index is not None and index.GetName() in names and (
        auto or bool(tag))  # fmt: skip


def _announce(pdf: Any, observables: list[Any]) -> None:
    names = ",".join(one.GetName() for one in observables)
    log(None, INFO, "Generation", "RooBinnedGenContext::ctor() setting up event special "
        f"generator context for sum p.d.f. {pdf.GetName()} for generation of observable(s) "
        f"({names})")  # fmt: skip


def binned_events(pdf: Any, observables: list[Any], count: float, extended: bool,
                  expected_data: bool = False) -> Any:  # fmt: skip
    """``RooBinnedGenContext::generate``: each bin's expected events, Poisson-varied - or made
    to add up to ``count`` - as a weighted dataset of the bins' centres."""
    _announce(pdf, observables)
    return _binned(pdf, observables, count, extended, expected_data)


def _binned(pdf: Any, observables: list[Any], count: float, extended: bool,
            expected_data: bool = False) -> Any:  # fmt: skip
    """:func:`binned_events` once its context is set up."""
    from ..cmdargs import RooCmdArg
    from ..data.datahist import RooDataHist
    from ..data.dataset import RooDataSet

    names = frozenset(one.GetName() for one in observables)
    events = _binned_total(pdf, names, float(count), extended or expected_data)
    hist = RooDataHist("genData", "genData", observables)
    ctx = {name: hist.column(name) for name in names}
    weights = np.asarray(pdf.value(ctx, names), dtype=np.float64) * hist.binVolumes()
    counts = _bin_counts(weights, events, extended, expected_data)
    data = RooDataSet("wu", "wu", observables, RooCmdArg("WeightVar", "weight"))
    data.add_columns({name: hist.column(name) for name in names}, counts)
    return data


def _binned_total(pdf: Any, names: frozenset[str], events: float, exact: bool) -> float:
    """The events asked for, else the expected number - rounded, unless ``exact``.

    Only a density with a yield comes here without a count: ``generate``
    makes ``emptyData`` for the others, and a split context has none.
    """
    if events > 0:
        return events
    expected = pdf.expected(names)
    return float(expected if exact else int(expected + 0.5))


def _bin_counts(weights: Any, events: float, extended: bool, expected_data: bool) -> Any:
    """Each bin's events: expected, Poisson-varied, or varied and made to add up."""
    from .binned import _fixed_total

    if expected_data:
        return weights * events
    drawn = [generator().Poisson(w * events) for w in weights]
    if extended:
        return np.array(drawn, dtype=np.float64)
    return np.array(_fixed_total(weights, drawn, events), dtype=np.float64)


def split_events(sim: Any, variables: list[Any], count: float, extended: bool, auto: bool,
                 tag: str) -> Any:  # fmt: skip
    """``RooSimSplitGenContext::generate``: each state's events from its own context, joined."""
    from ..cmdargs import RooCmdArg
    from ..data.dataset import RooDataSet

    if not sim.canBeExtended():  # said, as RooSimSplitGenContext's constructor says it
        log(sim, ERROR, "Generation", f"RooSimSplitGenContext::RooSimSplitGenContext("
            f"{sim.GetName()}): All components of the simultaneous PDF must be extended PDFs. "
            "Otherwise, it is impossible to calculate the number of events to be generated per "
            "component.")  # fmt: skip
        return None
    channels = list(sim.channels.items())
    states = [_one_state(pdf, variables, auto, tag) for _, pdf in channels]  # all set up first
    wanted = _wanted(sim, channels, variables, count, extended)
    together = zip(channels, states, wanted, strict=False)
    parts = [(label, own, run(number, extended))
             for (label, _), (own, run), number in together]  # fmt: skip
    weighted = any(part.isWeighted() for _, _, part in parts)
    data = RooDataSet("hmaster", "hmaster", variables,
                      *([RooCmdArg("WeightVar", "weight")] if weighted else []))  # fmt: skip
    for label, own, part in parts:
        _add_state(data, part, variables, own, sim.index.GetName(), sim.index.lookupIndex(label))
    return data


def _add_state(data: Any, part: Any, variables: list[Any], own: list[Any], index: str,
               code: int) -> None:  # fmt: skip
    """One state's events joined: its own columns, the others' values now, its index."""
    rows = part.numEntries()
    columns = {one.GetName(): np.full(rows, float(one.getVal())) for one in variables}
    columns.update({one.GetName(): part.column(one.GetName()) for one in own})
    columns[index] = np.full(rows, float(code))
    data.add_columns(columns, part.weights() if part.isWeighted() else None)


def _wanted(sim: Any, channels: list[Any], variables: list[Any], count: float,
            extended: bool) -> list[float]:  # fmt: skip
    """Each state's events: its expected number, or ``count`` shared between them."""
    names = frozenset(one.GetName() for one in variables)
    expected = [pdf.expected(names) for _, pdf in channels]
    events = float(count) if count > 0 else sum(expected)
    log(None, INFO, "Generation", f"RooSimSplitGenContext::{sim.GetName()}:generate: will "
        f"generate {_g(events)} events")  # fmt: skip
    return expected if extended else _shared(expected, events)


def _shared(expected: list[float], events: float) -> list[float]:
    """The events shared between the states, one by one, as their expectations weigh them."""
    total = sum(expected)
    found = [0.0] * len(expected)
    made = 0.0
    while made < events:
        pick = generator().Rndm() * total
        cumulative = 0.0
        for i, one in enumerate(expected):
            if cumulative <= pick < cumulative + one:
                found[i] += 1
                made += 1
                break
            cumulative += one
    return found


def _one_state(pdf: Any, variables: list[Any], auto: bool, tag: str) -> Any:
    """One state's context, set up - the binned one where it is binned - as its variables and
    what draws its events, ``run(number, extended)``."""
    from .generate import Generator

    own = [one for one in variables if one.GetName() in pdf.dependents()]
    if auto_binned(pdf, frozenset(one.GetName() for one in own), auto, tag):
        _announce(pdf, own)
        return own, lambda number, extended: _binned(pdf, own, number, extended)
    made = Generator(pdf, own)

    def run(number: float, extended: bool) -> Any:
        total = generator().Poisson(number) if extended else math.ceil(number)
        return made.sample(total, f"{pdf.GetName()}Data")

    return own, run


def _g(value: float) -> str:
    from ..printing import g

    return g(value)
