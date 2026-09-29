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
    from ..cmdargs import RooCmdArg
    from ..data.datahist import RooDataHist
    from ..data.dataset import RooDataSet

    names = frozenset(one.GetName() for one in observables)
    _announce(pdf, observables)
    events = _binned_total(pdf, names, float(count), extended or expected_data)
    if events is None:
        return None
    hist = RooDataHist("genData", "genData", observables)
    ctx = {name: hist.column(name) for name in names}
    weights = np.asarray(pdf.value(ctx, names), dtype=np.float64) * hist.binVolumes()
    counts = _bin_counts(weights, events, extended, expected_data)
    data = RooDataSet("wu", "wu", observables, RooCmdArg("WeightVar", "weight"))
    data.add_columns({name: hist.column(name) for name in names}, counts)
    return data


def _binned_total(pdf: Any, names: frozenset[str], events: float, exact: bool) -> Any:
    """The events asked for, else the expected number - rounded, unless ``exact`` - or
    ``None``, said, for a density that expects none."""
    if events > 0:
        return events
    if not pdf.canBeExtended():
        log(None, ERROR, "InputArguments", f"RooAbsPdf::generateBinned({pdf.GetName()}) "
            "ERROR: No event count provided and p.d.f does not provide expected number of "
            "events")  # fmt: skip
        return None
    expected = pdf.expected(names)
    return expected if exact else float(int(expected + 0.5))


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
    wanted = _wanted(sim, channels, variables, count, extended)
    index = sim.index
    data = RooDataSet("hmaster", "hmaster", variables, RooCmdArg("WeightVar", "weight"))
    for (label, pdf), number in zip(channels, wanted):
        own = [one for one in variables if one.GetName() in pdf.dependents()]
        part = _one_state(pdf, own, number, extended, auto, tag)
        if part is not None:
            _add_state(data, part, variables, own, index.GetName(), index.lookupIndex(label))
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


def _one_state(pdf: Any, own: list[Any], number: float, extended: bool, auto: bool,
               tag: str) -> Any:  # fmt: skip
    """One state's events, from the binned context where it is binned."""
    from ..cmdargs import RooCmdArg

    names = frozenset(one.GetName() for one in own)
    if auto_binned(pdf, names, auto, tag):
        return binned_events(pdf, own, number, extended)
    if extended:
        return pdf.generate(own, RooCmdArg("Extended"))
    return pdf.generate(own, RooCmdArg("NumEvents", int(number)))


def _g(value: float) -> str:
    from ..printing import g

    return g(value)
