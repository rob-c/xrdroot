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
    from ..data.datahist import RooDataHist
    from ..data.dataset import RooDataSet
    from .binned import _fixed_total

    names = frozenset(one.GetName() for one in observables)
    _announce(pdf, observables)
    events = float(count)
    if events <= 0:
        if not pdf.canBeExtended():
            log(None, ERROR, "InputArguments", f"RooAbsPdf::generateBinned({pdf.GetName()}) "
                "ERROR: No event count provided and p.d.f does not provide expected number of "
                "events")  # fmt: skip
            return None
        expected = pdf.expected(names)
        events = expected if (expected_data or extended) else float(int(expected + 0.5))
    hist = RooDataHist("genData", "genData", observables)
    ctx = {name: hist.column(name) for name in names}
    weights = np.asarray(pdf.value(ctx, names), dtype=np.float64) * hist.binVolumes()
    if expected_data:
        counts = weights * events
    elif extended:
        counts = np.array([generator().Poisson(w * events) for w in weights], dtype=np.float64)
    else:
        drawn = [generator().Poisson(w * events) for w in weights]
        counts = np.array(_fixed_total(weights, drawn, events), dtype=np.float64)
    from ..cmdargs import RooCmdArg

    data = RooDataSet("wu", "wu", observables, RooCmdArg("WeightVar", "weight"))
    data.add_columns({name: hist.column(name) for name in names}, counts)
    return data


def split_events(sim: Any, variables: list[Any], count: float, extended: bool, auto: bool,
                 tag: str) -> Any:  # fmt: skip
    """``RooSimSplitGenContext::generate``: each state's events from its own context, joined."""
    from ..cmdargs import RooCmdArg
    from ..data.dataset import RooDataSet

    if not sim.canBeExtended():
        log(sim, ERROR, "Generation", f"RooSimSplitGenContext::RooSimSplitGenContext("
            f"{sim.GetName()}): All components of the simultaneous PDF must be extended PDFs. "
            "Otherwise, it is impossible to calculate the number of events to be generated per "
            "component.")  # fmt: skip
        return None
    names = frozenset(one.GetName() for one in variables)
    channels = list(sim.channels.items())
    expected = [pdf.expected(names) for _, pdf in channels]
    events = float(count) if count > 0 else sum(expected)
    log(None, INFO, "Generation", f"RooSimSplitGenContext::{sim.GetName()}:generate: will "
        f"generate {_g(events)} events")  # fmt: skip
    wanted = expected if extended else _shared(expected, events)
    index = sim.index
    data = RooDataSet("hmaster", "hmaster", variables, RooCmdArg("WeightVar", "weight"))
    for (label, pdf), number in zip(channels, wanted):
        own = [one for one in variables if one.GetName() in pdf.dependents()]
        part = _one_state(pdf, own, number, extended, auto, tag)
        if part is None:
            continue
        rows = part.numEntries()
        columns = {one.GetName(): np.full(rows, float(one.getVal())) for one in variables}
        columns.update({one.GetName(): part.column(one.GetName()) for one in own})
        columns[index.GetName()] = np.full(rows, float(index.lookupIndex(label)))
        data.add_columns(columns, part.weights() if part.isWeighted() else None)
    return data


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
