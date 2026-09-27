"""``RooAbsPdf::generate``: a dataset of events drawn from a density, as RooFit draws them.

``generate(x, 1000)`` draws a thousand events; ``generate(x, Extended())``
- or a density that says how many it expects - draws a Poisson number of
them first, from RooFit's generator. The events come from the density's
generator context (:mod:`.contexts`), and the dataset is called after the
density: ``gaussData``, "Generated From gauss".
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..cmdargs import RooCmdArg, commands
from ..collections import as_list
from ..messages import ERROR, log
from ..rng import generator
from .contexts import context_for

__all__ = ["generate", "generate_binned", "parse"]


def parse(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[list[Any], Any, Any]:
    """The variables, the number of events asked for (or ``None``), and the options."""
    positional = [a for a in args if not isinstance(a, RooCmdArg)]
    options = commands([a for a in args if isinstance(a, RooCmdArg)], kwargs)
    variables = as_list(positional[0]) if positional else []
    count = positional[1] if len(positional) > 1 else options.get("NumEvents")
    return variables, count, options


def _how_many(pdf: Any, names: frozenset[str], count: Any, options: Any) -> int:
    """``RooAbsGenContext::generate``: the events asked for - or expected - Poisson-varied if extended."""
    extended = bool(options.get("Extended", 0, False))
    wanted = float(count) if count is not None else 0.0
    if wanted <= 0:
        if not pdf.canBeExtended():
            log(pdf, ERROR, "Generation", f"RooGenContext::{pdf.GetName()}:generate: PDF not "
                "extendable: cannot calculate expected number of events")  # fmt: skip
            return -1
        wanted = pdf.expected(names)
    if extended:
        return int(generator().Poisson(wanted))
    return int(math.ceil(wanted))  # events are drawn while there are fewer than asked for


def _uniform(var: Any) -> float:
    """``randomize``: a value drawn uniformly - a category's state by its order of definition."""
    rng = generator()
    if hasattr(var, "lookupIndex"):
        labels = list(var.states())
        return float(var.states()[labels[rng.Integer(len(labels))]])
    return var.getMin() + rng.Rndm() * (var.getMax() - var.getMin())


def generate(pdf: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``pdf.generate(vars, n, options...)``: a new dataset of generated events."""
    from ..data.dataset import RooDataSet
    from ..integration import announce

    from .proto import ProtoFeed

    variables, count, options = parse(args, kwargs)
    feed = ProtoFeed(pdf, options.get("ProtoData"))
    names = frozenset(one.GetName() for one in variables) & pdf.dependents() - feed.names
    uniform = sorted((one for one in variables if one.GetName() not in names), key=lambda v: v.GetName(), reverse=True)
    for _ in range(2):  # the generator's own copy of the density, and its context's
        announce(pdf, names)
    context = feed.context(pdf, names)
    total = _how_many(pdf, names, feed.count(count), options)
    if total < 0:
        return None
    name = options.get("Name") or f"{pdf.GetName()}Data"
    variables = variables + feed.extra(variables)
    data = RooDataSet(name, f"Generated From {pdf.GetName()}", variables)
    saved = [(one, one.getVal()) for one in pdf.leaves() if one.GetName() in names | feed.names]
    rows = []
    for i in range(total):
        known = feed.load(i)
        row = context.event(total - i)
        row.update(known)
        row.update({one.GetName(): _uniform(one) for one in uniform})
        rows.append(row)
    for one, value in saved:
        one.load_value(value)
    data.add_columns({one.GetName(): np.array([r[one.GetName()] for r in rows], dtype=np.float64)
                      for one in variables})  # fmt: skip
    return data


def generate_binned(pdf: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``generateBinned``: expected bin contents, Poisson-varied - or not, with ``ExpectedData``."""
    from .binned import binned

    variables, count, options = parse(args, kwargs)
    return binned(pdf, variables, count, options)
