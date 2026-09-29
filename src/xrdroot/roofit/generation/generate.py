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
    """``RooAbsGenContext::generate``: the events asked for - or expected, or as many as the
    prototype data has - Poisson-varied if extended."""
    extended = bool(options.get("Extended", 0, False))
    wanted = float(count) if count is not None else 0.0
    proto = options.get("ProtoData")
    if wanted <= 0 and proto is not None:
        wanted = float(proto.numEntries())
    if wanted <= 0:  # a density without a yield has made emptyData by now
        wanted = pdf.expected(names)
    if extended:
        return generator().Poisson(wanted)
    return math.ceil(wanted)  # events are drawn while there are fewer than asked for


def _uniform(var: Any) -> float:
    """``randomize``: a value drawn uniformly - a category's state by its order of definition."""
    rng = generator()
    if hasattr(var, "lookupIndex"):
        labels = list(var.states())
        return float(var.states()[labels[rng.Integer(len(labels))]])
    return float(var.getMin() + rng.Rndm() * (var.getMax() - var.getMin()))


class Generator:
    """A density's generator context, made once, drawing sample after sample:
    ``RooAbsGenContext``."""

    def __init__(self, pdf: Any, variables: list[Any], proto: Any = None) -> None:
        from ..integration import announce

        self.pdf = pdf
        self.proto = proto
        asked = {one.GetName() for one in variables}
        self.taken = _taken(proto, asked)
        self.variables = variables + self.taken
        self.names = frozenset(asked) & pdf.dependents()
        self.targets = _targets(pdf, {one.GetName() for one in self.taken})
        self.uniform = _uniformly_drawn(variables, self.names)
        for _ in range(2):  # the generator's own copy of the density, and its context's
            announce(pdf, self.names, normalising=True)
        self.context = _context(pdf, self.names, frozenset(one.GetName() for one in self.taken))

    def sample(self, total: int, name: str) -> Any:
        """``generate(n)``: a dataset of ``total`` events, the density's variables left as they
        were."""
        from ..data.dataset import RooDataSet

        data = RooDataSet(name, f"Generated From {self.pdf.GetTitle()}", self.variables)
        saved = self._saved()
        rows = [self._event(i, total) for i in range(total)]
        for one, value in saved:
            one.load_value(value)
        data.add_columns(
            {
                one.GetName(): np.array([r[one.GetName()] for r in rows], dtype=np.float64)
                for one in self.variables
            }
        )
        return data

    def _saved(self) -> list[tuple[Any, float]]:
        """The values generating moves - the observables', the prototype's - to be put back."""
        saved = [(one, one.getVal()) for one in self.pdf.leaves() if one.GetName() in self.names]
        return saved + [(one, one.getVal()) for group in self.targets.values() for one in group]

    def _event(self, i: int, total: int) -> dict[str, float]:
        """Event ``i``: the prototype's values, then the context's draws, then the uniform ones."""
        loaded = self._load(i)
        row: dict[str, float] = self.context.event(total - i)
        row.update(loaded)
        row.update({one.GetName(): _uniform(one) for one in self.uniform})
        return row

    def _load(self, i: int) -> dict[str, float]:
        """The prototype data's event ``i`` - round again if there are more to draw - in its
        variables."""
        if not self.taken:
            return {}
        index = i % self.proto.numEntries()
        loaded = {
            one.GetName(): float(self.proto.column(one.GetName())[index]) for one in self.taken
        }
        for name, value in loaded.items():
            for one in self.targets.get(name, []):
                one.load_value(value)
        return loaded


def _taken(proto: Any, asked: set[str]) -> list[Any]:
    """The prototype's variables that are not generated: taken from it, event by event."""
    return [one for one in (proto.get() if proto is not None else []) if one.GetName() not in asked]


def _uniformly_drawn(variables: list[Any], names: frozenset[str]) -> list[Any]:
    """The variables asked for that the density does not depend on, in the order asked for.

    ``RooGenContext`` randomizes them in the order of its snapshot of the
    variables asked for - so ``{x, b0flav, tagCat}`` draws ``b0flav`` first.
    From a Python set that order is the set's, which PyROOT's proxies hash
    by address: ROOT's own order then changes from one process to the next.
    """
    return [one for one in variables if one.GetName() not in names]


def _context(pdf: Any, names: frozenset[str], proto: frozenset[str]) -> Any:
    """The density's context, told the prototype's variables if it has any - and wants to know."""
    import inspect

    make = getattr(pdf, "gen_context", None)
    if proto and make is not None and "proto" in inspect.signature(make).parameters:
        return make(names, proto=proto)
    return context_for(pdf, names, proto & pdf.dependents() if proto else None)


def _targets(pdf: Any, names: set[str]) -> dict[str, list[Any]]:
    """The objects under ``pdf`` called ``names``: its variables, and those its ranges end at."""
    found: dict[str, list[Any]] = {}
    for leaf in pdf.leaves():
        ends = leaf.getBinning().servers() if hasattr(leaf, "getBinning") else []
        for one in [leaf, *ends]:
            if one.GetName() in names and all(
                one is not other for other in found.get(one.GetName(), [])
            ):
                found.setdefault(one.GetName(), []).append(one)
    return found


def generate(pdf: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``pdf.generate(vars, n, options...)``: a new dataset of generated events."""
    variables, count, options = parse(args, kwargs)
    if options.get("ProtoData") is None:
        found = _without_context(pdf, variables, count, options)
        if found is not None:
            return _named(found[0], options) if found else None
    proto = options.get("ProtoData")
    made = Generator(pdf, variables, proto)
    if not count and proto is not None and proto.numEntries() == 0:
        return _named(_empty(variables, options), options)
    total = _how_many(pdf, made.names, count, options)
    return made.sample(total, options.get("Name") or f"{pdf.GetName()}Data")


def _without_context(pdf: Any, variables: list[Any], count: Any, options: Any) -> Any:
    """``[data]`` made other than event by event - ``emptyData`` for no events of a density
    without a yield, as ``RooAbsPdf::generate`` makes it - or ``[]`` for none, or ``None``."""
    if not count and not pdf.canBeExtended():
        return [_empty(variables, options)]
    special = _special(pdf, variables, count, options)
    if special is None:
        return None
    found = special()
    return [] if found is None else [found]


def _named(data: Any, options: Any) -> Any:
    """``data``, renamed as ``Name`` asks."""
    if options.get("Name"):
        data.SetName(options.get("Name"))
    return data


def _empty(variables: list[Any], options: Any) -> Any:
    """``RooAbsPdf::generate``'s ``emptyData``: no events asked for, none to be had."""
    from ..data.dataset import RooDataSet

    return RooDataSet("emptyData", "emptyData", variables)


def _special(pdf: Any, variables: list[Any], count: Any, options: Any) -> Any:
    """The binned or split context's generation, where ``autoGenContext`` would choose one."""
    from .split import auto_binned, binned_events, split_events, splits

    auto = bool(options.get("AutoBinned", 0, True))
    expected_data = bool(options.get("ExpectedData", 0, False))
    every = expected_data or bool(options.every("AllBinned"))  # binnedTag "*", as RooFit
    tag = "*" if every else str(options.get("GenBinned", 0, "") or "")
    extended = bool(options.get("Extended", 0, False))
    names = frozenset(one.GetName() for one in variables)
    events = _special_count(pdf, names, count, extended)
    if splits(pdf, names, auto, tag):
        return lambda: split_events(pdf, variables, events, extended, auto, tag)
    own = names & pdf.dependents()
    if auto_binned(pdf, own, auto, tag):
        observables = [one for one in variables if one.GetName() in own]
        return lambda: binned_events(pdf, observables, events, extended, expected_data)
    return None


def _special_count(pdf: Any, names: frozenset[str], count: Any, extended: bool) -> float:
    """The events asked for - none, unless extended, when the density expects them."""
    events = float(count) if count is not None else 0.0
    return pdf.expected(names) if extended and events == 0 else events


def generate_binned(pdf: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``generateBinned``: expected bin contents, Poisson-varied - or not, with ``ExpectedData``."""
    from .binned import binned

    variables, count, options = parse(args, kwargs)
    return binned(pdf, variables, count, options)
