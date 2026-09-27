"""``RooSimultaneous``: one density per state of a category, for fitting several samples at once.

``RooSimultaneous("s", "s", {"physics": model, "control": model_ctl},
sample)`` is ``model`` for the events of state ``physics`` and ``model_ctl``
for those of ``control``. Its likelihood is the sum of each channel's, each
channel's density normalised over its own observables, plus - as RooFit
normalises a simultaneous density over the category too - ``log`` of the
number of channels for every event (``RooNLLVarNew::setSimCount``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..collections import as_list
from ..pdf import CAN_BE_EXTENDED, CAN_NOT_BE_EXTENDED, MUST_BE_EXTENDED, RooAbsPdf
from ..real import Context

__all__ = ["RooSimultaneous"]


class RooSimultaneous(RooAbsPdf):
    """A density per state of an index category."""

    def __init__(self, name: Any, title: Any, *args: Any) -> None:
        super().__init__(name, title)
        index = next(a for a in args if hasattr(a, "lookupIndex"))
        self.index = self._proxy("indexCat", index)
        self.channels: dict[str, Any] = {}
        self.chosen = self._list_proxy("!pdfs", [])
        for arg in args:
            if isinstance(arg, dict):
                for label, pdf in arg.items():
                    self.addPdf(pdf, label)
        lists = [a for a in args if not isinstance(a, dict) and a is not index]
        if len(lists) == 1:
            for pdf, (label, _) in zip(as_list(lists[0]), index.states().items()):
                self.addPdf(pdf, label)

    def addPdf(self, pdf: Any, label: str) -> bool:
        if str(label) in self.channels:
            return True
        self.channels[str(label)] = pdf
        self.chosen.add(pdf)
        return False

    def getPdf(self, label: Any) -> Any:
        return self.channels.get(str(label))

    def indexCat(self) -> Any:
        return self.index

    def servers(self) -> list[Any]:
        return [self.index, *(self.channels[label] for label in sorted(self.channels))]

    def plotOn(self, frame: Any, *args: Any, **kwargs: Any) -> Any:
        return plot_simultaneous(self, frame, args, kwargs)

    # -- values -------------------------------------------------------------------

    def _state_of(self, ctx: Context) -> Any:
        return np.asarray(ctx.get(self.index.GetName(), self.index.getIndex()))

    def value(self, ctx: Context, nset: Any = None, rng: Any = None) -> Any:
        states = self._state_of(ctx)
        found: Any = np.zeros(np.shape(states)) if np.ndim(states) else 0.0
        for label, pdf in self.channels.items():
            number = self.index.lookupIndex(label)
            own = frozenset(nset or ()) & pdf.dependents() if nset else nset
            part = pdf.value(ctx, own, rng)
            found = np.where(states == number, part, found)
        return found if np.ndim(found) else float(found)

    def compute(self, ctx: Context) -> Any:
        return self.value(ctx, None)

    def selfNormalized(self) -> bool:
        return True

    def channel_terms(self, nll: Any) -> float:
        """The likelihood of each channel's events, and ``N log(channels)``, summed."""
        states = np.asarray(nll.columns[self.index.GetName()])
        total = 0.0
        count = len(self.channels)
        for label, pdf in self.channels.items():
            keep = states == self.index.lookupIndex(label)
            if not keep.any():
                continue
            total += nll.channel(pdf, keep, count if count > 1 else 0)
        return total

    # -- extended -----------------------------------------------------------------

    def extendMode(self) -> int:
        modes = [pdf.extendMode() for pdf in self.channels.values()]
        if modes and all(m != CAN_NOT_BE_EXTENDED for m in modes):
            return MUST_BE_EXTENDED if all(m == MUST_BE_EXTENDED for m in modes) else CAN_BE_EXTENDED
        return CAN_NOT_BE_EXTENDED

    def expected(self, nset: Any, rng: Any = None) -> float:
        return float(sum(pdf.expected(nset, rng) for pdf in self.channels.values()))

    def printMetaArgs(self) -> str:
        return ""


class _Projection(RooAbsPdf):
    """The simultaneous density averaged over its category as the projection data weigh the states."""

    def __init__(self, sim: Any, weights: dict[str, float]) -> None:
        super().__init__(sim.GetName(), sim.GetTitle())
        self.sim = sim
        self.weights = weights
        self.parts = self._list_proxy("!pdfs", [sim.channels[label] for label in weights])

    def value(self, ctx: Context, nset: Any = None, rng: Any = None) -> Any:
        from ..selection import active

        total: Any = 0.0
        for (label, weight), pdf in zip(self.weights.items(), self.parts):
            if active(pdf):
                own = frozenset(nset or ()) & pdf.dependents()
                total = total + weight * pdf.value(ctx, own, rng)
        return total

    def compute(self, ctx: Context) -> Any:
        return self.value(ctx, None)

    def selfNormalized(self) -> bool:
        return True

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return names

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        return self.fraction(names, ctx, names, rng)

    def fraction(self, names: frozenset[str], ctx: Context, nset: Any, rng: Any,
                 norm_rng: Any = None) -> Any:  # fmt: skip
        total: Any = 0.0
        for (label, weight), pdf in zip(self.weights.items(), self.parts):
            own = frozenset(nset or ()) & pdf.dependents()
            total = total + weight * pdf.fraction(names & own, ctx, own, rng, norm_rng)
        return total


def plot_simultaneous(sim: Any, frame: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``RooSimultaneous::plotOn``: with ``ProjWData``, the channels averaged as the data weigh them."""
    from ..messages import INFO, log
    from ..plot.cmdlist import CmdList
    from ..plot.pdfplot import pdf_plot

    cmds = CmdList.of(args, kwargs)
    projection = cmds.find("ProjWData")
    sliced = cmds.find("Slice")
    if projection is None and sliced is None:
        return pdf_plot(sim, frame, cmds)
    data = projection.value(1) if projection is not None and projection.value(1) is not None else (
        projection.value(0) if projection is not None else None)
    weights = _weights(sim, data, sliced)
    if not (cmds.has("SelectCompSpec") or cmds.has("SelectCompSet")):
        for _ in range(2):
            log(sim, INFO, "Plotting", f"RooSimultaneous::plotOn({sim.GetName()}) plot on "
                f"{frame.getPlotVar().GetName()} averages with data index category ({sim.index.GetName()})")
    cmds.strip("ProjWData", "Slice")
    return pdf_plot(_Projection(sim, weights), frame, cmds)


def _weights(sim: Any, data: Any, sliced: Any) -> dict[str, float]:
    """Each channel's share: its events in the projection data - or one, for the slice chosen."""
    labels = sorted(sim.channels)
    if sliced is not None:
        return {str(sliced.value(1)): 1.0}
    column = data.column(sim.index.GetName())
    weights = data.weights()
    total = float(np.sum(weights))
    return {label: float(np.sum(weights[column == sim.index.lookupIndex(label)])) / total
            for label in labels}  # fmt: skip
