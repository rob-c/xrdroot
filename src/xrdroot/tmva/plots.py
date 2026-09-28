"""``TransformationHandler::PlotVariables``: each variable's distribution, per class, for the file.

For every variable (and target) and every class TMVA books a 40-bin
histogram over the mean plus or minus eight RMS - clipped to the range -
and, with no more than 20 variables, a 300 by 300 scatter plot and a
profile for every pair, kept in ``CorrelationPlots``. The same histograms
give the Factory's ranking of the variables by their separation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..hist import Histogram
from ..profile import Profile
from . import hists
from .dataset import DataSetInfo, Events
from .evaluation import separation_of_hists
from .variables import VariableInfo

__all__ = ["VariablePlots", "plot_variables"]

#: ``gConfig().GetVariablePlotting()``.
TIMES_RMS, NBINS_1D, NBINS_2D, MAX_SCATTER = 8.0, 40, 300, 20


def _range(
    info: VariableInfo, stats: tuple[float, float, float, float]
) -> tuple[int, float, float]:
    """The bins and range of one variable's histogram, as ``PlotVariables`` works them out."""
    mean, rms, low, high = stats
    if info.vartype == "I":
        start, stop = int(round(low)), int(round(high + 1))
        return stop - start, float(start), float(stop)
    xmin = max(low, mean - TIMES_RMS * rms)
    xmax = min(high, mean + TIMES_RMS * rms)
    if xmin >= xmax:
        xmax = xmin * 1.1
    if xmin >= xmax:
        xmax = xmin + 1
    xmax += (xmax - xmin) / NBINS_1D
    return NBINS_1D, xmin, xmax


def _unit(title: str, unit: str) -> str:
    return title if not unit else f"{title}  [{unit}]"


class VariablePlots:
    """The histograms of one set of events, by where they go, and the separation of each variable."""

    def __init__(self) -> None:
        self.histograms: list[Histogram] = []
        self.correlations: list[Histogram] = []
        self.separations: list[tuple[str, float]] = []
        #: A regression's rankings: each title and its variables' values.
        self.rankings: list[tuple[str, list[tuple[str, float]]]] = []


def _infos(dsi: DataSetInfo) -> list[tuple[VariableInfo, str]]:
    kinds = [(info, "") for info in dsi.variables]
    suffix = "_target" if len(dsi.targets) == 1 else ""
    return kinds + [(info, suffix) for info in dsi.targets]


def plot_variables(
    dsi: DataSetInfo, events: Events, stats: Any, suffix: str = "", axis_note: str = ""
) -> VariablePlots:
    """The plots of ``events`` - already transformed - named ``<var>__<class><suffix>``.

    ``stats`` are the handler's means, RMS and ranges over all classes;
    ``axis_note`` is the transformation's name that the axis titles carry.
    """
    made = VariablePlots()
    values = np.column_stack([events.values, events.targets])
    infos = _infos(dsi)
    booked: dict[tuple[int, int], Histogram] = {}
    for index, (info, extra) in enumerate(infos):
        spec = _range(info, tuple(float(part[index]) for part in stats))
        title = info.title + (f" ({axis_note})" if axis_note else "")
        for cls in dsi.classes:
            name = f"{info.internal}__{cls.name}{extra}{suffix}"
            width = (spec[2] - spec[1]) / spec[0]
            ytitle = f"dN_{{ }}/^{{ }}{width:.3g} {info.unit}"
            histogram = hists.book(name, f"{info.title};{_unit(title, info.unit)};{ytitle}", *spec)
            mask = events.classes == cls.number
            histogram.fill(values[mask, index], weight=events.weights[mask])
            booked[(cls.number, index)] = histogram
            made.histograms.append(histogram)
    if len(infos) <= MAX_SCATTER:
        made.correlations = _scatters(dsi, events, values, stats, suffix)
    _rank(dsi, booked, made)
    return made


def _scatters(
    dsi: DataSetInfo, events: Events, values: Any, stats: Any, suffix: str
) -> list[Histogram]:
    """Every pair's scatter plot and profile, per class, as ``PlotVariables`` books them."""
    infos = _infos(dsi)
    made: list[Histogram] = []
    low, high = stats[2], stats[3]
    for i, (info, _) in enumerate(infos):
        for j in range(i + 1, len(infos)):
            other = infos[j][0]
            for cls in dsi.classes:
                mask = events.classes == cls.number
                x, y, w = values[mask, i], values[mask, j], events.weights[mask]
                label = f"{other.title} versus {info.title} ({cls.name}){suffix}"
                scatter = Histogram.book(
                    f"scat_{other.internal}_vs_{info.internal}_{cls.name}{suffix}",
                    (NBINS_2D, float(low[i]), float(high[i])),
                    (NBINS_2D, float(low[j]), float(high[j])),
                    title=label,
                    kind="F",
                )
                scatter.fill(x, y, weight=w)
                profile = Profile.book(
                    f"prof_{other.internal}_vs_{info.internal}_{cls.name}{suffix}",
                    (NBINS_1D, float(low[i]), float(high[i])),
                    title=f"profile {label}",
                )
                profile.fill(x, y, weight=w)
                made += [scatter, profile]
    return made


def _rank(dsi: DataSetInfo, booked: dict[tuple[int, int], Histogram], made: VariablePlots) -> None:
    """The separation of each variable's signal and background, for a two-class classification."""
    signal, background = dsi.GetClassInfo("Signal"), dsi.GetClassInfo("Background")
    if dsi.targets or len(dsi.classes) != 2 or signal is None or background is None:
        return
    for index, info in enumerate(dsi.variables):
        value = separation_of_hists(
            booked[(signal.number, index)], booked[(background.number, index)]
        )
        made.separations.append((info.title, value))
