"""``TH2::FitSlicesX`` and ``FitSlicesY``: a function fitted to each slice of a 2D histogram.

``FitSlicesY`` projects each x bin - or each group of ``gN`` of them - onto y,
fits the function (a Gaussian unless one is given) to the projection, and
fills one histogram per parameter, ``<name>_<i>``, with the fitted values
and their errors, and ``<name>_chi2`` with each fit's chi-square per degree
of freedom - as ``TH2::DoFitSlices`` does, its messages, the slices it skips
and the histograms it replaces in the current directory included.
"""

from __future__ import annotations

import re
from typing import Any

from ...errors import UnsupportedFeatureError
from ...fillrandom import AXIS_RANGE
from .directories import current_directory
from .messages import message

__all__: list[str] = []


def _bins(axis: Any, first: int, last: int) -> tuple[int, int]:
    """The outer bins the slices are taken over, as ``DoFitSlices`` settles them."""
    nbins = axis.GetNbins()
    if last < first and axis.TestBit(AXIS_RANGE):
        first, last = axis.GetFirst(), axis.GetLast()
        if first == 0 and last == 0:
            first, last = 1, nbins
    first = max(first, 0)
    if last < 0 or last > nbins + 1:
        last = nbins + 1
    if last < first:
        first, last = 0, nbins + 1
    return first, last


def _options(option: str) -> tuple[str, str, int, int]:
    """The fit's options, the projection's, the bins grouped by ``gN`` and the step."""
    text, projection = str(option), "e"
    bracket = re.search(r"\[.*?\]", text)
    if bracket:
        raise UnsupportedFeatureError(
            f"the slices of FitSlices through the graphical cut {bracket.group(0)} are not "
            "made here: a projection here takes every bin of its range, and no TCutG"
        )
    text = text.lower()
    group = 1
    for n in (2, 3, 4, 5):
        if f"g{n}" in text:
            group, text = n, text.replace(f"g{n}", "")
    return text, projection, group, (1 if "s" in text else group)


def _booked(h2: Any, outer: Any, first: int, last: int, step: int, name: str, title: str) -> Any:
    """One of the histograms the slices fill, replacing one of its name in the directory."""
    from .hists import TH1D

    here = current_directory()
    old = here.FindObject(name)
    if old is not None:
        here.Remove(old)
    ranged = outer.TestBit(AXIS_RANGE)
    low, high = (max(first, 1), min(last, outer.GetNbins())) if ranged else (1, outer.GetNbins())
    edges = list(outer.GetXbins())
    if edges:
        made = TH1D(name, title, high - low + 1, edges[low - 1 : high])
    else:
        made = TH1D(
            name,
            title,
            (high - low + 1) // step,
            outer.GetBinLowEdge(low),
            outer.GetBinUpEdge(high),
        )
    made.GetXaxis().SetTitle(outer.GetTitle())
    return made


def fit_slices(
    h2: Any, on_x: bool, f1: Any, first: int, last: int, cut: int, option: str, arr: Any
) -> None:
    """``TH2::DoFitSlices``: slices along the other axis, each fitted, the results booked."""
    outer, inner = (h2.GetYaxis(), h2.GetXaxis()) if on_x else (h2.GetXaxis(), h2.GetYaxis())
    first, last = _bins(outer, int(first), int(last))
    opt, projection, group, step = _options(option)
    f1 = _function(f1, inner)
    npar = f1.GetNpar()
    if npar <= 0:
        return
    saved = list(f1.GetParameters())[:npar]
    name = h2.GetName()
    hists = [
        _booked(
            h2,
            outer,
            first,
            last,
            step,
            f"{name}_{i}",
            f"Fitted value of par[{i}]={f1.GetParName(i)}",
        )
        for i in range(npar)
    ]
    chi2 = _booked(h2, outer, first, last, step, f"{name}_chi2", "chisquare")
    if arr is not None:
        arr.SetOwner()
        for one in [*hists, chi2]:
            arr.Add(one)
    where = f"{h2.ClassName()}::DoFitSlices"
    project = h2.ProjectionX if on_x else h2.ProjectionY
    for bin in range(first, last - group + 2, step):
        hp = project("_temp", bin, bin + group - 1, projection)
        _fitted(hp, f1, saved, (hists, chi2), (outer, bin, group, cut), opt + " n ", where)
        current_directory().Remove(hp)


def _function(f1: Any, inner: Any) -> Any:
    """The function fitted: a Gaussian over the inner axis unless one is given."""
    if f1 is not None:
        return f1
    from .troot import gROOT

    found = gROOT.GetFunction("gaus")
    found.SetRange(inner.GetXmin(), inner.GetXmax())
    return found


def _fitted(
    hp: Any,
    f1: Any,
    saved: list[float],
    booked: tuple[list[Any], Any],
    at: tuple[Any, int, int, int],
    opt: str,
    where: str,
) -> None:
    """One slice fitted and its results filled in - or skipped, as ROOT skips one of too few
    entries, or whose fit had too few points, saying so unless ``q``uiet."""
    (hists, chi2), (outer, bin, group, cut) = booked, at
    quiet, entries = "q" in opt, hp.GetEntries()
    if entries <= 0 or entries < cut:
        if not quiet:
            message(
                "Info",
                where,
                "Slice %d skipped, the number of entries is zero or smaller "
                "than the given cut value, n=%f",
                bin,
                entries,
            )
        return
    f1.SetParameters(saved)
    on = hists[0].FindBin(outer.GetBinCenter(bin + group // 2))
    if not quiet:
        axis = hists[0].GetXaxis()
        message(
            "Info", where, "Slice fit %d (%f,%f)", on, axis.GetBinLowEdge(on), axis.GetBinUpEdge(on)
        )
    hp.Fit(f1, opt)
    points, npar = f1.GetNumberFitPoints(), len(saved)
    if points > npar and points >= cut:
        for index, one in enumerate(hists):
            one.SetBinContent(on, f1.GetParameter(index))
            one.SetBinError(on, f1.GetParError(index))
        chi2.SetBinContent(on, f1.GetChisquare() / (points - npar))
    elif not quiet:
        message(
            "Info",
            where,
            "Fitted slice %d skipped, the number of fitted points is too small, n=%d",
            on,
            points,
        )
