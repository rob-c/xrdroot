"""``RooAbsPdf::plotOn``: a density's scale on the frame, then ``RooAbsReal::plotOn`` to draw it.

This is ROOT's function step by step, over the shared option list
(:mod:`.cmdlist`): the fit range made the plotting and normalisation range
if none was given, the options read - warning of any given twice - the
components selected, the curve's scale worked out from the data on the
frame and put in the list as a raw ``Normalization``, a curve-name suffix
added, and the list handed to :func:`.realplot.real_plot`.
"""

from __future__ import annotations

from typing import Any

from ..cmdargs import Commands, RooCmdArg
from ..messages import ERROR, INFO, log
from ..printing import g
from .cmdlist import CmdList
from .curves import _announce_plot, _norm_vars, _selected

__all__ = ["pdf_plot"]

#: ``RooAbsReal::ScaleType``.
RELATIVE, NUM_EVENT, RELATIVE_EXPECTED, RAW = 0, 1, 2, 3
#: What ``plotOn`` says of a fit range it plots in, after the options it added.
FIT_RANGE_ADVICE = (
    " was specified. Plotting / normalising in fit range. To override, do one of the following"
    '\n\t- Clear the automatic fit range attribute: <pdf>.removeStringAttribute("fitrange");'
    '\n\t- Explicitly specify the plotting range: Range("<rangeName>").'
    '\n\t- Explicitly specify where to compute the normalisation: NormRange("<rangeName>").'
    '\n\tThe default (full) range can be denoted with Range("") / NormRange("").'
)


def _fit_range_options(pdf: Any, cmds: CmdList) -> list[RooCmdArg]:
    """The ``Range`` and ``NormRange`` of the fit range, where the options have none."""
    fitted = pdf.getStringAttribute("fitrange")
    added = []
    if fitted and not cmds.has("Range") and not cmds.has("RangeWithName"):
        added.append(RooCmdArg("RangeWithName", fitted, True))
    if fitted and not cmds.has("NormRange"):
        added.append(RooCmdArg("NormRange", fitted))
    if added:
        said = " and ".join(
            "Range()" if a.name == "RangeWithName" else "NormRange()" for a in added
        )
        log(
            pdf,
            INFO,
            "Plotting",
            f"RooAbsPdf::plotOn({pdf.GetName()}) p.d.f was fitted in a subrange "
            f"and no explicit {said}" + FIT_RANGE_ADVICE,
        )
    return added


def _limits(pdf: Any, frame: Any, text: Any) -> list[tuple[float, float]]:
    """Each named range's ends; a name the variable has no range of is said and passed over."""
    var = frame.getPlotVar()
    found = []
    for name in str(text).split(","):
        if name and not var.hasRange(name):
            log(
                pdf,
                ERROR,
                "Plotting",
                f"Range '{name}' not defined for variable '{var.GetName()}'. Ignoring ...",
            )
            continue
        found.append((var.getMin(name or None), var.getMax(name or None)))
    return found


def _custom_ranges(
    pdf: Any, frame: Any, options: Commands
) -> tuple[list[tuple[float, float]], bool]:
    """The ranges the curve is normalised to the data of, said as ``plotOn`` says them."""
    name = pdf.GetName()
    rest = "" if "NormRange" in options else ", curve is normalized to data in {} range"
    limits: list[tuple[float, float]] = []
    adjust = False
    if "Range" in options:
        low, high = options.get("Range", 0), options.get("Range", 1)
        limits, adjust = [(float(low), float(high))], bool(options.get("Range", 2, True))
        log(
            pdf,
            INFO,
            "Plotting",
            f"RooAbsPdf::plotOn({name}) only plotting range [{g(low)},{g(high)}]"
            + rest.format("given" if adjust else "full"),
        )
    elif "RangeWithName" in options:
        text = options.get("RangeWithName")
        limits, adjust = _limits(pdf, frame, text), bool(options.get("RangeWithName", 1, True))
        log(
            pdf,
            INFO,
            "Plotting",
            f"RooAbsPdf::plotOn({name}) only plotting range '{text}'"
            + rest.format("given" if adjust else "full"),
        )
    if "NormRange" in options:
        limits, adjust = _limits(pdf, frame, options.get("NormRange")), True
        log(
            pdf,
            INFO,
            "Plotting",
            f"RooAbsPdf::plotOn({name}) p.d.f. curve is normalized using "
            f"explicit choice of ranges '{options.get('NormRange')}'",
        )
    return limits, adjust


def _scale(pdf: Any, frame: Any, options: Commands, nset: frozenset[str]) -> float:
    """The factor the density is drawn times: events it stands for, times the bin width."""
    scale = float(options.get("Normalization", 0, 1.0))
    kind = int(options.get("Normalization", 1, RELATIVE))
    if kind == RAW:
        return scale
    expected = pdf.expected(nset) if kind == RELATIVE_EXPECTED else 1.0
    if frame.getFitRangeNEvt() and kind == RELATIVE:
        limits, adjust = _custom_ranges(pdf, frame, options)
        if limits and adjust:
            scale *= sum(frame.getFitRangeNEvt(low, high) for low, high in limits) / expected
        elif pdf.canBeExtended() and pdf.expected(nset) > 0:
            scale *= pdf.expected(nset) / expected
        else:
            scale *= frame.getFitRangeNEvt() / expected
    elif kind == RELATIVE_EXPECTED:
        scale *= expected
    elif kind == NUM_EVENT:
        scale /= expected
    return scale * frame.getFitRangeBinW()


def pdf_plot(pdf: Any, frame: Any, cmds: CmdList) -> Any:
    """``RooAbsPdf::plotOn(frame, cmdList)``."""
    from .realplot import real_plot

    with cmds.adding(*_fit_range_options(pdf, cmds)):
        options = cmds.process(f"RooAbsPdf::plotOn({pdf.GetName()})")
        if "Asymmetry" in options:
            from .asymmetry import plot_asymmetry

            return plot_asymmetry(pdf, frame, options)
        nset = _norm_vars(pdf, frame)
        chosen, suffix = _selected(pdf, _components(options))
        cmds.strip("SelectCompSet", "SelectCompSpec")
        scale = _scale(pdf, frame, options, nset)
        norm = RooCmdArg("Normalization", scale, RAW, 1)
        replaced = cmds.replace_or_add(norm)
        try:
            with cmds.adding(RooCmdArg("CurveNameSuffix", suffix)):
                return real_plot(pdf, frame, cmds, chosen, nset)
        finally:
            if replaced is not None:
                cmds.replace_or_add(replaced)
            else:
                cmds.strip_one(norm)


def _components(options: Commands) -> Commands:
    """``Components`` as :func:`.curves._selected` reads it."""
    found = options.get("SelectCompSpec")
    if found is None:
        found = options.get("SelectCompSet")
    return Commands([RooCmdArg("Components", found)] if found is not None else [])


def announce(pdf: Any, frame: Any) -> None:
    _announce_plot(pdf, frame, _norm_vars(pdf, frame))
