"""Where a density's curve is drawn and what it is normalised to: ``Range`` and ``NormRange``.

A density fitted in a range remembers it (the ``fitrange`` attribute), and
is then drawn over that range, normalised to the data in it, unless told
otherwise - saying so, as ``RooAbsPdf::plotOn`` does. ``Range("a,b")``
draws one curve per range; each is the density normalised over the full
range, divided by its fraction in the normalisation range and scaled to the
events there (``postRangeFracScale``), so the pieces join up.
"""

from __future__ import annotations

from typing import Any

from ..cmdargs import Commands, RooCmdArg
from ..messages import ERROR, INFO, log
from ..printing import g

__all__ = ["Plan", "plan"]

#: ``RooAbsReal::ScaleType``.
RELATIVE, NUM_EVENT, RELATIVE_EXPECTED, RAW = 0, 1, 2, 3


class Plan:
    """What to draw: the scale, the pieces of range, the range normalised in, and the wings."""

    def __init__(self) -> None:
        self.scale = 1.0
        self.pieces: list[tuple[float, float]] = []
        self.norm_range: str | None = None
        self.post_scale = False
        self.wings = True
        self.suffix = ""


def _fit_range(pdf: Any, options: Commands) -> Commands:
    """The options with the density's fit range as ``Range`` and ``NormRange`` where none were given."""
    fitted = pdf.getStringAttribute("fitrange")
    if not fitted:
        return options
    added = []
    if "Range" not in options:
        added.append("Range()")
        options.given.append(RooCmdArg("Range", fitted, True))
    if "NormRange" not in options:
        added.append("NormRange()")
        options.given.append(RooCmdArg("NormRange", fitted))
    if added:
        log(pdf, INFO, "Plotting", f"RooAbsPdf::plotOn({pdf.GetName()}) p.d.f was fitted in a subrange "
            f"and no explicit {' and '.join(added)} was specified. Plotting / normalising in fit range. "
            "To override, do one of the following\n\t- Clear the automatic fit range attribute: "
            "<pdf>.removeStringAttribute(\"fitrange\");\n\t- Explicitly specify the plotting range: "
            "Range(\"<rangeName>\").\n\t- Explicitly specify where to compute the normalisation: "
            "NormRange(\"<rangeName>\").\n\tThe default (full) range can be denoted with Range(\"\") / "
            "NormRange(\"\").")  # fmt: skip
    return type(options)(options.given)


def _names(text: Any) -> list[str]:
    return [one for one in str(text).split(",")]


def _limits(frame: Any, names: list[str], say: bool = True) -> list[tuple[float, float]]:
    """Each named range's ends; a name the variable has no range of is said and passed over."""
    var = frame.getPlotVar()
    found = []
    for name in names:
        if name and not var.hasRange(name) and say:
            log(None, ERROR, "Plotting", f"Range '{name}' not defined for variable "
                f"'{var.GetName()}'. Ignoring ...")  # fmt: skip
            continue
        found.append((var.getMin(name or None), var.getMax(name or None)))
    return found


def _relative(pdf: Any, frame: Any, options: Commands, nset: frozenset[str]) -> float:
    """The events a relative curve stands for, with the messages of the ranges chosen."""
    given = options.args("Range")
    limits: list[tuple[float, float]] = []
    adjust = False
    rest = "" if "NormRange" in options else ", curve is normalized to data in {} range"
    if given and not isinstance(given[0], str):
        limits = [(float(given[0]), float(given[1]))]
        adjust = bool(given[2]) if len(given) > 2 else True
        log(pdf, INFO, "Plotting", f"RooAbsPdf::plotOn({pdf.GetName()}) only plotting range "
            f"[{g(given[0])},{g(given[1])}]" + rest.format("given" if adjust else "full"))  # fmt: skip
    elif given:
        limits = _limits(frame, _names(given[0]))
        adjust = bool(given[1]) if len(given) > 1 else True
        log(pdf, INFO, "Plotting", f"RooAbsPdf::plotOn({pdf.GetName()}) only plotting range "
            f"'{given[0]}'" + rest.format("given" if adjust else "full"))  # fmt: skip
    if "NormRange" in options:
        limits, adjust = _limits(frame, _names(options.get("NormRange"))), True
        log(pdf, INFO, "Plotting", f"RooAbsPdf::plotOn({pdf.GetName()}) p.d.f. curve is normalized "
            f"using explicit choice of ranges '{options.get('NormRange')}'")  # fmt: skip
    if limits and adjust:
        return float(sum(frame.getFitRangeNEvt(low, high) for low, high in limits))
    if pdf.canBeExtended() and pdf.expected(nset) > 0:
        return float(pdf.expected(nset))
    return float(frame.getFitRangeNEvt())


def _scale(pdf: Any, frame: Any, options: Commands, nset: frozenset[str]) -> float:
    scale = float(options.get("Normalization", 0, 1.0))
    kind = int(options.get("Normalization", 1, RELATIVE))
    if kind == RAW:
        return scale
    expected = pdf.expected(nset) if kind == RELATIVE_EXPECTED else 1.0
    if frame.getFitRangeNEvt() and kind == RELATIVE:
        scale *= _relative(pdf, frame, options, nset)
    elif kind == RELATIVE_EXPECTED:
        scale *= expected
    elif kind == NUM_EVENT:
        scale /= expected
    return scale * frame.getFitRangeBinW()


def plan(pdf: Any, frame: Any, options: Commands, nset: frozenset[str], is_pdf: bool = True) -> Plan:
    """The curve pieces ``plotOn`` draws, and how they are scaled."""
    made = Plan()
    if is_pdf:
        options = _fit_range(pdf, options)
        made.scale = _scale(pdf, frame, options, nset)
    else:
        made.scale = float(options.get("Normalization", 0, 1.0))
    given = options.args("Range")
    if given and isinstance(given[0], str):
        names = _names(given[0])
        made.pieces = _limits(frame, names, say=False)
        made.norm_range = str(given[0])
        made.post_scale = bool(given[1]) if len(given) > 1 else True
        made.wings = False
    elif given:
        made.pieces = [(float(given[0]), float(given[1]))]
        made.post_scale = bool(given[2]) if len(given) > 2 else True
        made.wings = False
    else:
        made.pieces = [(frame.GetXmin(), frame.GetXmax())]
    if "NormRange" in options:
        made.norm_range, made.post_scale = str(options.get("NormRange")), True
    if "VLines" in options:
        made.wings = True
    return made
