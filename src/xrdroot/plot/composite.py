"""The plots made of several things: a ratio under its two histograms, a comparison, a stack.

:func:`ratio` is ROOT's ``TRatioPlot``: the two histograms in an upper panel
and, in a lower one sharing its x axis, their ratio - ``divsym``, with the
errors of both, as ``TH1::Divide`` gives them; ``pois``, the interval for a
ratio of two Poisson counts, as ``TGraphAsymmErrors::Divide`` does; or
``diff`` and ``diffsig``, the difference and the difference in units of its
error. The denominator may be a function - a fit - which makes the lower
panel its residuals or pulls.

:func:`compare` overlays things in colours told apart, with a legend;
:func:`stack` piles histograms up as ``THStack`` does, each filled.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ..efficiency import beta_quantile
from ..function import Function
from ..hist import Histogram
from ..stacks import Stack
from . import backends, build, styles
from .api import draw_picture
from .colors import PETROFF
from .model import Curve, Frame, Look, Picture, Points
from .options import Chosen
from .request import Request, styled

__all__ = ["compare", "ratio", "stack"]

#: The level ``pois`` quotes its interval at: one Gaussian sigma.
ONE_SIGMA = 0.682689492137086

#: What the lower panel's axis says, and the line it is drawn about, for each ratio.
RATIOS = {
    "divsym": ("ratio", 1.0),
    "pois": ("ratio", 1.0),
    "diff": ("difference", 0.0),
    "diffsig": ("pull", 0.0),
}


def _flat(obj: Any, what: str) -> Histogram:
    if not isinstance(obj, Histogram) or len(obj.axes) != 1:
        raise ValueError(
            f"the {what} of a ratio plot is a histogram of one axis, and "
            f"{getattr(obj, 'name', obj)!r} is not one"
        )
    return obj


def _denominator(obj: Any, numerator: Histogram) -> tuple[Any, Any]:
    """The denominator's value and error in each of the numerator's bins."""
    if isinstance(obj, Function):
        centres = numerator.axes[0].centers()
        values = np.asarray(obj(centres), dtype=np.float64)
        return values, np.zeros_like(values)
    denominator = _flat(obj, "denominator")
    if not np.array_equal(denominator.edges(), numerator.edges()):
        raise ValueError(
            f"{numerator.name!r} and {denominator.name!r} are binned differently, and a "
            f"ratio is taken bin by bin: rebin one to the other's edges first"
        )
    return denominator.values(), denominator.errors()


def _divsym(n: Any, en: Any, d: Any, ed: Any) -> tuple[Any, Any, Any]:
    """``TH1::Divide``: the ratio, with the errors of both added as ROOT adds them."""
    safe = np.where(d != 0, d, np.nan)
    value = n / safe
    error = np.sqrt((en / safe) ** 2 + (n * ed / safe**2) ** 2)
    return value, error, error


def _diff(n: Any, en: Any, d: Any, ed: Any) -> tuple[Any, Any, Any]:
    error = np.hypot(en, ed)
    return n - d, error, error


def _diffsig(n: Any, en: Any, d: Any, ed: Any) -> tuple[Any, Any, Any]:
    spread = np.hypot(en, ed)
    value = (n - d) / np.where(spread > 0, spread, np.nan)
    zeros = np.zeros_like(value)
    return value, zeros, zeros


def _pois(n: Any, en: Any, d: Any, ed: Any) -> tuple[Any, Any, Any]:
    """``TGraphAsymmErrors::Divide(..., "pois")``: a ratio of counts, from a binomial interval.

    ``n / (n + d)`` is a binomial fraction whose Clopper-Pearson interval,
    mapped through ``p / (1 - p)``, is the interval on the ratio.
    """
    tail = (1 - ONE_SIGMA) / 2
    value = n / np.where(d > 0, d, np.nan)
    low, high = np.full_like(value, np.nan), np.full_like(value, np.nan)
    for at in np.flatnonzero(d > 0).tolist():
        a, b = float(n[at]), float(d[at])
        lower = beta_quantile(tail, a, b + 1) if a > 0 else 0.0
        upper = beta_quantile(1 - tail, a + 1, b)
        low[at] = value[at] - lower / (1 - lower)
        high[at] = upper / (1 - upper) - value[at]
    return value, low, high


#: How each ratio is worked out.
WORKED = {"divsym": _divsym, "pois": _pois, "diff": _diff, "diffsig": _diffsig}


def _worked(method: str, numerator: Histogram, denominator: Any) -> tuple[Any, Any, Any]:
    if method not in WORKED:
        raise ValueError(
            f"option={method!r} is not one of TRatioPlot's drawn here: {', '.join(WORKED)}"
        )
    if method == "pois" and isinstance(denominator, Function):
        raise ValueError(
            "'pois' is a ratio of two counts, and a function is not a count: use 'divsym', "
            "'diff' or 'diffsig' against a fit"
        )
    d, ed = _denominator(denominator, numerator)
    return WORKED[method](numerator.values(), numerator.errors(), d, ed)


def _lower(numerator: Histogram, method: str, worked: tuple[Any, Any, Any], words: Any) -> Picture:
    """The lower panel: the ratio's points about a dashed line at one, or at zero."""
    value, low, high = worked
    axis = numerator.axes[0]
    keep = np.isfinite(value) & np.isfinite(low) & np.isfinite(high)
    half = axis.widths()[keep] / 2
    look = styled(numerator.members, Request(Chosen(frozenset()), {}))
    points = Points(axis.centers()[keep], value[keep], half, half, low[keep], high[keep], look)
    label, level = RATIOS[method]
    edges = numerator.edges()
    line = Curve(
        np.array([edges[0], edges[-1]]), np.array([level, level]),
        Look(color="#7f7f7f", dash="dashed"),
    )  # fmt: skip
    frame = Frame(
        xlabel=words.get("xlabel") or axis.title, ylabel=label, logx=bool(words.get("logx"))
    )
    return Picture((line, points), frame)


def _next(target: Any, drawn: Any) -> Any:
    """What the next layer is drawn on: the same panel, or the text so far."""
    return drawn if target is None or isinstance(target, str) else target


def _dress(name: str, hep: str | None, pieces: Sequence[Any]) -> None:
    for piece in pieces:
        styles.dressed(piece, hep, name)


def ratio(
    numerator: Any,
    denominator: Any,
    option: str = "divsym",
    *,
    backend: str | None = None,
    labels: Sequence[str] | None = None,
    numerator_option: str = "E",
    denominator_option: str = "HIST",
    style: str | None = None,
    **words: Any,
) -> Any:
    """``TRatioPlot``: the two above, and their ratio - or difference, or pull - below.

        >>> upper, lower = xrdroot.plot.ratio(data, mc)                 # doctest: +SKIP
        >>> fig = xrdroot.plot.ratio(h, fit, "diffsig", backend="plotly")  # doctest: +SKIP

    ``option`` is ``divsym`` (the default), ``pois``, ``diff`` or
    ``diffsig``. The keywords style the upper panel as :func:`plot`'s do.
    What comes back is the backend's: matplotlib's two axes, upper first; a
    plotly figure of two rows; a bokeh column of two figures; or text.
    """
    upper_one = _flat(numerator, "numerator")
    worked = _worked(option, upper_one, denominator)
    name, module = backends.backend(backend)
    names = list(labels or (upper_one.title or upper_one.name, getattr(denominator, "title", "")))
    upper_words = {**words, "xlabel": ""}
    with styles.applied(style, name):
        upper, lower, whole = module.panels(Frame(logx=bool(words.get("logx")),
                                                  logy=bool(words.get("logy"))))  # fmt: skip
        pictures = (
            build.picture(numerator, numerator_option, {**upper_words, "label": names[0]}),
            build.picture(denominator, "" if isinstance(denominator, Function)
                          else denominator_option, {**upper_words, "label": names[1] or None}),
        )  # fmt: skip
        for picture in pictures:
            upper = _next(upper, module.render(picture, upper))
        lower = _next(lower, module.render(_lower(upper_one, option, worked, words), lower))
    _dress(name, style, [whole] if name == "plotly" else [upper, lower] if name == "bokeh" else [])
    return backends.remember(name, module.joined(upper, lower, whole))


def compare(
    objects: Iterable[Any],
    labels: Sequence[str] | None = None,
    *,
    option: str = "",
    norm: bool = False,
    colors: Sequence[Any] | None = None,
    ax: Any = None,
    backend: str | None = None,
    style: str | None = None,
    **words: Any,
) -> Any:
    """Several things drawn over one another, each its own colour, with a legend.

        >>> ax = xrdroot.plot.compare([data, mc], ["data", "MC"], norm=True)  # doctest: +SKIP

    Each is labelled by ``labels``, or its title, and coloured by ``colors``
    or, in turn, ROOT's ten Petroff colours. ``norm`` draws each with
    ``NORM``, scaled to a sum of one, which is how shapes are compared.
    """
    items = list(objects)
    names = list(
        labels or (getattr(item, "title", "") or getattr(item, "name", "") for item in items)
    )
    shades = list(colors or (PETROFF[index % len(PETROFF)] for index in range(len(items))))
    drawn = ax
    for index, item in enumerate(items):
        written = f"{option} NORM" if norm else option
        look = {"color": shades[index], "label": names[index], **words}
        drawn = draw_picture(build.picture(item, written, look), drawn, backend, style)
    return drawn


def stack(
    histograms: Iterable[Any],
    labels: Sequence[str] | None = None,
    *,
    option: str = "",
    colors: Sequence[Any] | None = None,
    ax: Any = None,
    backend: str | None = None,
    style: str | None = None,
    **words: Any,
) -> Any:
    """Histograms piled up as ``THStack`` piles them, each filled in its own colour.

        >>> ax = xrdroot.plot.stack([ttbar, wjets, qcd], ["tt", "W+jets", "QCD"])  # doctest: +SKIP

    The first is at the bottom. ``option`` is a stack's - ``NOSTACK`` to
    overlay them, ``NOSTACKB`` for bars side by side - and each is filled
    with ``colors`` or, in turn, ROOT's ten Petroff colours.
    """
    items = list(histograms)
    names = list(labels or (item.title or item.name for item in items))
    shades = list(colors or (PETROFF[index % len(PETROFF)] for index in range(len(items))))
    each = [{"fill": shades[index], "label": names[index]} for index in range(len(items))]
    held = Stack(
        "THStack", {"TNamed": {"fName": "stack", "fTitle": str(words.pop("title", ""))},
                    "fHists": items}
    )  # fmt: skip
    return draw_picture(build.picture(held, option, words, each), ax, backend, style)
