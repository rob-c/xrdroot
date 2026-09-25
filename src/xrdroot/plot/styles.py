"""House styles: ROOT's own look, and the experiments' through mplhep.

``style="ROOT"`` is built in and needs nothing: white, ticks inside on all
four sides with minor ticks between, axis titles at the far ends of their
axes, as a ``TCanvas`` draws them. ``"CMS"``, ``"ATLAS"``, ``"LHCb"``,
``"ALICE"`` and the rest are mplhep's, used when mplhep is installed and
refused with the command to install it when not. They are matplotlib
styles; plotly and bokeh draw ``"ROOT"``, as a layout of their own.

:func:`label` writes an experiment's label above a plot - ``CMS
Preliminary`` on the left, the luminosity and energy on the right - on any
backend, in the same place.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from ..errors import UnsupportedFeatureError
from .colors import PETROFF

__all__ = ["ROOT_STYLE", "applied", "dressed", "label", "use_style"]

#: ROOT's look as matplotlib settings.
ROOT_STYLE: dict[str, Any] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.linewidth": 1.0,
    "axes.xmargin": 0.0,
    "axes.titlesize": "medium",
    "axes.titlelocation": "center",
    "xaxis.labellocation": "right",
    "yaxis.labellocation": "top",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 8.0,
    "ytick.major.size": 8.0,
    "xtick.minor.size": 4.0,
    "ytick.minor.size": 4.0,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "axes.grid": False,
}


def _mplhep_style(name: str) -> Any:
    """One of mplhep's styles, refusing with the way to get it when it is not installed."""
    try:
        import mplhep
    except ImportError:
        raise UnsupportedFeatureError(
            f"style={name!r} is mplhep's, which is not installed: pip install mplhep - or use "
            f"style='ROOT', which is built in"
        ) from None
    found = getattr(mplhep.style, name, None)
    if not isinstance(found, (dict, list)):
        raise ValueError(
            f"style={name!r} is not one of mplhep's styles; 'ROOT' is built in, and mplhep "
            f"has 'CMS', 'ATLAS', 'LHCb', 'ALICE' and more"
        )
    return found


def _matplotlib(name: str) -> Any:
    return ROOT_STYLE if name.upper() == "ROOT" else _mplhep_style(name)


def _checked(name: str, backend: str) -> None:
    if backend == "text":
        raise ValueError(f"style={name!r} has nothing to style in characters: drop it for text")
    if backend != "matplotlib" and name.upper() != "ROOT":
        raise ValueError(
            f"style={name!r} is mplhep's, which styles matplotlib; {backend} draws "
            f"style='ROOT' only"
        )


@contextlib.contextmanager
def applied(name: str | None, backend: str) -> Iterator[None]:
    """The style in force while drawing: matplotlib's settings, which new axes are made with."""
    if name is None:
        yield
        return
    _checked(name, backend)
    if backend != "matplotlib":
        yield
        return
    from matplotlib import style

    with style.context(_matplotlib(name)):
        yield


def _plotly_root(figure: Any) -> None:
    figure.update_layout(template="simple_white", colorway=list(PETROFF))
    for update in (figure.update_xaxes, figure.update_yaxes):
        update(mirror="allticks", ticks="inside", showline=True, linecolor="black")


def _bokeh_root(fig: Any) -> None:
    fig.background_fill_color = "white"
    fig.xgrid.visible = fig.ygrid.visible = False
    for axis in (*fig.xaxis, *fig.yaxis):
        axis.major_tick_in, axis.major_tick_out = 8, 0
        axis.minor_tick_in, axis.minor_tick_out = 4, 0


def dressed(drawn: Any, name: str | None, backend: str) -> Any:
    """``drawn`` restyled after the fact, for the backends styled by their objects."""
    if name is not None and backend == "plotly":
        _plotly_root(drawn)
    if name is not None and backend == "bokeh":
        _bokeh_root(drawn)
    return drawn


def use_style(name: str) -> None:
    """Draw with matplotlib in ``name``'s style from now on: ``"ROOT"``, or mplhep's ``"CMS"``."""
    from matplotlib import style

    style.use(_matplotlib(name))


# -- experiment labels --------------------------------------------------------------------------


def _right(lumi: Any, energy: Any) -> str:
    parts = [f"{lumi} fb$^{{-1}}$" if lumi is not None else "", f"({energy} TeV)" if energy else ""]
    return " ".join(part for part in parts if part)


def _plain(text: str) -> str:
    return text.replace("$^{-1}$", "⁻¹")


def _math(text: str, face: str) -> str:
    """``text`` in matplotlib's mathtext ``face``, its spaces kept."""
    if not text:
        return ""
    return "$\\" + face + "{" + text.replace(" ", "\\ ") + "}$"


def _label_matplotlib(ax: Any, left: tuple[str, str], right: str) -> Any:
    written = f"{_math(left[0], 'mathbf')} {_math(left[1], 'mathit')}".strip()
    ax.text(0.0, 1.01, written, transform=ax.transAxes, va="bottom")
    ax.text(1.0, 1.01, right, transform=ax.transAxes, ha="right", va="bottom")
    return ax


def _label_plotly(figure: Any, left: tuple[str, str], right: str) -> Any:
    corner = {"xref": "paper", "yref": "paper", "y": 1.0, "yanchor": "bottom", "showarrow": False}
    figure.add_annotation(text=f"<b>{left[0]}</b> <i>{left[1]}</i>", x=0, xanchor="left", **corner)
    figure.add_annotation(text=_plain(right), x=1, xanchor="right", **corner)
    return figure


def _label_bokeh(fig: Any, left: tuple[str, str], right: str) -> Any:
    from bokeh.models import Title

    fig.add_layout(Title(text=_plain(right), align="right"), "above")
    fig.add_layout(Title(text=f"{left[0]} {left[1]}", text_font_style="bold"), "above")
    return fig


def _label_text(text: str, left: tuple[str, str], right: str) -> str:
    return f"{left[0]} {left[1]}   {_plain(right)}".strip() + "\n" + text


def _backend_of(drawn: Any) -> str:
    if isinstance(drawn, str):
        return "text"
    if hasattr(drawn, "add_trace"):
        return "plotly"
    if hasattr(drawn, "add_layout"):
        return "bokeh"
    return "matplotlib"


#: How each backend writes a label, by what it was given.
LABELLERS = {
    "matplotlib": _label_matplotlib,
    "plotly": _label_plotly,
    "bokeh": _label_bokeh,
    "text": _label_text,
}


def label(
    drawn: Any,
    experiment: str = "CMS",
    text: str = "Preliminary",
    *,
    lumi: Any = None,
    energy: Any = 13,
) -> Any:
    """An experiment's label on what ``plot()`` returned, which is returned again.

        >>> ax = h.plot()                                                # doctest: +SKIP
        >>> xrdroot.plot.label(ax, "CMS", "Preliminary", lumi=138)       # doctest: +SKIP

    ``lumi`` is in inverse femtobarns and ``energy`` in TeV, on the right;
    the experiment in bold and ``text`` after it, on the left. A ratio
    plot's pair of axes is labelled on the upper one.
    """
    target = drawn[0] if isinstance(drawn, tuple) else drawn
    made = LABELLERS[_backend_of(target)](target, (experiment, text), _right(lumi, energy))
    return made if isinstance(target, str) else drawn
