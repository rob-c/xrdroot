"""Plotting, through the libraries Python already draws with, in one way for all of them.

    >>> import xrdroot.plot
    >>> ax = f["h_pt"].plot(option="E1", color="kRed+1", logy=True)        # doctest: +SKIP
    >>> fig = f["h2"].plot(backend="plotly", option="SURF")                  # doctest: +SKIP
    >>> upper, lower = xrdroot.plot.ratio(data, mc, style="CMS")            # doctest: +SKIP
    >>> xrdroot.plot.set_backend("bokeh")                                    # doctest: +SKIP

ROOT draws on a ``TCanvas``. This draws on matplotlib axes, a plotly figure,
a bokeh figure or plain text - ``backend=`` or :func:`set_backend` chooses -
and hands back that library's own object, so what ROOT's canvas could not
do, the library can: save it as a PDF, put it in a notebook, lay it out
with others. What does not change with the backend is what is drawn: ROOT's
draw options (``HIST``, ``E1``, ``E2``, ``COLZ``, ``LEGO``, ``SAME``,
``AP``, ``NOSTACK`` ...), its defaults, its colours and markers, read off
the object as ROOT would read them.

- :func:`plot` draws anything; every histogram, graph, profile, efficiency,
  stack, multigraph and function has it as ``.plot()``.
- :func:`ratio`, :func:`compare` and :func:`stack` are the plots made of
  several things.
- :func:`use_style` and ``style=`` give ROOT's look, or an experiment's
  through mplhep; :func:`label` writes an experiment's label.
- :func:`color` and :func:`palette` are ROOT's colour table and palettes.
"""

from __future__ import annotations

from .api import draw_picture, plot
from .backends import BACKENDS, get_backend, set_backend
from .build import picture
from .colors import PETROFF, color, palette
from .composite import compare, ratio, stack
from .styles import ROOT_STYLE, label, use_style

__all__ = [
    "plot",
    "set_backend",
    "get_backend",
    "BACKENDS",
    "ratio",
    "compare",
    "stack",
    "use_style",
    "label",
    "ROOT_STYLE",
    "color",
    "palette",
    "PETROFF",
    "picture",
    "draw_picture",
]
