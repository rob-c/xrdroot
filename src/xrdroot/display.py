"""``.plot()`` and what a notebook shows, for everything that is drawn.

A histogram, graph, profile, efficiency, stack, multigraph and function all
draw the same way, through :mod:`xrdroot.plot`, which is imported the first
time something is drawn or shown rather than when the library is.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Displayed"]


class Displayed:
    """Drawing and notebook display, shared by everything :func:`xrdroot.plot.plot` draws."""

    __slots__ = ()

    def plot(
        self, ax: Any = None, backend: str | None = None, option: str = "", **style: Any
    ) -> Any:
        """Draw this with ROOT's ``option``, and hand back what it was drawn on.

            >>> ax = h.plot()                                    # doctest: +SKIP
            >>> h.plot(ax=ax, option="E1 SAME", color="kRed+1")  # doctest: +SKIP
            >>> fig = h2.plot(backend="plotly", option="LEGO2Z")  # doctest: +SKIP

        ROOT's defaults, colours and markers apply unless the keywords say
        otherwise; the backend - matplotlib unless
        :func:`~xrdroot.plot.set_backend` or ``backend=`` says - gives back
        its own object to keep styling. See :func:`xrdroot.plot.plot`.
        """
        from .plot import plot

        return plot(self, ax, backend, option, **style)

    def _repr_html_(self) -> str:
        from .plot.notebook import figure

        return figure(self)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return {"text/plain": repr(self), "text/html": self._repr_html_()}
