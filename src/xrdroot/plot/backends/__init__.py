"""The drawing libraries a picture can be handed to, and which one is in use.

Each backend is a module with the same three functions: ``render`` draws a
:class:`~xrdroot.plot.model.Picture` onto what it is given - or onto
something new - and returns the library's own object; ``panels`` makes the
two stacked panels of a ratio plot; and ``joined`` puts those two back
together as whatever the library calls a figure of several plots. They are
imported when first used, so choosing plotly costs nothing until it draws.

ROOT draws ``SAME`` onto the current pad. Here each backend remembers the
last thing it drew on, and ``SAME`` draws there again.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from ...errors import UnsupportedFeatureError

__all__ = ["BACKENDS", "backend", "current", "get_backend", "remember", "set_backend"]

#: The backends, against the module that draws with each and the package it needs.
BACKENDS = {
    "matplotlib": ("withmatplotlib", "matplotlib"),
    "plotly": ("withplotly", "plotly"),
    "bokeh": ("withbokeh", "bokeh"),
    "text": ("astext", None),
}

_state: dict[str, Any] = {"backend": "matplotlib", "current": {}}


def _checked(name: str) -> str:
    chosen = str(name).lower()
    if chosen not in BACKENDS:
        raise ValueError(
            f"backend={name!r} is not one this draws with: it draws with {', '.join(BACKENDS)}"
        )
    return chosen


def set_backend(name: str) -> None:
    """Draw with ``name`` from now on unless a call says otherwise: ``"plotly"``, say."""
    _state["backend"] = _checked(name)


def get_backend() -> str:
    """The backend drawn with when a call does not say."""
    return str(_state["backend"])


def backend(name: str | None = None) -> tuple[str, ModuleType]:
    """The backend ``name`` - or the one set - and its module, refusing one not installed."""
    chosen = _checked(name) if name is not None else get_backend()
    module, needs = BACKENDS[chosen]
    if needs is not None:
        try:
            importlib.import_module(needs)
        except ImportError:
            raise UnsupportedFeatureError(
                f"drawing with {chosen} needs {needs}, which is not installed: pip install "
                f"{needs} - or use backend='text', which draws with nothing installed at all"
            ) from None
    return chosen, importlib.import_module(f"{__name__}.{module}")


def current(name: str) -> Any:
    """What ``name`` last drew on, for ``SAME``; ``None`` before it has drawn."""
    return _state["current"].get(name)


def remember(name: str, target: Any) -> Any:
    """Keep ``target`` as what ``name`` last drew on, and hand it back."""
    _state["current"][name] = target
    return target
