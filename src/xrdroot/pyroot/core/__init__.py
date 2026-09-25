"""ROOT's core classes - a placeholder holding only the hook ``Draw`` calls.

``TObject.Draw(option)`` hands the object and its option to
:func:`draw_hook`, which does whatever the hook installed last does: by
default it remembers them in :data:`DRAWN` and nothing more, and
:mod:`xrdroot.pyroot.graphics` installs one that puts them on the current
pad.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__: list[str] = []

#: What was drawn while no drawing hook was installed, in order.
DRAWN: list[tuple[Any, str]] = []

Hook = Callable[[Any, str], Any]


def _remember(obj: Any, option: str) -> None:
    """The hook before any other is installed: remember, and draw nothing."""
    DRAWN.append((obj, option))


_hook: list[Hook] = [_remember]


def draw_hook(obj: Any, option: str = "") -> Any:
    """Draw ``obj`` with ``option`` by whatever hook is installed."""
    return _hook[0](obj, option)


def set_draw_hook(fn: Hook) -> None:
    """Install ``fn`` as what ``Draw`` does from now on."""
    _hook[0] = fn
