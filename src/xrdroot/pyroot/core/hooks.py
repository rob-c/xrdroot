"""What ``Draw`` does, which is whatever the graphics part of the kit says it does.

``TObject::Draw`` puts an object on the current pad. Pads are another part of
the kit's business, and may not be installed; until one installs a hook with
:func:`set_draw_hook`, drawing remembers what was drawn with which option in
:data:`DRAWN` and does nothing else - which is what ROOT in batch mode with
no canvas saved amounts to.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["DRAWN", "draw_hook", "set_draw_hook"]

#: Every ``(object, option)`` drawn while no hook is installed, in order.
DRAWN: list[tuple[Any, str]] = []

Hook = Callable[[Any, str], Any]


def _remember(obj: Any, option: str) -> None:
    """The hook there is until another is installed: note what was drawn."""
    DRAWN.append((obj, option))


_installed: list[Hook] = [_remember]


def draw_hook(obj: Any, option: str = "") -> Any:
    """Draw ``obj`` with ``option`` through the hook installed now."""
    return _installed[0](obj, option)


def set_draw_hook(fn: Hook | None) -> None:
    """Install ``fn`` as what ``Draw`` does; ``None`` puts back remembering."""
    _installed[0] = _remember if fn is None else fn
