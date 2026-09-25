"""What one call to draw asks for, carried to whatever builds the layers.

The option has been read, the caller's keywords sorted; what is left is to
make the layers, and every builder needs the same few things to do it. For
several things drawn together - a stack, a multigraph, a comparison - it
also says which of them this is, which is what ROOT's ``PLC``, ``PMC`` and
``PFC`` colour each one by.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

from .attributes import look, restyled
from .colors import palette_color
from .model import Look
from .options import Chosen

__all__ = ["Request", "styled"]


class Request(NamedTuple):
    """The option read, the caller's look, and where this thing is among others."""

    chosen: Chosen
    marks: Mapping[str, Any]
    palette: Any = "bird"
    levels: int = 20
    #: Which of how many things drawn together this is.
    place: int = 0
    places: int = 1
    #: The look of each of several things, in order - its label, its colour -
    #: which wins over :attr:`marks` for that one thing.
    each: tuple[Mapping[str, Any], ...] = ()

    def again(self, chosen: Chosen, place: int = 0, places: int = 1) -> Request:
        """The same request for one of several things, drawn with ``chosen``."""
        return self._replace(chosen=chosen, place=place, places=places)


def _from_palette(base: Look, request: Request) -> Look:
    """``PLC``, ``PMC``, ``PFC``: the line, marker or fill coloured from the palette."""
    chosen = request.chosen
    if not chosen.has("PLC", "PMC", "PFC"):
        return base
    picked = palette_color(request.place, request.places, request.palette)
    changes: dict[str, Any] = {
        field: picked
        for word, field in (("PLC", "color"), ("PMC", "marker_color"), ("PFC", "fill"))
        if chosen.has(word)
    }
    return base._replace(**changes)


def styled(members: Any, request: Request, markers: bool = True) -> Look:
    """The look an object is drawn in: its own attributes, the palette's, then the caller's."""
    base = _from_palette(look(members, markers), request)
    if request.chosen.has("*"):
        base = base._replace(marker="asterisk", hollow=False)
    own = request.each[request.place] if request.place < len(request.each) else {}
    return restyled(base, {**request.marks, **own})
