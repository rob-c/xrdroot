"""From an object, an option and the caller's keywords to a :class:`~.model.Picture`.

This is the half of drawing that needs no drawing library: read the option
for the kind of thing being drawn, sort the keywords into the look, the
frame and whatever is the library's own, and ask the right builder for the
layers. What comes back is the same picture whichever backend draws it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..efficiency import Efficiency
from ..function import Function
from ..graph import Graph
from ..hist import Histogram
from ..stacks import MultiGraph, Stack
from . import containers, drawers
from .attributes import frame as reframed
from .attributes import split_style
from .model import Frame, Picture
from .options import choose
from .request import Request

__all__ = ["picture", "relabelled"]

Builder = Callable[[Any, Request], "tuple[list[Any], Frame]"]

#: What each class is drawn by, and the kind of thing its option is read as;
#: ``None`` for a kind that depends on the object's dimensions.
BUILDERS: tuple[tuple[type, Builder, str | None], ...] = (
    (Stack, containers.stack, "stack"),
    (MultiGraph, containers.multigraph, "graph"),
    (Histogram, drawers.histogram, None),
    (Efficiency, drawers.efficiency, None),
    (Graph, drawers.graph, "graph"),
    (Function, drawers.function, None),
)

#: The keywords that are neither the look nor the frame but still this library's.
OWN = ("palette", "levels", "labels")


def _builder(obj: Any) -> tuple[Builder, str]:
    for cls, builder, kind in BUILDERS:
        if isinstance(obj, cls):
            return builder, kind or drawers.kind_of(obj)
    raise TypeError(
        f"a {type(obj).__name__} is not something plot() draws: it draws histograms, "
        f"profiles, efficiencies, graphs, multigraphs, stacks and functions"
    )


def relabelled(layers: Sequence[Any]) -> tuple[Any, ...]:
    """The layers with each legend label kept on the first layer to carry it only.

    A graph drawn ``LP`` is a line and markers under one name, and the
    legend should say that name once.
    """
    seen: set[str] = set()
    kept = []
    for layer in layers:
        label = getattr(getattr(layer, "look", None), "label", None)
        if label is not None and label in seen:
            layer = layer._replace(look=layer.look._replace(label=None))
        elif label is not None:
            seen.add(label)
        kept.append(layer)
    return tuple(kept)


def request(
    option: str, kind: str, style: Mapping[str, Any], each: Sequence[Mapping[str, Any]] = ()
) -> tuple[Request, dict[str, Any]]:
    """The option read for ``kind``, and the caller's keywords sorted: what to draw, and how."""
    words = dict(style)
    own = {name: words.pop(name) for name in OWN if name in words}
    marks, framing, native = split_style(words)
    made = Request(
        choose(option, kind),
        marks,
        palette=own.get("palette", "bird"),
        levels=int(own.get("levels", 20)),
        each=tuple(each) or tuple({"label": str(label)} for label in own.get("labels") or ()),
    )
    return made, {"frame": framing, "native": native}


def picture(
    obj: Any,
    option: str = "",
    style: Mapping[str, Any] | None = None,
    each: Sequence[Mapping[str, Any]] = (),
) -> Picture:
    """What ``obj`` drawn with ``option`` looks like, before anything draws it.

    ``each`` is the look of each of the things a stack or multigraph holds,
    in order; ``labels=[...]`` among the keywords is the same for labels.
    """
    builder, kind = _builder(obj)
    made, rest = request(option, kind, style or {}, each)
    layers, framed = builder(obj, made)
    return Picture(
        relabelled(layers),
        reframed(framed, rest["frame"]),
        same=made.chosen.has("SAME"),
        native=rest["native"],
    )
