"""What drawing a volume paints: ``TGeoPainter::PaintVolume``'s walk of the tree.

From the volume drawn, each placed volume below it is visited depth first
through the matrices above it. With the default visualisation option
(``kGeoVisLeaves``, 1) a visible volume is painted where the walk stops -
at a leaf, at the visible depth, or where daughters are hidden; with
``kGeoVisDefault`` (0) every visible volume down to the depth is; with
``kGeoVisOnly`` (2) the volume drawn alone. The volume drawn is painted
itself only when it is visible and the geometry says so
(``SetTopVisible``), or when it has nothing below it to paint. An iterator
plugin (``TGeoIteratorPlugin``) sees each volume before it is painted and
may change its look for that painting only, as ROOT's does.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ...canvas import Primitive
from ...geom import IDENTITY, Matrix, Solid

__all__ = ["solids", "primitive", "draw_object", "draw_volume", "walk"]

#: The visualisation options, as ``TVirtualGeoPainter``'s enumeration numbers them.
VIS_DEFAULT, VIS_LEAVES, VIS_ONLY = 0, 1, 2


def _manager() -> Any:
    from .manager import current_manager

    return current_manager()


#: A node met on a walk: the node, its placement in the world, its depth, its path.
Visit = tuple[Any, Matrix, int, tuple[Any, ...]]


def walk(volume: Any, matrix: Matrix = IDENTITY, level: int = 1,
         path: tuple[Any, ...] = ()) -> Iterator[Visit]:  # fmt: skip
    """Every node below ``volume``, depth first: the node, its placement in the world, its
    depth (1 for a daughter) and the nodes from the top down to it."""
    for node in volume.GetNodes():
        placed = matrix @ node.GetMatrix()._xrd
        below = (*path, node)
        yield node, placed, level, below
        yield from walk(node.GetVolume(), placed, level + 1, below)


def _solid(volume: Any, matrix: Matrix) -> Solid:
    mesh = volume.GetShape().mesh().transformed(matrix)
    return Solid(mesh, int(volume.GetLineColor()), int(volume.GetLineWidth()),
                 int(volume.GetLineStyle()), int(volume.GetFillColor()),
                 int(volume.GetTransparency()), volume.GetName())  # fmt: skip


def _painted(node: Any, level: int, depth: int, option: int) -> tuple[bool, bool]:
    """Whether a node is painted, and whether the walk stops below it."""
    volume = node.GetVolume()
    visible = volume.IsVisible() and not volume.IsAssembly()
    hidden_below = not volume.IsVisDaughters()
    if option == VIS_DEFAULT:
        return visible and level <= depth, level >= depth or hidden_below
    last = not volume.GetNdaughters() or level >= depth or hidden_below
    return visible and last, last


def _plugged(plugin: Any, volume: Any, path: tuple[Any, ...], level: int) -> tuple[int, ...]:
    """The look ``plugin`` is shown, and the look to put back once painted."""
    kept = (volume.GetLineColor(), volume.GetLineWidth(), volume.GetLineStyle())
    plugin._visit(path, level)
    plugin.ProcessNode()
    return kept


def _below(top: Any, depth: int, option: int) -> Iterator[Visit]:
    """The nodes below ``top`` that are painted, the walk pruned where ROOT's stops."""
    skipping: tuple[Any, ...] | None = None
    for node, matrix, level, path in walk(top):
        if skipping is not None and path[: len(skipping)] == skipping:
            continue
        paint, stop = _painted(node, level, depth, option)
        skipping = path if stop else None
        if paint:
            yield node, matrix, level, path


def _top(top: Any, manager: Any, option: int) -> tuple[list[Solid], bool]:
    """The volume drawn, if it is painted itself, and whether it is painted alone."""
    shown = top.IsVisible() and not top.IsAssembly()
    alone = not top.GetNdaughters() or not top.IsVisDaughters() or option == VIS_ONLY
    painted = top.GetShape() is not None and ((manager.GetTopVisible() and shown) or alone)
    return ([_solid(top, IDENTITY)] if painted else []), alone


def solids(top: Any) -> list[Solid]:
    """What painting ``top`` paints, each shape where the tree puts it."""
    manager = _manager()
    option, depth = manager.GetVisOption(), manager.GetVisLevel()
    found, alone = _top(top, manager, option)
    if alone:
        return found
    plugin = manager.GetGeomPainter().plugin
    return found + [_through(plugin, node.GetVolume(), matrix, path, level)
                    for node, matrix, level, path in _below(top, depth, option)]  # fmt: skip


def _through(plugin: Any, volume: Any, matrix: Matrix, path: tuple[Any, ...], level: int) -> Solid:
    """One volume painted, its plugin - if there is one - allowed to change how, for now."""
    if plugin is None:
        return _solid(volume, matrix)
    color, width, style = _plugged(plugin, volume, path, level)
    made = _solid(volume, matrix)
    volume.SetLineColor(color)
    volume.SetLineWidth(width)
    volume.SetLineStyle(style)
    return made


def primitive(volume: Any) -> Primitive:
    """A drawn volume as the canvas paints it: the solids of the tree below it."""
    return Primitive("TGeoVolume", {"fName": volume.GetName(), "solids": solids(volume)})


def draw_object(obj: Any, option: str) -> None:
    """``TGeoPainter::DrawVolume``, ``TNode::Draw``: the pad cleared unless ``same``, a
    perspective view made for it if it has none, and ``obj`` put in it."""
    from ..core import hooks
    from ..graphics.canvas import default_canvas
    from ..graphics.pads import current
    from ..graphics.views3d import PERSPECTIVE, TView3D

    pad = current() or default_canvas()
    if "same" not in option.lower():
        pad.Clear()
    if pad.GetView() is None:
        TView3D(PERSPECTIVE)
    hooks.draw_hook(obj, option)


def draw_volume(volume: Any, option: str) -> None:
    """A volume drawn: the painter told which, then drawn as any 3-D object is."""
    _manager().GetGeomPainter().top = volume
    draw_object(volume, option)
