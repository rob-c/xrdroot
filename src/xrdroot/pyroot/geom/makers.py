"""``TGeoManager::MakeBox`` and its kin: a volume of a new shape of the same name, in one call.

Each ``Make<Shape>(name, medium, parameters...)`` makes the shape with
those parameters, named as the volume is, and the volume of it in
``medium`` - which is all ROOT's do.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import polyshapes, shapes
from .volumes import TGeoVolume

__all__ = ["Makers", "MAKERS"]

#: Each maker's name, and the shape it makes.
MAKERS: dict[str, type] = {
    "MakeBox": shapes.TGeoBBox, "MakeArb8": shapes.TGeoArb8, "MakeTrd1": shapes.TGeoTrd1,
    "MakeTrd2": shapes.TGeoTrd2, "MakePara": shapes.TGeoPara, "MakeTrap": shapes.TGeoTrap,
    "MakeGtra": shapes.TGeoGtra, "MakeTube": shapes.TGeoTube, "MakeTubs": shapes.TGeoTubeSeg,
    "MakeCtub": shapes.TGeoCtub, "MakeCone": shapes.TGeoCone, "MakeCons": shapes.TGeoConeSeg,
    "MakeSphere": shapes.TGeoSphere, "MakeTorus": shapes.TGeoTorus, "MakeEltu": shapes.TGeoEltu,
    "MakeParaboloid": shapes.TGeoParaboloid, "MakeHype": shapes.TGeoHype,
    "MakePcon": polyshapes.TGeoPcon, "MakePgon": polyshapes.TGeoPgon,
    "MakeXtru": polyshapes.TGeoXtru,
}  # fmt: skip


def _maker(shape: type) -> Callable[..., Any]:
    def make(self: Any, name: Any, medium: Any, *parameters: Any) -> Any:
        return TGeoVolume(name, shape(str(name), *parameters), medium)

    make.__doc__ = f"A volume of a new ``{shape.__name__}`` of the same name, in ``medium``."
    return make


class Makers:
    """``TGeoManager``'s ``Make`` methods, one per shape."""


for _name, _shape in MAKERS.items():
    setattr(Makers, _name, _maker(_shape))
