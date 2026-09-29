"""Each of ROOT's solids as a :class:`~.mesh.Mesh`, from the parameters ``TGeo`` gives it.

Lengths are half-lengths, as ROOT's are (a box ``dx, dy, dz`` is ``2 dx``
wide), and angles are in degrees. The eight-cornered solids put their
corners where ``TGeoArb8`` and its kin do - four round the ``-dz`` face,
four round the ``+dz`` one, in ROOT's order - and the round ones are
turned through ``steps`` steps of their range of phi, as
``TGeoManager::SetNsegments`` sets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .mesh import Mesh, circle_profile, extrusion, hexahedron, revolve

__all__ = [
    "box", "arb8", "trd1", "trd2", "para", "trap", "gtra", "old_gtra", "tube", "cone", "pcon",
    "pgon", "sphere", "torus", "eltu", "ctub", "paraboloid", "hype", "xtru", "tessellated",
]  # fmt: skip

#: How many steps a curved profile (a sphere's, a paraboloid's) is drawn in.
PROFILE_STEPS = 10


def box(dx: float, dy: float, dz: float, origin: Sequence[float] = (0.0, 0.0, 0.0)) -> Mesh:
    """``TGeoBBox``: the box of half-sides ``dx, dy, dz`` round ``origin``."""
    return trd2(dx, dx, dy, dy, dz, origin)


def arb8(dz: float, vertices: Sequence[Sequence[float]]) -> Mesh:
    """``TGeoArb8``: eight ``(x, y)`` corners, the first four at ``-dz``, the last at ``+dz``."""
    xy = np.asarray(vertices, dtype=np.float64).reshape(8, 2)
    z = np.repeat([-dz, dz], 4)
    return hexahedron(np.column_stack([xy, z]))


def trd1(dx1: float, dx2: float, dy: float, dz: float) -> Mesh:
    """``TGeoTrd1``: half-width ``dx1`` at ``-dz`` and ``dx2`` at ``+dz``."""
    return trd2(dx1, dx2, dy, dy, dz)


def trd2(
    dx1: float, dx2: float, dy1: float, dy2: float, dz: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> Mesh:  # fmt: skip
    """``TGeoTrd2``: half-widths in x and y changing from ``-dz`` to ``+dz``."""
    corners = []
    for dx, dy in ((dx1, dy1), (dx2, dy2)):
        corners += [(-dx, -dy), (-dx, dy), (dx, dy), (dx, -dy)]
    mesh = arb8(dz, corners)
    mesh.points += np.asarray(origin, dtype=np.float64)
    return mesh


def para(dx: float, dy: float, dz: float, alpha: float, theta: float, phi: float) -> Mesh:
    """``TGeoPara``: a box sheared by ``alpha`` in y and leaning by ``theta`` towards ``phi``."""
    t = math.tan(math.radians(theta))
    txy = math.tan(math.radians(alpha))
    txz, tyz = t * math.cos(math.radians(phi)), t * math.sin(math.radians(phi))
    points = box(dx, dy, dz).points
    x, y, z = points.T
    return hexahedron(np.column_stack([x + txy * y + txz * z, y + tyz * z, z]))


def _trap_corners(dz: float, theta: float, phi: float, faces: Sequence[Sequence[float]]) -> Any:
    """``TGeoTrap``'s corners: each face ``(h, bl, tl, alpha)``, the lower at ``-dz``."""
    t = math.tan(math.radians(theta))
    tx, ty = t * math.cos(math.radians(phi)), t * math.sin(math.radians(phi))
    corners = []
    for z, (h, bl, tl, alpha) in zip((-dz, dz), faces):
        ta = math.tan(math.radians(alpha))
        x0, y0 = z * tx, z * ty
        corners += [(x0 - h * ta - bl, y0 - h), (x0 + h * ta - tl, y0 + h),
                    (x0 + h * ta + tl, y0 + h), (x0 - h * ta + bl, y0 - h)]  # fmt: skip
    return np.array(corners)


def trap(dz: float, theta: float, phi: float, *faces: float) -> Mesh:
    """``TGeoTrap(dz, theta, phi, h1, bl1, tl1, alpha1, h2, bl2, tl2, alpha2)``."""
    return arb8(dz, _trap_corners(dz, theta, phi, (faces[:4], faces[4:8])))


def gtra(dz: float, theta: float, phi: float, twist: float, *faces: float) -> Mesh:
    """``TGeoGtra``: a trapezoid whose faces are turned by half of ``twist`` each way."""
    corners = _trap_corners(dz, theta, phi, (faces[:4], faces[4:8]))
    th, ph = math.radians(theta), math.radians(phi)
    centre = np.array([-dz * math.sin(th) * math.cos(ph), -dz * math.sin(th) * math.sin(ph)])
    for rows, half, at in ((slice(0, 4), -0.5, centre), (slice(4, 8), 0.5, -centre)):
        a = math.radians(half * twist)
        c, s = math.cos(a), math.sin(a)
        x, y = (corners[rows] - at).T
        corners[rows] = np.column_stack([x * c + y * s, -x * s + y * c]) + at
    return arb8(dz, corners)


def old_gtra(dz: float, theta: float, phi: float, twist: float, *faces: float) -> Mesh:
    """The old package's ``TGTRA``: unlike ``TGeoGtra``, its lower face starts at the origin,
    its upper face is shifted by ``2 dz sin(theta)`` and turned by the whole ``twist``."""
    h1, bl1, tl1, alpha1, h2, bl2, tl2, alpha2 = faces[:8]
    th, ph, turn = math.radians(theta), math.radians(phi), math.radians(twist)
    dx, dy = 2 * dz * math.sin(th) * math.cos(ph), 2 * dz * math.sin(th) * math.sin(ph)
    dx1, dx2 = 2 * h1 * math.tan(math.radians(alpha1)), 2 * h2 * math.tan(math.radians(alpha2))
    lower = [(-bl1, -h1), (-tl1 + dx1, h1), (tl1 + dx1, h1), (bl1, -h1)]
    upper = np.array([(-bl2 + dx, -h2 + dy), (-tl2 + dx + dx2, h2 + dy),
                      (tl2 + dx + dx2, h2 + dy), (bl2 + dx, -h2 + dy)])  # fmt: skip
    x, y = upper.T
    c, s = math.cos(turn), math.sin(turn)
    turned = np.column_stack([x * c + y * s, -x * s + y * c])
    return arb8(dz, np.concatenate([np.array(lower), turned]))


def tube(rmin: float, rmax: float, dz: float, phi1: float = 0.0, phi2: float = 360.0,
         steps: int = 20) -> Mesh:  # fmt: skip
    """``TGeoTube`` and, with ``phi1, phi2``, ``TGeoTubeSeg``."""
    return cone(dz, rmin, rmax, rmin, rmax, phi1, phi2, steps)


def _span(phi1: float, phi2: float) -> float:
    """``phi1`` to ``phi2`` as ROOT reads it: the way round, never less than nothing."""
    span = phi2 - phi1
    return span + 360.0 if span <= 0 else span


def cone(dz: float, rmin1: float, rmax1: float, rmin2: float, rmax2: float,
         phi1: float = 0.0, phi2: float = 360.0, steps: int = 20) -> Mesh:  # fmt: skip
    """``TGeoCone`` and, with ``phi1, phi2``, ``TGeoConeSeg``."""
    return revolve([(rmax1, -dz), (rmax2, dz)], [(rmin1, -dz), (rmin2, dz)],
                   phi1, _span(phi1, phi2), steps)  # fmt: skip


def pcon(phi1: float, dphi: float, sections: Sequence[Sequence[float]], steps: int = 20) -> Mesh:
    """``TGeoPcon``: each section ``(z, rmin, rmax)``, from ``phi1`` through ``dphi``."""
    return revolve([(rmax, z) for z, _, rmax in sections], [(rmin, z) for z, rmin, _ in sections],
                   phi1, dphi, steps)  # fmt: skip


def pgon(phi1: float, dphi: float, edges: int, sections: Sequence[Sequence[float]]) -> Mesh:
    """``TGeoPgon``: a polycone of ``edges`` flat sides, each radius measured to a side."""
    stretch = 1.0 / math.cos(math.radians(0.5 * dphi / max(edges, 1)))
    scaled = [(z, rmin * stretch, rmax * stretch) for z, rmin, rmax in sections]
    return pcon(phi1, dphi, scaled, max(edges, 1))


def _profile(radius: float, theta1: float, theta2: float, steps: int) -> list[tuple[float, float]]:
    """A meridian of a sphere of ``radius``, polar angle ``theta1`` to ``theta2``: ``(rho, z)``."""
    angles = np.radians(np.linspace(theta1, theta2, steps + 1))
    return [(radius * math.sin(t), radius * math.cos(t)) for t in angles]


def sphere(rmin: float, rmax: float, theta1: float = 0.0, theta2: float = 180.0,
           phi1: float = 0.0, phi2: float = 360.0, steps: int = 20) -> Mesh:  # fmt: skip
    """``TGeoSphere``: a shell, cut to polar angles ``theta1..theta2`` and ``phi1..phi2``."""
    rings = max(steps // 2, 1)
    return revolve(_profile(rmax, theta1, theta2, rings), _profile(rmin, theta1, theta2, rings),
                   phi1, _span(phi1, phi2), steps)  # fmt: skip


def torus(r: float, rmin: float, rmax: float, phi1: float = 0.0, dphi: float = 360.0,
          steps: int = 20) -> Mesh:  # fmt: skip
    """``TGeoTorus``: a tube of radii ``rmin..rmax`` bent round a circle of radius ``r``."""
    outer = circle_profile(r, rmax, steps)
    inner = circle_profile(r, rmin, steps)
    return revolve(outer, inner, phi1, dphi, steps, closed=True)


def eltu(a: float, b: float, dz: float, steps: int = 20) -> Mesh:
    """``TGeoEltu``: a cylinder of elliptic section, semi-axes ``a`` and ``b``."""
    mesh = tube(0.0, 1.0, dz, steps=steps)
    mesh.points[:, 0] *= a
    mesh.points[:, 1] *= b
    return mesh


def ctub(rmin: float, rmax: float, dz: float, phi1: float, phi2: float,
         low: Sequence[float], high: Sequence[float], steps: int = 20) -> Mesh:  # fmt: skip
    """``TGeoCtub``: a tube segment whose ends are cut by planes of normals ``low`` and ``high``.

    Each end's points are moved along z onto its plane, which passes through
    ``(0, 0, -dz)`` or ``(0, 0, +dz)``.
    """
    mesh = tube(rmin, rmax, dz, phi1, phi2, steps)
    x, y, z = mesh.points.T
    for side, normal in ((-1.0, low), (1.0, high)):
        at = z * side > 0
        z[at] = side * dz - (normal[0] * x[at] + normal[1] * y[at]) / normal[2]
    return mesh


def _rings(radius: Any, dz: float) -> list[tuple[float, float]]:
    zs = np.linspace(-dz, dz, PROFILE_STEPS + 1)
    return [(float(radius(z)), float(z)) for z in zs]


def paraboloid(rlo: float, rhi: float, dz: float, steps: int = 20) -> Mesh:
    """``TGeoParaboloid``: ``z = a rho^2 + b``, of radius ``rlo`` at ``-dz``, ``rhi`` at ``+dz``."""
    a = 2 * dz / (rhi * rhi - rlo * rlo)
    b = dz * (rlo * rlo + rhi * rhi) / (rlo * rlo - rhi * rhi)
    outer = _rings(lambda z: math.sqrt(max((z - b) / a, 0.0)), dz)
    return revolve(outer, [(0.0, z) for _, z in outer], 0.0, 360.0, steps)


def hype(rin: float, stin: float, rout: float, stout: float, dz: float, steps: int = 20) -> Mesh:
    """``TGeoHype``: hyperbolic walls, ``rho^2 = r^2 + (z tan(stereo))^2`` inside and out."""
    def wall(r: float, stereo: float) -> list[tuple[float, float]]:
        t2 = math.tan(math.radians(stereo)) ** 2
        return _rings(lambda z: math.sqrt(r * r + z * z * t2), dz)

    return revolve(wall(rout, stout), wall(rin, stin), 0.0, 360.0, steps)


def xtru(polygon: Sequence[Sequence[float]], sections: Sequence[Sequence[float]]) -> Mesh:
    """``TGeoXtru``: the ``(x, y)`` polygon at each section ``(z, x0, y0, scale)``."""
    return extrusion([(float(x), float(y)) for x, y in polygon], sections)


def tessellated(vertices: Sequence[Sequence[float]], facets: Sequence[Sequence[int]]) -> Mesh:
    """``TGeoTessellated``: its vertices, and each facet a loop of three or four of them."""
    return Mesh(vertices, facets)
