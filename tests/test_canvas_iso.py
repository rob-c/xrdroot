"""``PaintH3Iso``'s isosurface: where the contents cross their mean, lit as ROOT lights it.

Each cell of eight bin centres is cut into six tetrahedra; the surface
crosses a tetrahedron in a triangle when one corner is on the other side
from the rest, and in a quad of two when two are. Each corner of each
triangle is lit by ``Luminosity`` - ambient 1 at 0.15, and a light of 10
from the screen's (1, 1, 1) - and each triangle is filled in bands of 28
shades of the fill colour between the levels of light (``FillPolygon``).
"""

from __future__ import annotations

import colorsys

import numpy as np
import pytest

from xrdroot.canvas import iso
from xrdroot.canvas.view3d import View3D

VIEW = View3D((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), -120.0, 60.0)


def _corner(value, place):
    return value, np.array(place, float), np.array([0.0, 0.0, 1.0])


def _tetra(*values):
    places = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    return [_corner(value, place) for value, place in zip(values, places, strict=False)]


@pytest.mark.parametrize(
    ("values", "triangles"),
    [((0, 0, 0, 0), 0), ((2, 2, 2, 2), 0), ((2, 0, 0, 0), 1), ((0, 2, 2, 2), 1), ((2, 2, 0, 0), 2)],
)
def test_a_tetrahedron_is_crossed_by_nothing_a_triangle_or_a_quad(values, triangles):
    assert len(iso._tetrahedron(_tetra(*values), 1.0)) == triangles


def test_the_surface_crosses_an_edge_where_the_contents_reach_the_level_there():
    ((first, _second, _third),) = iso._tetrahedron(_tetra(3, 0, 0, 0), 1.0)
    place, _gradient = first
    assert place.tolist() == pytest.approx([2 / 3, 0.0, 0.0])


def test_a_cube_of_one_full_bin_among_empty_ones_is_closed_round_its_centre():
    values = np.zeros((3, 3, 3))
    values[1, 1, 1] = 27.0
    centres = [np.array([0.0, 1.0, 2.0])] * 3
    triangles = iso.isosurface(values, centres, 13.5)
    points = np.array([place for triangle in triangles for place, _gradient in triangle])
    assert triangles and points.min() >= 0.5 and points.max() <= 1.5  # half way to each side


def _facing(screen):
    """The normal of a surface that faces ``screen``'s way once the view has turned it."""
    turn = np.column_stack([VIEW.normal_to_ndc(axis) for axis in np.eye(3)])
    return np.linalg.solve(turn, np.array(screen, float))


def test_a_surface_facing_the_light_is_brighter_than_one_facing_away_and_either_side_is_one():
    ambient = iso.AMBIENT * iso.QA
    facing = iso.luminosity(VIEW, _facing([1.0, 1.0, 1.0]))
    # full on: all its diffuse light, and the reflection's share towards the screen, 1/sqrt 3
    assert facing == pytest.approx(ambient + iso.LIGHT * (iso.QD + iso.QS / np.sqrt(3.0)))
    assert iso.luminosity(VIEW, _facing([-1.0, -1.0, 1.0])) == ambient  # the light behind it
    assert iso.luminosity(VIEW, np.zeros(3)) == ambient  # no gradient, no direction
    assert iso.luminosity(VIEW, -_facing([1.0, 1.0, 1.0])) == pytest.approx(facing)


def test_the_shades_are_the_fill_colours_hue_from_lightness_four_tenths_up():
    shades = iso.shades((0.0, 0.0, 1.0))
    assert len(shades) == iso.SHADES
    lightness = [colorsys.rgb_to_hls(*shade)[1] for shade in shades]
    assert lightness[0] == pytest.approx(0.4) and lightness == sorted(lightness)


def test_a_triangle_is_cut_into_the_bands_its_light_passes_through():
    points = [np.array(p, float) for p in ((0, 0, 0), (1, 0, 0), (0, 1, 0))]
    lights = [0.0, 2.0, 2.0]
    low = iso._band(points, lights, -1.0, 1.0)
    high = iso._band(points, lights, 1.0, 3.0)
    assert len(low) == 3 and len(high) == 4  # the corner, and the rest a quad
    assert iso._band(points, [2.0, 0.0, 0.0], 1.0, 3.0)[0].tolist() == [0.0, 0.0, 0.0]


def test_the_pads_colour_fills_what_is_darker_than_the_first_level_and_the_rest_shades():
    colours = iso.shades((0.0, 0.0, 1.0))
    assert iso._shade(0, colours, "white") == "white"
    assert iso._shade(1, colours, "white") == iso._shade(2, colours, "white") == colours[0]
    assert iso._bounds([0.1, 20.0], [0.15, 9.745]) == [-0.9, 0.15, 9.745, 21.0]


def test_the_surface_is_laid_flat_the_furthest_triangle_first():
    far = [(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))] * 3
    near = [(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0]))] * 3
    ordered = iso._back_to_front(VIEW, [near, far])
    depths = [float(np.mean([p[2] for p in points])) for points, _lights in ordered]
    assert depths == sorted(depths)
    assert iso.iso_polygons(VIEW, [], (0.0, 0.0, 1.0), "white") == []
