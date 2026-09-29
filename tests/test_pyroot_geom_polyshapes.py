"""The ``TGeo`` solids built after they are made: sections, outlines, facets, expressions.

``TGeoTessellated`` reads the ``.obj`` file ROOT's ``visualizeWavefrontObj.C``
reads, and says what ROOT 6.40 said of it: 1598 vertices, 3192 facets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from geomsupport import geometry_session  # noqa: F401
from pyrootsupport import expect

#: A pyramid in the ``.obj`` format: four vertices, four triangular faces.
PYRAMID = "v 0 0 0\nv 2 0 0\nv 0 2 0\nv 0 0 2\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"


def test_a_polycone_and_a_polygon_are_given_their_sections_one_by_one() -> None:
    pcon = ROOT.TGeoPcon("pcon", 0, 360, 2)
    pcon.DefineSection(0, -10, 1, 2)
    pcon.DefineSection(1, 10, 1, 4)
    pgon = ROOT.TGeoPgon(0, 360, 4, 2)
    pgon.DefineSection(0, -1, 0, 1)
    pgon.DefineSection(1, 1, 0, 1)
    expect((pcon.GetZ(1), 10.0), (pcon.GetRmin(0), 1.0), (pcon.GetRmax(1), 4.0),
           (pcon.GetDZ(), 10.0), (pgon.GetNedges(), 4.0),
           (pgon.GetDX(), pytest.approx(np.sqrt(2.0))))  # fmt: skip


def test_an_extruded_polygon_is_its_outline_through_its_sections() -> None:
    xtru = ROOT.TGeoXtru(2)
    assert xtru.DefinePolygon(3, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    xtru.DefineSection(0, -5)
    xtru.DefineSection(1, 5, 1.0, 1.0, 2.0)
    expect((xtru.GetNvert(), 3), (xtru.GetDZ(), 5.0), (xtru.GetDX(), 1.5))


def test_a_tessellated_solid_is_given_facets_by_index_or_by_point(
    capsys: pytest.CaptureFixture[str],
) -> None:
    solid = ROOT.TGeoTessellated("solid", 4)
    solid.AddFacet((0, 0, 0), (1, 0, 0), (0, 1, 0))
    solid.AddFacet(0, 1, 2)
    solid.CloseShape()
    solid.Print()
    from_points = ROOT.TGeoTessellated("again", [(0, 0, 0), (4, 0, 0), (0, 4, 0)])
    from_points.AddFacet(0, 1, 2)
    from_points.ResizeCenter(1.0)
    expect((solid.GetNvertices(), 3), (solid.GetNfacets(), 2), (from_points.GetDX(), 1.0),
           (capsys.readouterr().out,
            "=== Tessellated shape solid having 3 vertices and 2 facets\n"))  # fmt: skip


def test_an_obj_file_is_imported_and_said_as_root_says_it(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyramid.obj").write_text(PYRAMID)
    made = ROOT.TGeoTessellated.ImportFromObjFormat(str(tmp_path / "pyramid.obj"), False, True)
    said = capsys.readouterr().out.splitlines()
    expect((made.GetNfacets(), 4), (made.GetName(), str(tmp_path / "pyramid")),
           (said[0], f"Read 4 vertices and 4 facets from {tmp_path / 'pyramid.obj'}"),
           (said[1], f"=== Tessellated shape {tmp_path / 'pyramid'} having 4 vertices and 4 "
                     f"facets"))  # fmt: skip


def test_an_obj_file_root_would_not_import_gives_none_and_root_s_error(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ROOT.TGeoTessellated.ImportFromObjFormat(str(tmp_path / "none.obj")) is None
    assert "Error in <TGeoTessellated::ImportFromObjFormat>: Unable to open" in (
        capsys.readouterr().err)


def test_a_composite_is_drawn_as_its_components_where_their_matrices_put_them() -> None:
    ROOT.TGeoBBox("A", 1, 1, 1)
    ROOT.TGeoBBox("B", 1, 1, 1)
    shift = ROOT.TGeoTranslation("t1", 10, 0, 0)
    shift.RegisterYourself()
    both = ROOT.TGeoCompositeShape("both", "A + B:t1")
    unregistered = ROOT.TGeoCompositeShape("A - B:nowhere")
    empty = ROOT.TGeoCompositeShape()
    expect((both.IsComposite(), True), (both.GetExpression(), "A + B:t1"), (both.GetDX(), 6.0),
           (unregistered.GetDX(), 1.0), (len(empty.mesh().points), 0))  # fmt: skip


def test_a_composite_of_a_shape_the_geometry_does_not_have_is_refused() -> None:
    with pytest.raises(ValueError, match="names a shape 'Nothing'"):
        ROOT.TGeoCompositeShape("c", "Nothing+Else").mesh()
