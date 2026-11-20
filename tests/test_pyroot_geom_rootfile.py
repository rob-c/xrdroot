"""``TGeoManager::Import`` reading a ``TGeoManager`` back from a ROOT file.

``TGeoManager::Export`` writes a geometry as a ROOT file and ``Import`` reads
it again; ROOT's ``roottest/root/io/geo`` keeps eighteen such files, and the
test of that reads them all. The file here is a small one ROOT wrote, with a
box, a tube and a cone, a material and a mixture, so that what the manager is
made of can be compared with what ROOT says closing the same file says.
"""

from __future__ import annotations

from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from geomsupport import DATA, geometry_session  # noqa: F401
from pyrootsupport import expect
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.geom import rootfile

#: The small geometry ROOT wrote: ``mkgeom``-made, three volumes and two materials.
GEOMETRY = DATA / "geom-small-6.40.root"
#: A ROOT file of ROOT's old ``TGeometry``, which holds no ``TGeoManager``.
NO_GEOMETRY = DATA / "tgeometry-small-6.40.root"


def test_importing_a_geometry_from_a_root_file_says_what_root_says_reading_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    geometry = ROOT.TGeoManager.Import(str(GEOMETRY))
    said = capsys.readouterr().err.splitlines()
    expect((said[0], f"Info in <TGeoManager::Import>: Reading geometry from file: {GEOMETRY}"),
           (said[1], "Info in <TGeoManager::CloseGeometry>: Geometry loaded from file..."),
           (said[2], "Info in <TGeoManager::SetTopVolume>: Top volume is WORLD. Master "
                     "volume is WORLD"),
           (said[3], "Info in <TGeoNavigator::BuildCache>: --- Maximum geometry depth set to "
                     "100"),
           (said[4], "Info in <TGeoManager::CloseGeometry>: Voxelization retrieved from file"),
           (said[5], "Info in <TGeoManager::CountLevels>: max level = 2, max placements = 1"),
           (said[6], "Info in <TGeoManager::CloseGeometry>: 3 nodes/ 3 volume UID's in two "
                     "volumes"),
           (said[7], "Info in <TGeoManager::CloseGeometry>: ----------------modeler ready"
                     "----------------"),
           (len(said), 8),
           (geometry.GetTitle(), "two volumes"),
           (geometry.GetTopVolume().GetName(), "WORLD"),
           (geometry.IsClosed(), True),
           (ROOT.gGeoManager == geometry, True))


def test_the_numbers_a_file_counted_are_the_ones_closing_it_reports() -> None:
    """ROOT counts a geometry's nodes when it writes them, and says so again reading it back."""
    geometry = ROOT.TGeoManager.Import(str(GEOMETRY))
    expect((geometry.GetListOfVolumes().GetEntries(), 3),
           (geometry.GetTopVolume().GetNdaughters(), 1),
           (geometry.GetVolume("PIPE").GetNdaughters(), 1),
           (geometry.GetVolume("PIPE").GetShape().GetKind()
            if isinstance(geometry.GetVolume("PIPE").GetShape(), rootfile.TGeoImportedShape)
            else type(geometry.GetVolume("PIPE").GetShape()).__name__, "TGeoTube"),
           (geometry.GetVolume("WORLD").GetShape().GetDX(), 100.0),
           (geometry.GetVolume("CONE").GetShape().GetRmax1(), 10.0))


def test_a_volume_is_made_of_the_medium_and_material_the_file_names() -> None:
    geometry = ROOT.TGeoManager.Import(str(GEOMETRY))
    expect(([geometry.GetListOfMaterials().At(index).GetName() for index in range(2)],
            ["IRON", "AIR"]),
           (geometry.GetMaterial("IRON").GetA(), 55.85),
           (geometry.GetMaterial("AIR").GetNelements(), 2),
           (geometry.GetVolume("PIPE").GetMaterial().GetName(), "IRON"),
           (geometry.GetVolume("WORLD").GetMaterial().GetName(), "AIR"),
           (geometry.GetMedium("air").GetId(), 2))


def test_a_second_geometry_read_in_one_process_is_not_told_the_cache_depth_again(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT's navigator makes room for one geometry depth per process, and says it once."""
    ROOT.TGeoManager.Import(str(GEOMETRY))
    capsys.readouterr()
    ROOT.TGeoManager.Import(str(GEOMETRY))
    said = capsys.readouterr().err
    expect(("BuildCache" in said, False), ("Geometry loaded from file..." in said, True))


def test_a_root_file_holding_no_geometry_is_refused_by_name() -> None:
    with pytest.raises(UnsupportedFeatureError, match=r"holds no TGeoManager"):
        ROOT.TGeoManager.Import(str(NO_GEOMETRY))


def test_a_manager_read_as_no_more_than_its_name_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record the reader could make nothing of says so, rather than passing off an empty
    geometry as the one the file holds."""

    class Key:
        @staticmethod
        def GetName() -> str:
            return "Geometry"

        @staticmethod
        def GetClassName() -> str:
            return "TGeoManager"

    class Managed:
        @staticmethod
        def GetListOfKeys() -> list[Any]:
            return [Key()]

        @staticmethod
        def Get(name: str) -> Any:
            from xrdroot.pyroot.geom import manager

            return manager._CURRENT[0]

    class Reading:
        _xrd = None

        @staticmethod
        def GetListOfKeys() -> list[Any]:
            return [Key()]

        @staticmethod
        def Get(name: str) -> Any:
            return object()

    monkeypatch.setattr(rootfile, "TFile", type("TFile", (), {"Open": staticmethod(
        lambda filename: Reading())}))
    with pytest.raises(UnsupportedFeatureError, match=r"was read as no more than its name"):
        rootfile.from_file("some.root")
    monkeypatch.setattr(rootfile, "TFile", type("TFile", (), {"Open": staticmethod(
        lambda filename: Managed())}))
    made = ROOT.TGeoManager()
    expect((rootfile.from_file("some.root") == made, True))


def _members() -> dict[str, Any]:
    """A manager's members, as the reader hands them back: two volumes, a mixture and a solid
    of a kind xrdroot has no class for."""
    return {
        "TNamed": {"fName": "made", "fTitle": "made by hand", "fUniqueID": 0, "fBits": 0},
        "fNNodes": 3,
        "fMaterials": [{"TNamed": {"fName": "AIR", "fTitle": ""}, "fNelements": 2,
                        "fZmixture": [7.0, 8.0], "fAmixture": [14.01, 16.0],
                        "fWeights": [0.755, 0.245],
                        "TGeoMaterial": {"fDensity": 1.2e-3}}],
        "fMedia": [{"TNamed": {"fName": "gas", "fTitle": ""}, "fId": 3,
                    "fMaterial": {"TNamed": {"fName": "AIR", "fTitle": ""}},
                    "fParams": [0.0, 3.0]}],
        "fVolumes": [
            {"TNamed": {"fName": "WORLD", "fTitle": "the world"}, "fNumber": 1,
             "fMedium": {"TNamed": {"fName": "gas", "fTitle": ""}},
             "fShape": {"TGeoBBox": {"fDX": 1.0, "fDY": 2.0, "fDZ": 3.0}},
             "fNodes": [{"TGeoNode": {"fVolume": {"fNumber": 2}}},
                        {"TGeoNode": {"fVolume": {"fNumber": 2}}},
                        {"TGeoNode": {"fVolume": {"fNumber": 9}}},
                        {"TGeoNode": {}}]},
            {"TNamed": {"fName": "WORLD", "fTitle": "the world"}, "fNumber": 1,
             "fNodes": []},
            {"TNamed": {"fName": "OUTER", "fTitle": ""}, "fNumber": 2, "fNodes": None,
             "fShape": {"TGeoPcon": {"TNamed": {"fName": "poly"}, "fNz": 2,
                                     "TGeoShape": {"fName": "poly"}}}},
        ],
        "fUniqueVolumes": [{"TNamed": {"fName": "", "fTitle": ""}},
                           {"TNamed": {"fName": "WORLD", "fTitle": "the world"}}],
    }


def test_a_mixture_keeps_its_parts_and_the_material_it_derives_from_keeps_its_density() -> None:
    geometry = rootfile._built(_members())
    mixture = geometry.GetMaterial("AIR")
    expect((geometry.GetTitle(), "made by hand"),
           (mixture.GetNelements(), 2),
           (mixture.GetDensity(), 1.2e-3),
           (geometry.GetMedium("gas").GetId(), 3),
           (geometry.GetVolume("WORLD").GetTitle(), "the world"),
           (geometry.GetVolume("WORLD").GetMedium().GetName(), "gas"),
           (geometry.GetVolume("WORLD").GetShape().GetDY(), 2.0))


def test_a_volume_of_no_daughters_has_no_daughters_to_count() -> None:
    geometry = rootfile._built(_members())
    expect((geometry.GetVolume("OUTER").GetNdaughters(), 0),
           (geometry.GetTopVolume().GetName(), "WORLD"),
           (geometry.GetListOfVolumes().GetEntries(), 2))


def test_a_solid_of_a_kind_xrdroot_has_no_class_for_is_refused_by_name_when_it_is_drawn() -> None:
    geometry = rootfile._built(_members())
    polycone = geometry.GetVolume("OUTER").GetShape()
    expect((type(polycone).__name__, "TGeoImportedShape"), (polycone.GetKind(), "TGeoPcon"))
    with pytest.raises(UnsupportedFeatureError, match=r"is a TGeoPcon, which xrdroot.pyroot"):
        polycone.mesh()


def test_a_polycone_is_built_from_the_sections_the_file_keeps_in_three_arrays() -> None:
    members = _members()
    members["fVolumes"][0]["fShape"] = {
        "TGeoBBox": {"TGeoShape": {"fName": "poly"}, "fDX": 1.0, "fDY": 1.0, "fDZ": 9.0},
        "fNz": 2, "fPhi1": 10.0, "fDphi": 350.0,
        "fZ": [-1.0, 1.0], "fRmin": [0.5, 0.25], "fRmax": [2.0, 1.5]}
    geometry = rootfile._built(members)
    polycone = geometry.GetVolume("WORLD").GetShape()
    expect((type(polycone).__name__, "TGeoPcon"),
           (polycone.GetPhi1(), 10.0),
           (polycone.GetDphi(), 350.0),
           (polycone.GetNz(), 2),
           (polycone.GetZ(1), 1.0),
           (polycone.GetRmin(0), 0.5),
           (polycone.GetRmax(1), 1.5))
    assert polycone.mesh().faces  # a surface, so the polygon can be drawn


def test_a_polygon_of_flat_sides_is_built_from_its_sections_and_its_edge_count() -> None:
    members = _members()
    members["fVolumes"][0]["fShape"] = {
        "TGeoBBox": {"TGeoShape": {"fName": "poly"}, "fDX": 1.0, "fDY": 1.0, "fDZ": 9.0},
        "fNz": 1, "fNedges": 8, "fPhi1": 0.0, "fDphi": 360.0,
        "fZ": [0.0], "fRmin": [0.0], "fRmax": [3.0]}
    geometry = rootfile._built(members)
    polygon = geometry.GetVolume("WORLD").GetShape()
    expect((type(polygon).__name__, "TGeoPgon"),
           (polygon.GetNedges(), 8),
           (polygon.GetDphi(), 360.0),
           (polygon.GetRmax(0), 3.0))
    assert polygon.mesh().faces  # eight flat sides, drawn as their faces


def test_a_sphere_whose_file_keeps_its_segment_count_is_not_taken_for_a_tube() -> None:
    members = _members()
    members["fVolumes"][0]["fShape"] = {
        "TGeoBBox": {"TGeoShape": {"fName": "mirror"}, "fDX": 5.0, "fDY": 5.0, "fDZ": 5.0},
        "fRmin": 3.0, "fRmax": 5.0, "fTheta1": 20.0, "fTheta2": 160.0, "fPhi1": 0.0,
        "fPhi2": 90.0, "fNseg": 12, "fNz": 8}
    geometry = rootfile._built(members)
    mirror = geometry.GetVolume("WORLD").GetShape()
    expect((type(mirror).__name__, "TGeoSphere"),
           (mirror.GetRmin(), 3.0),
           (mirror.GetTheta2(), 160.0),
           (mirror.GetPhi2(), 90.0))


def test_a_solid_missing_one_of_its_numbers_is_kept_and_refused_rather_than_guessed() -> None:
    members = _members()
    members["fVolumes"][0]["fShape"] = {"TGeoTube": {"fRmin": 1.0, "fRmax": 2.0}}
    geometry = rootfile._built(members)
    shape = geometry.GetVolume("WORLD").GetShape()
    expect((type(shape).__name__, "TGeoImportedShape"), (shape.GetKind(), "TGeoTube"))


def test_a_file_that_never_lists_its_volume_uids_is_counted_by_the_volumes_it_lists() -> None:
    members = _members()
    del members["fUniqueVolumes"]
    geometry = rootfile._built(members)
    expect((geometry.GetTopVolume().GetName(), "WORLD"),
           (geometry.GetListOfVolumes().GetEntries(), 2))


def test_a_file_naming_no_master_volume_hangs_its_tree_from_the_first() -> None:
    members = _members()
    members["fMasterVolume"] = {"TNamed": {"fName": "NOWHERE", "fTitle": ""}}
    expect((rootfile._built(members).GetTopVolume().GetName(), "WORLD"))


def test_a_solid_whose_numbers_lie_in_its_base_classs_part_is_still_made() -> None:
    """ROOT writes a solid's numbers under the class that serialised them, which is the base
    class where the solid was written that way; the numbers say which solid it is."""
    members = _members()
    members["fVolumes"][0]["fShape"] = {"TGeoBBox": {"fDX": 1.0, "fDY": 2.0, "fDZ": 3.0,
                                                     "TGeoShape": {"fName": "box"}}}
    expect((type(rootfile._built(members).GetVolume("WORLD").GetShape()).__name__, "TGeoBBox"))


def test_an_arb8_is_made_from_the_eight_corners_the_file_holds() -> None:
    members = _members()
    corners = [float(n) for n in range(16)]
    members["fVolumes"][0]["fShape"] = {"TGeoArb8": {"fDz": 4.0, "fXY": corners}}
    shape = rootfile._built(members).GetVolume("WORLD").GetShape()
    expect((type(shape).__name__, "TGeoArb8"), (shape.GetVertices().tolist(), corners))


def test_a_volume_of_no_solid_at_all_is_a_volume_of_no_solid() -> None:
    members = _members()
    members["fVolumes"][0]["fShape"] = None
    expect((rootfile._built(members).GetVolume("WORLD").GetShape(), None))


def test_a_geometry_read_from_a_file_is_written_back_as_the_file_it_came_from(
    tmp_path: Any,
) -> None:
    """What ``Export`` writes is the record the geometry was read from, so reading it back
    gives the same tree: the export ``roottest/root/io/geo`` asks for of every geometry it
    imports."""
    target = tmp_path / "exported.export.root"
    read = ROOT.TGeoManager.Import(str(GEOMETRY))
    expect((read.Export(str(target)), 1))
    written = ROOT.TGeoManager.Import(str(target))
    expect((written.GetTitle(), read.GetTitle()),
           (written.GetTopVolume().GetName(), read.GetTopVolume().GetName()),
           (written.GetListOfVolumes().GetEntries(), read.GetListOfVolumes().GetEntries()),
           (written.GetListOfMaterials().GetEntries(),
            read.GetListOfMaterials().GetEntries()),
           (written.GetListOfMedia().GetEntries(), read.GetListOfMedia().GetEntries()))


def test_what_export_writes_is_read_by_rown_as_the_geometry_that_went_in(
    tmp_path: Any,
) -> None:
    """The file is ROOT's own: it describes the classes it holds, so ROOT rebuilding the
    manager from it counts the nodes the geometry had."""
    target = tmp_path / "counted.export.root"
    read = ROOT.TGeoManager.Import(str(GEOMETRY))
    read.Export(str(target))
    written = ROOT.TGeoManager.Import(str(target))
    expect((written.GetVolume("WORLD").GetNdaughters(),
            read.GetVolume("WORLD").GetNdaughters()),
           (written.GetVolume("PIPE").GetShape().GetKind()
            if isinstance(written.GetVolume("PIPE").GetShape(), rootfile.TGeoImportedShape)
            else type(written.GetVolume("PIPE").GetShape()).__name__,
            type(read.GetVolume("PIPE").GetShape()).__name__))


def test_a_geometry_made_in_memory_is_refused_by_name_and_told_what_to_do(
    tmp_path: Any,
) -> None:
    """A manager built by hand is only the tree xrdroot models, which is not what ROOT's
    ``TGeoManager`` streams: writing it would be a file ROOT misreads."""
    made = ROOT.TGeoManager("made", "in memory")
    with pytest.raises(UnsupportedFeatureError, match=r"made in memory.*\.gdml instead"):
        made.Export(str(tmp_path / "made.export.root"))


def test_an_xml_file_name_is_refused_for_the_flavour_of_root_file_it_names(
    tmp_path: Any,
) -> None:
    """ROOT writes a ``.xml`` name as its own format in XML, which xrdroot writes no
    flavour of; the refusal says so, and names both files that would work."""
    with pytest.raises(UnsupportedFeatureError,
                       match=r"XML flavour.*\.root, or export the geometry to .*\.gdml"):
        ROOT.TGeoManager("made", "in memory").Export(str(tmp_path / "made.xml"))


def test_a_streamed_record_is_written_as_the_bytes_it_was_read_with() -> None:
    """The record counts every length inside itself from its own start, so nothing of it
    depends on the key in front: the writer is handed the same bytes, and the descriptions
    the source file gave the classes, whatever key it is written under."""
    carried = {("TGeoManager", 1): b"a layout"}
    streamed = rootfile._Streamed("TGeoManager", "a geometry", b"\x00\x01\x02", carried)
    expect((streamed.root_payload(0), (b"\x00\x01\x02", (), carried)),
           (streamed.root_payload(4096), (b"\x00\x01\x02", (), carried)),
           ((streamed.classname, streamed.title), ("TGeoManager", "a geometry")))
