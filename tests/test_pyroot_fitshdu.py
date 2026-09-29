"""``TFITSHDU``: a FITS unit opened, printed and read as ROOT's ``TFITSHDU`` does it.

The print formats are ``TFITS.cxx``'s, and ROOT 6.40 printed the same for
the ``FITS_tutorial`` macros' files; the files here are made byte by byte.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from fitssupport import ascii_table, binary_table, card, image_unit
from geomsupport import geometry_session  # noqa: F401
from pyrootsupport import expect


def _be(values: list[Any], kind: str) -> bytes:
    return np.asarray(values, dtype=kind).tobytes()


@pytest.fixture()
def fits(tmp_path: Any) -> str:
    """A primary picture, a binary table with arrays, a named ASCII table and a cube."""
    pixels = np.array([[1, 2, 3], [4, 5, -6]], dtype=np.int16)
    extra = [card("EXPTIME", 5.007, "seconds"), card("", comment="  note")]
    table = binary_table([("x", "E", _be([1.5, 2.5], ">f4")), ("v", "2E", _be([1, 2, 3, 4], ">f4")),
                          ("p", "PI(2)", _be([2, 0, 0, 4], ">i4")), ("w", "K", _be([1, 2], ">i8")),
                          ("q", "PB(1)", _be([1, 0, 0, 0], ">i4"))],
                         2, _be([7, 8], ">i2"), [card("EXTNAME", "'ARRAYS'", "")])  # fmt: skip
    text = ascii_table([("NAME", "A5", 1), ("VAL", "E8.1", 6)], ["alpha   1.5", "be    250."],
                       "TXT")
    cube = image_unit(np.arange(8, dtype=np.float32).reshape(2, 2, 2), -32, first=False)
    path = tmp_path / "t.fits"
    path.write_bytes(image_unit(pixels, 16, extra=extra) + table + text + cube)
    return str(path)


def test_the_file_and_its_units_are_printed_as_root_prints_them(fits: str, capsys: Any) -> None:
    hdu = ROOT.TFITSHDU(fits)
    hdu.Print("F")
    listed = capsys.readouterr().out
    assert listed == ("Total: 4 HDUs\n   [0] IMAGE (PRIMARY)\n\n   [1] BINARY TABLE ('ARRAYS')\n\n"
                      "   [2] ASCII TABLE ('TXT')\n\n   [3] IMAGE ('TXT')\n\n")  # fmt: skip
    hdu.Print("f+")
    lines = capsys.readouterr().out.splitlines()
    assert lines[2:4] == ["      SIMPLE     = T / ", "      BITPIX     = 16 / "]
    assert lines[7:9] == ["      EXPTIME    = 5.007 / seconds", "                 =  /   note"]
    assert lines[11] == "      XTENSION   = 'BINTABLE'"
    hdu.Print()
    records = capsys.readouterr().out.splitlines()
    assert records[5:] == ["EXPTIME    = 5.007 / seconds", "           =  /   note"]


def test_a_units_keywords_and_kind_are_read(fits: str, capsys: Any) -> None:
    image, table = ROOT.TFITSHDU(fits), ROOT.TFITSHDU(fits, "arrays")
    expect((image.GetKeywordValue("EXPTIME").Data(), "5.007"),
           (image.GetKeywordValue("NONE").Data(), ""),
           (image.GetExtensionName().Data(), "PRIMARY"),
           (table.GetExtensionName().Data(), "'ARRAYS'"), (table.GetHDUNumber(), 2),
           ((image.GetType(), table.GetType()), (ROOT.TFITSHDU.kImageHDU, ROOT.TFITSHDU.kTableHDU)),
           (image.GetFilePath().Data(), fits))  # fmt: skip
    err = capsys.readouterr().err
    assert "Image Extension" in err and "Column type 81 is currently not supported" in err
    assert err.count("The variable-length array type in column 5 is unknown") == 2


def test_a_picture_is_read_as_a_matrix_an_image_and_a_histogram(fits: str, capsys: Any) -> None:
    hdu = ROOT.TFITSHDU(fits)
    matrix, scaled = hdu.ReadAsMatrix(0), hdu.ReadAsMatrix(0, "S")
    picture, histogram = hdu.ReadAsImage(0), hdu.ReadAsHistogram()
    expect((matrix.matrix().tolist(), [[1, 2, 3], [4, 5, -6]]),
           (scaled(1, 1), 1.0), (scaled(1, 2), 0.0),
           ((picture.GetWidth(), picture.GetHeight()), (3, 2)),
           (picture.rgba[0, 0].tolist(), [231, 231, 231, 255]),
           ((histogram.GetNbinsX(), histogram.GetNbinsY()), (3, 2)),
           (histogram.GetSumOfWeights(), 3.0), (histogram.GetEntries(), 6.0),
           (hdu.GetArrayRow(1).GetMatrixArray().tolist(), [4, 5, -6]),
           (list(hdu.GetArrayColumn(2)), [3.0, -6.0]),
           (hdu.GetArrayRow(2), None), (hdu.ReadAsMatrix(1), None))  # fmt: skip
    assert "index out of bounds" in capsys.readouterr().err


def test_a_table_is_printed_by_its_columns_and_whole(fits: str, capsys: Any,
                                                      tmp_path: Any) -> None:
    text, arrays, image = (ROOT.TFITSHDU(fits, 2), ROOT.TFITSHDU(fits + "[1]"),
                           ROOT.TFITSHDU(fits))  # fmt: skip
    capsys.readouterr()
    text.Print("T")
    text.Print("t+")
    assert capsys.readouterr().out == (
        "NAME                 : STRING\nVAL                  : REAL NUMBER\n\n"
        "NAME      | VAL       | \n" + "-" * 24 + "\n"
        "alpha     | 1.5       | \nbe        | 2.5e+02   | \n")  # fmt: skip
    arrays.Print("T")
    assert capsys.readouterr().out.splitlines()[1:3] == [
        "v                    : FIXED-LENGTH ARRAY", "p                    : VARIABLE-LENGTH ARRAY"]
    arrays.Print("T+")
    varying = binary_table([("p", "PI(1)", _be([0, 0], ">i4"))], 1)
    (tmp_path / "v.fits").write_bytes(image_unit(np.zeros((0,)), 8) + varying)
    ROOT.TFITSHDU(str(tmp_path / "v.fits[1]")).Print("T+")
    image.Print("T")
    image.Print("T+")
    err = capsys.readouterr().err
    assert "column with fixed-length arrays" in err and "column with variable-length" in err
    assert err.count("this is not a table HDU") == 2


def test_a_tables_columns_and_cells_are_read_by_number_or_name(fits: str, capsys: Any) -> None:
    hdu = ROOT.TFITSHDU(fits, 1)
    names = [hdu.GetColumnName(i).Data() for i in range(hdu.GetTabNColumns())]
    cells = hdu.GetTabRealVectorCells("v")
    expect((names, ["x", "v", "p", "w", "q"]), (hdu.GetTabNRows(), 2),
           (hdu.GetColumnNumber("p"), 2), (hdu.GetColumnNumber("none"), -1),
           (list(hdu.GetTabRealVectorColumn("x")), [1.5, 2.5]),
           (list(hdu.GetTabRealVectorColumn(0)), [1.5, 2.5]),
           ([list(v) for v in cells], [[1.0, 2.0], [3.0, 4.0]]),
           (list(hdu.GetTabRealVectorCell(1, "v")), [3.0, 4.0]),
           (hdu.GetTabVarLengthVectorCell(0, "p").GetArray().tolist(), [7.0, 8.0]),
           (hdu.GetTabVarLengthVectorCell(1, 2).GetSize(), 0))  # fmt: skip
    capsys.readouterr()
    refused = [hdu.GetTabRealVectorColumn("v"), hdu.GetTabRealVectorColumn("p"),
               hdu.GetTabRealVectorCells("p"), hdu.GetTabRealVectorCell(0, "p"),
               hdu.GetTabRealVectorCell(5, "v"), hdu.GetTabVarLengthVectorCell(5, "p"),
               hdu.GetTabRealVectorColumn(9), hdu.GetTabRealVectorColumn("none"),
               hdu.GetTabStringColumn("x"), hdu.GetTabVarLengthVectorCell(0, "none"),
               hdu.GetTabStringColumn("none")]  # fmt: skip
    assert refused == [None] * len(refused)
    assert hdu.GetColumnName(9).Data() == ""
    err = capsys.readouterr().err
    for said in ("fixed-length arrays", "Use GetTabVarLengthCell() instead", "row index out of",
                 "column index out of", "column not found", "not of type 'kString'",
                 "Error in <GetColumnName>: column index out of bounds"):  # fmt: skip
        assert said in err


def test_strings_and_the_wrong_kind_of_unit_are_read_as_root_reads_them(fits: str,
                                                                         capsys: Any) -> None:
    text, image = ROOT.TFITSHDU(fits, "TXT"), ROOT.TFITSHDU(fits)
    names = text.GetTabStringColumn(0)
    expect(([names.At(i).GetName() for i in range(2)], ["alpha", "be"]),
           (image.GetTabNColumns(), 0), (image.GetColumnName(0).Data(), ""),
           (image.GetTabRealVectorColumn(0), None), (text.ReadAsImage(), None),
           (text.ReadAsMatrix(), None), (text.ReadAsHistogram(), None),
           (text.GetArrayRow(0), None))  # fmt: skip
    err = capsys.readouterr().err
    assert "this is not an image HDU" in err and "Table Extension" in err


def test_a_cube_and_a_line_are_histograms_and_not_pictures(fits: str, tmp_path: Any,
                                                           capsys: Any) -> None:  # fmt: skip
    cube = ROOT.TFITSHDU(fits, 3)
    line_path = tmp_path / "l.fits"
    line_path.write_bytes(image_unit(np.array([3, 0, 2], dtype=np.uint8), 8))
    line = ROOT.TFITSHDU(str(line_path))
    line.Draw()
    flat_path = tmp_path / "f.fits"
    flat_path.write_bytes(image_unit(np.full((2, 2), 4.0), -64))
    flat = ROOT.TFITSHDU(str(flat_path))
    four_path = tmp_path / "4.fits"
    four_path.write_bytes(image_unit(np.zeros((1, 1, 1, 1, 1)), -64))
    expect((cube.ReadAsHistogram().GetEntries(), 8.0), (cube.ReadAsImage(1).GetHeight(), 2),
           (cube.ReadAsImage(2), None), (cube.GetArrayRow(0), None),
           (line.ReadAsHistogram().GetNbinsX(), 3), (line.ReadAsImage(), None),
           (flat.ReadAsMatrix(0, "s"), None), (np.ptp(flat.ReadAsImage().rgba[..., 0]), 0),
           (ROOT.TFITSHDU(str(four_path)).ReadAsHistogram(), None))  # fmt: skip
    err = capsys.readouterr().err
    assert "because it has 1 dimensions" in err and "because it has 5 dimensions" in err
    assert "layer out of bounds" in err and "could not get row" in err


def test_a_unit_that_cannot_be_opened_is_roots_warning_and_an_error(tmp_path: Any,
                                                                   capsys: Any) -> None:
    with pytest.raises(OSError, match="could not open"):
        ROOT.TFITSHDU(str(tmp_path / "none.fits"))
    (tmp_path / "bad.fits").write_bytes(b" " * 2880)
    with pytest.raises(OSError):
        ROOT.TFITSHDU(str(tmp_path / "bad.fits"))
    assert capsys.readouterr().err.count("error opening FITS file. Details:") == 2


def test_a_unit_changes_to_another_and_keeps_its_own_when_it_cannot(fits: str,
                                                                    capsys: Any) -> None:
    hdu = ROOT.TFITSHDU(fits)
    expect((hdu.Change(2), True), (hdu.GetTabNColumns(), 2), (hdu.Change("[9]"), False),
           (hdu.GetTabNColumns(), 2), (hdu.Change("[ARRAYS][x > 2]"), True),
           (hdu.GetTabNRows(), 1))  # fmt: skip
    assert "Restoring the previous one" in capsys.readouterr().err


def test_a_picture_is_drawn_in_a_canvas_of_its_size(fits: str, tmp_path: Any,
                                                    capsys: Any) -> None:
    hdu = ROOT.TFITSHDU(fits)
    hdu.Draw()
    canvas = ROOT.gPad.GetCanvas()
    expect((canvas.GetName(), "HDU"), (canvas.GetTitle(), "3 x 2"),
           (canvas.GetListOfPrimitives().At(0).ClassName(), "TASImage"))  # fmt: skip
    canvas.SaveAs(str(tmp_path / "hdu.png"))
    ROOT.TFITSHDU(fits, 1).Draw()
    assert (tmp_path / "hdu.png").exists()
    assert "cannot draw. This is not an image HDU" in capsys.readouterr().err
