"""Reading the ROOT files ROOT's own tutorials ship, or download.

These are under ``tests/data/tutorials`` because each once failed to open:
two trees older than ROOT 5 whose fixed fields no hard-coded layout had
right, a gallery of pictures, which stream themselves as PNGs, and a
detector geometry whose materials hold a null array. The tree values
asserted were checked against uproot where uproot can read the file
(``mlpHiggs.root``) and against ROOT 6.40 everywhere; ``stock.root`` is
compressed with the pre-2005 algorithm, which uproot does not undo, and its
values are the prices the tutorial says it holds. The pixels were checked
against Pillow's decoding of the same PNGs.
"""

from __future__ import annotations

import pathlib
import struct
import zlib

import numpy as np
import pytest

from xrdroot import FormatError, Image, UnsupportedFeatureError, open_root
from xrdroot.buffer import BYTE_COUNT_MASK, Buffer
from xrdroot.image import decode_png, read_image
from xrdroot.known import read_recorder

TUTORIALS = pathlib.Path(__file__).parent / "data" / "tutorials"


def test_a_tree_written_by_ROOT_3_reads_the_events_the_tutorial_trains_on():
    """``mlpHiggs.root`` is ROOT 3.04's, with ``TTree`` version 9: doubles and a weight."""
    with open_root(TUTORIALS / "mlpHiggs.root") as root:
        assert root.trees() == ["bg_filtered", "sig_filtered"]
        background, signal = root["bg_filtered"], root["sig_filtered"]
        assert (len(background), len(signal)) == (1350, 608)
        assert background.title == "Filtered background (WW) events"
        assert set(background.typenames().values()) == {"float32"}
        first = background.arrays(entry_stop=2)
        assert first["nch"].tolist() == [17.0, 10.0]
        assert first["acolin"].tolist() == pytest.approx([162.728989, 103.899628])
        assert float(signal["nch"].array().sum()) == 10504.0
        assert float(signal["qelep"].array().sum()) == 114912.0


def test_a_tree_written_by_ROOT_4_reads_the_prices_the_portfolio_is_made_of():
    """``stock.root`` is ROOT 4.00's: ten stocks, 974 trading days, prices in cents."""
    with open_root(TUTORIALS / "stock.root") as root:
        assert len(root.trees()) == 10
        ge = root["GE"]
        assert (len(ge), ge.title, ge.unreadable) == (974, "GE daily stock data from Yahoo", {})
        assert ge["fDate"].array()[[0, -1]].tolist() == [20040602, 20000908]
        assert ge["fClose"].array()[[0, -1]].tolist() == [3110, 5988]
        assert root["IBM"]["daily"].array(0, 1) == [
            {
                "fDate": 20040602,
                "fOpen": 8864,
                "fHigh": 8864,
                "fLow": 8789,
                "fClose": 8798,
                "fVol": 3912600,
                "fCloseAdj": 8798,
            }
        ]


def test_a_geometry_reads_whole_with_the_volumes_and_materials_ROOT_finds_in_it():
    """``brahms.root`` is the BRAHMS detector the ``geomBrahms`` tutorial imports.

    Its mixtures keep ``fNatoms`` as a null pointer, which ROOT writes as a
    zero marker and no values - the byte that once threw the rest out. The
    counts, names and densities are the ones ROOT 6.40 reads from it.
    """
    with open_root(TUTORIALS / "brahms.root") as root:
        geometry = root["brahms"]
    assert (len(geometry["fVolumes"]), len(geometry["fShapes"]), len(geometry["fMedia"])) == (
        278,
        279,
        37,
    )
    materials = geometry["fMaterials"]
    names = [(m.get("TGeoMaterial") or m)["TNamed"]["fName"] for m in materials]
    assert (len(names), names[:5]) == (31, ["helium", "carbon", "aluminum", "iron", "air"])
    assert (materials[3]["fDensity"], materials[9]["TGeoMaterial"]["fDensity"]) == (7.87, 0.000717)
    assert materials[9]["fNatoms"].tolist() == []
    cave = geometry["fMasterVolume"]
    assert (cave["TNamed"]["fName"], cave["TNamed"]["fTitle"]) == ("CAVE", "Top volume")
    assert [cave["fShape"][d] for d in ("fDX", "fDY", "fDZ")] == [800.0, 250.0, 2500.0]
    assert len([node for node in cave["fNodes"] if node is not None]) == 23


def test_a_GUI_recording_reads_its_recorder_and_the_events_it_replays():
    """``fitpanel_playback.root`` holds a ``TRecorder`` its file never describes.

    The recorder is read by the layout ``TRecorder.h`` declares; the sums are
    ROOT 6.40's over the first thousand GUI events.
    """
    with open_root(TUTORIALS / "fitpanel_playback.root") as root:
        assert root["recorder"] == {
            "TObject": {"fUniqueID": 0, "fBits": 0x03000000},
            "fFilename": "TransientTests.root",
        }
        events = root["GuiEvents"]
        assert (len(events), len(root["WindowsTree"])) == (7814, 1274)
        assert int(events["fType"].array(0, 1000).sum()) == 8067
        assert int(events["fCode"].array(0, 1000).sum()) == 181696268741
        assert root["CmdEvents"]["fText"].array() == [".x qa2.C"]


def test_a_class_the_file_does_not_describe_is_read_by_its_declaration_only_at_its_version():
    raw = struct.pack(">IH", BYTE_COUNT_MASK | 2, 3)
    with pytest.raises(UnsupportedFeatureError, match="TRecorder is version 3"):
        read_recorder(None)(Buffer(raw))


def test_a_picture_in_a_file_reads_as_the_PNG_it_was_written_as():
    with open_root(TUTORIALS / "gallery.root") as root:
        assert set(root.classnames().values()) == {"TASImage"}
        image = root["hsimple.png."]
    assert isinstance(image, Image)
    assert repr(image) == "<Image 'hsimple.png.' of 696x472 pixels>"
    assert (image.width, image.height, len(image.png)) == (696, 472, 10670)
    assert image.title.startswith("/* hsimple.png. */")


def test_a_picture_in_a_file_decodes_to_its_pixels_and_saves_as_any_format(tmp_path):
    with open_root(TUTORIALS / "gallery.root") as root:
        image = root["hsimple.png."]
    pixels = image.array
    assert (pixels.shape, pixels.dtype) == ((472, 696, 4), np.uint8)
    assert pixels is image.array
    assert pixels[0, 0].tolist() == [248, 252, 48, 255]
    assert pixels.reshape(-1, 4).sum(axis=0).tolist() == [65698784, 55660128, 41259496, 83770560]
    assert image._repr_png_() == image.png
    image.save(tmp_path / "copy.png")
    assert (tmp_path / "copy.png").read_bytes() == image.png
    image.save(tmp_path / "copy.jpg")
    assert (tmp_path / "copy.jpg").read_bytes()[:2] == b"\xff\xd8"


def test_a_picture_saved_as_other_than_PNG_without_matplotlib_says_what_to_install(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(__import__("sys").modules, "matplotlib", None)
    image = Image("TASImage", {"fName": "p", "fTitle": "", "png": png_bytes(2, [b"\x00" * 3])})
    with pytest.raises(UnsupportedFeatureError, match="install xrdroot\\[plot\\]"):
        image.save(tmp_path / "p.jpg")


# -- crafted PNGs ---------------------------------------------------------


def chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def png_bytes(
    colour: int,
    rows: list[bytes],
    *,
    filters: list[int] | None = None,
    depth: int = 8,
    interlace: int = 0,
    extra: bytes = b"",
    width: int = 1,
) -> bytes:
    """A PNG of the rows given, each already filtered by the filter named for it."""
    filters = filters or [0] * len(rows)
    raw = b"".join(bytes([f]) + row for f, row in zip(filters, rows))
    head = struct.pack(">IIBBBBB", width, len(rows), depth, colour, 0, 0, interlace)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", head)
        + extra
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_every_colour_type_a_PNG_has_decodes_to_RGBA():
    grey = decode_png(png_bytes(0, [b"\x07"]))
    assert grey.tolist() == [[[7, 7, 7, 255]]]
    shaded = decode_png(png_bytes(4, [b"\x07\x80"]))
    assert shaded.tolist() == [[[7, 7, 7, 128]]]
    colour = decode_png(png_bytes(2, [b"\x01\x02\x03"]))
    assert colour.tolist() == [[[1, 2, 3, 255]]]
    clear = decode_png(png_bytes(6, [b"\x01\x02\x03\x04"]))
    assert clear.tolist() == [[[1, 2, 3, 4]]]
    palette = chunk(b"PLTE", b"\x0a\x0b\x0c\x14\x15\x16") + chunk(b"tRNS", b"\x40")
    indexed = decode_png(png_bytes(3, [b"\x00\x01"], extra=palette, width=2))
    assert indexed.tolist() == [[[10, 11, 12, 64], [20, 21, 22, 255]]]


def test_every_filter_a_PNG_row_is_written_under_is_taken_off():
    """Three grey pixels a row, the second and later rows filtered four ways."""
    rows = [b"\x0a\x14\x1e", b"\x01\x01\x01", b"\x05\x05\x05", b"\x02\x03\x04", b"\xff\x00\x01"]
    pixels = decode_png(png_bytes(0, rows, filters=[0, 1, 2, 3, 4], width=3))[..., 0]
    assert pixels[0].tolist() == [10, 20, 30]
    assert pixels[1].tolist() == [1, 2, 3]  # sub: a running sum
    assert pixels[2].tolist() == [6, 7, 8]  # up: plus the row above
    assert pixels[3].tolist() == [5, 9, 12]  # average of the left and the upper
    # paeth: the upper for the first; then whichever of left, upper, corner is nearest
    assert pixels[4].tolist() == [4, 9, 13]


def test_a_PNG_this_reader_cannot_decode_is_refused_by_what_it_is():
    with pytest.raises(UnsupportedFeatureError, match="16-bit, colour type 2, and"):
        decode_png(png_bytes(2, [b"\x00" * 6], depth=16))
    with pytest.raises(UnsupportedFeatureError, match="colour type 0, interlaced"):
        decode_png(png_bytes(0, [b"\x00"], interlace=1))
    with pytest.raises(FormatError, match="names filter 9"):
        decode_png(png_bytes(0, [b"\x00"], filters=[9]))
    with pytest.raises(FormatError, match="its bytes are not one"):
        decode_png(b"GIF89a" + bytes(30))


def test_a_PNG_whose_pixels_are_broken_says_so():
    whole = png_bytes(0, [b"\x00"])
    head = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    with pytest.raises(FormatError, match="would not inflate"):
        decode_png(head + chunk(b"IDAT", b"not deflate"))
    with pytest.raises(FormatError, match="inflated to 2 bytes, where 4 were promised"):
        decode_png(head.replace(struct.pack(">II", 1, 1), struct.pack(">II", 1, 2)) + whole[33:])


# -- crafted TASImage records ----------------------------------------------


def image_record(version: int, kept: int, body: bytes = b"") -> Buffer:
    named = struct.pack(">HII", 1, 0, 0) + b"\x01p\x00"
    tnamed = struct.pack(">IH", BYTE_COUNT_MASK | (2 + len(named)), 1) + named
    inside = struct.pack(">H", version) + tnamed + bytes([kept]) + body
    return Buffer(struct.pack(">I", BYTE_COUNT_MASK | len(inside)) + inside)


def test_a_TASImage_record_holds_its_name_and_its_PNG():
    png = png_bytes(0, [b"\x00"])
    row = read_image(None)(image_record(2, 1, struct.pack(">i", len(png)) + png))
    assert row == {"fName": "p", "fTitle": "", "png": png}


def test_a_TASImage_with_no_picture_in_it_is_refused_by_what_it_kept():
    read = read_image(None)
    with pytest.raises(UnsupportedFeatureError, match="version 1 whose bytes do not go on"):
        read(image_record(1, 0x40))
    with pytest.raises(UnsupportedFeatureError, match="version 0 whose bytes"):
        read(image_record(0, 1))


def vector_image_body(
    values: list[list[float]], points: list[float], levels: list[list[int]]
) -> bytes:
    """What follows the byte saying an image is numbers: its ``TAttImage``, then the grid.

    ``values`` is given top row first and written bottom row first, as ROOT
    writes it; ``levels`` is the red, green, blue and alpha at each stop.
    """
    n = len(points)
    palette = struct.pack(">HII", 1, 0, 0) + struct.pack(">I", n)
    palette += struct.pack(f">{n}d", *points)
    for channel in levels:
        palette += struct.pack(f">{n}H", *channel)
    palette = struct.pack(">IH", BYTE_COUNT_MASK | (2 + len(palette)), 1) + palette
    attributes = struct.pack(">HiIB", 1, 0, 0, 1) + palette
    attributes = struct.pack(">I", BYTE_COUNT_MASK | len(attributes)) + attributes
    height, width = len(values), len(values[0])
    grid = [value for row in reversed(values) for value in row]
    return attributes + struct.pack(f">ii{width * height}d", width, height, *grid)


def test_an_image_made_of_numbers_is_coloured_the_way_ROOT_colours_it():
    """Three stops over the range 0 to 4: black, mid grey at 1, then white, all opaque.

    The stops stretch over the values' own range, a value between two is
    interpolated at 16 bits and truncated, and the top eight bits are kept.
    """
    values = [[0.0, 1.0], [2.0, 4.0]]
    levels = [[0, 0x8000, 0xFFFF]] * 3 + [[0xFFFF] * 3]
    body = vector_image_body(values, [0.0, 0.25, 1.0], levels)
    row = read_image(None)(image_record(2, 0, body))
    image = Image("TASImage", row)
    assert (image.width, image.height, repr(image)) == (2, 2, "<Image 'p' of 2x2 pixels>")
    assert image.values.tolist() == values
    assert image.palette["fPoints"].tolist() == [0.0, 0.25, 1.0]
    # 2 is a third of the way from 1 to 4: 0x8000 + 0x7FFF / 3, whose top byte is 0xAA
    assert image.array[..., 0].tolist() == [[0, 0x80], [0xAA, 0xFF]]
    assert image.array[..., 3].tolist() == [[255, 255], [255, 255]]
    assert np.array_equal(decode_png(image.png), image.array)
    assert image.png is image.png


def test_the_walk_to_a_stop_carries_on_from_the_last_value_and_past_the_last_stop():
    """Stops at 0, 2.25 and 4.5 over values to 9: the walk starts at the middle stop.

    5 is coloured from it; the second 5 takes the same stop without a walk; 0
    walks down to the first; 9 walks up and is held at the last segment, its
    colour carried on past the last stop and clipped at the top of the range;
    and 1 walks back down from there.
    """
    levels = [[0, 0x4000, 0x8000]] * 4
    body = vector_image_body([[5.0, 5.0, 0.0, 9.0, 1.0]], [0.0, 0.25, 0.5], levels)
    image = Image("TASImage", read_image(None)(image_record(2, 0, body)))
    assert image.array[0, :, 0].tolist() == [142, 142, 0, 255, 28]


def test_an_image_whose_stops_coincide_steps_rather_than_dividing_by_nothing():
    levels = [[0, 0x4000, 0xFFFF]] * 4
    body = vector_image_body([[0.0, 2.0]], [0.0, 0.0, 1.0], levels)
    image = Image("TASImage", read_image(None)(image_record(2, 0, body)))
    assert image.array[0, :, 0].tolist() == [0x40, 0xFF]
