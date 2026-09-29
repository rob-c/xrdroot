"""``TVectorD``, ``TArrayD``, ``TImage`` and ``std::array``, as the FITS macros use them."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from xrdroot.canvas.image import paint_image
from xrdroot.errors import UnsupportedFeatureError


def test_a_vector_is_made_every_way_root_makes_one() -> None:
    made = [ROOT.TVectorD(), ROOT.TVectorD(3), ROOT.TVectorD([1, 2]), ROOT.TVectorD(2, [5, 6, 7]),
            ROOT.TVectorD(1, 3), ROOT.TVectorD(1, 2, [8, 9, 10])]  # fmt: skip
    copy = ROOT.TVectorD(made[5])
    copy[2] = 4.0
    copy.__setcall__(1, 3.0)
    expect(([v.GetNoElements() for v in made], [0, 3, 2, 2, 3, 2]),
           ((made[4].GetLwb(), made[4].GetUpb()), (1, 3)), (made[5](2), 9.0),
           (list(copy), [3.0, 4.0]), (list(made[5]), [8.0, 9.0]), (made[3].Sum(), 11.0),
           (made[2].Norm2Sqr(), 5.0), (len(made[1]), 3), (made[2].GetNrows(), 2),
           (np.asarray(made[3]).tolist(), [5.0, 6.0]))  # fmt: skip


def test_a_vector_prints_as_root_prints_one(capsys: Any) -> None:
    ROOT.TVectorD(1, 2, [0.5, 2]).Print()
    assert capsys.readouterr().out == (
        "\nVector (2)  is as follows\n\n     |        1  |\n------------------\n"
        "   1 |0.5 \n   2 |2 \n\n")  # fmt: skip


def test_an_array_of_doubles_is_read_and_set() -> None:
    array, sized, empty = ROOT.TArrayD([1, 2, 3]), ROOT.TArrayD(2, [4, 5, 6]), ROOT.TArrayD(2)
    array.SetAt(7.0, 0)
    expect((array.At(0), 7.0), (array[1], 2.0), (sized.GetAt(1), 5.0), (sized.GetSize(), 2),
           (len(empty), 2), (np.asarray(empty).tolist(), [0.0, 0.0]),
           (array.GetArray().tolist(), [7.0, 2.0, 3.0]))  # fmt: skip


def test_a_picture_is_made_from_values_and_a_palette() -> None:
    palette = ROOT.TImagePalette(2)
    palette.fPoints = [0.0, 1.0]
    palette.fColorRed, palette.fColorGreen = [0, 0xFFFF], [0, 0]
    palette.fColorBlue, palette.fColorAlpha = [0xFFFF, 0], [0xFFFF, 0xFFFF]
    picture = ROOT.TImage.Create()
    empty = picture.IsValid()
    picture.SetImage(ROOT.TArrayD([0, 1, 2, 3]), 2, palette)
    expect((empty, False), (picture.IsValid(), True),
           (picture.rgba[1, 0].tolist(), [0, 0, 255, 255]),
           (picture.rgba[0, 1].tolist(), [255, 0, 0, 255]))  # fmt: skip


def test_what_a_picture_cannot_do_yet_is_refused() -> None:
    with pytest.raises(UnsupportedFeatureError, match="TImage::Open"):
        ROOT.TImage.Open("x.png")
    with pytest.raises(UnsupportedFeatureError, match="made to a size"):
        ROOT.TASImage(10, 10)
    with pytest.raises(UnsupportedFeatureError, match="TASImage::WriteImage"):
        ROOT.TImage.Create().WriteImage("x.png")
    with pytest.raises(AttributeError):
        ROOT.TImage.Create().Nothing  # noqa: B018


def test_an_empty_picture_paints_nothing() -> None:
    class Prim:
        def get(self, key: str) -> Any:
            return np.zeros((0, 0, 4))

    assert paint_image(None, Prim(), "") is None


def test_std_array_is_a_vector_of_its_size_filled_from_its_list() -> None:
    numbers = ROOT.std.array["double", 3]([1.5, 2.5])
    pointers = ROOT.std.array["std::unique_ptr<TVectorD>", 2]([ROOT.TVectorD(1)])
    expect((list(numbers), [1.5, 2.5, 0.0]), (pointers[1], None),
           (pointers[0].GetNoElements(), 1), (repr(ROOT.std.array), "<std.array template>"))


def test_a_name_given_a_smart_pointer_is_a_cast_to_the_pointer(tmp_path: Any,
                                                              capsys: Any) -> None:
    from xrdroot.cint.execute import run

    macro = tmp_path / "alias.C"
    macro.write_text("using Up = std::unique_ptr<TVectorD>;\ntypedef TVectorD *Vp;\n"
                     "void alias() {\n  auto v = Up(new TVectorD(3));\n"
                     "  auto w = Vp(new TVectorD(2));\n"
                     '  printf("%d %d\\n", v->GetNoElements(), w->GetNoElements());\n}\n')
    run(str(macro), use_cache=False)
    assert capsys.readouterr().out == "3 2\n"
