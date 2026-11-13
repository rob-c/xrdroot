"""``T a[];``: an array sized by its initialiser, or by the definition its declaration names."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.pycint import Refusal, translate
from xrdroot.pycint.execute import run_source


def output(capsys: pytest.CaptureFixture[str], source: str) -> str:
    run_source(source, "t.C", root=fake())
    return capsys.readouterr().out


def test_a_static_members_size_is_its_out_of_class_definitions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    class Hand { public: static float fgX[]; static const char fgName[]; };
    float Hand::fgX[] = { -0.05, 0, 0.05 };
    const char Hand::fgName[] = "hand";
    void t() { printf("%d %g %s %d\\n", (int)(sizeof(Hand::fgX) / sizeof(float)), Hand::fgX[2],
                      Hand::fgName, (int)sizeof(Hand::fgName)); }
    """
    assert "fgX = array('float', 3, [-0.05, 0, 0.05])" in translate(source, "t.C")
    assert output(capsys, source) == "3 0.05 hand 5\n"


def test_an_extern_array_is_the_one_its_definition_makes_before_or_after_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    extern int gK[];
    int gK[] = {1, 2, 3, 4};
    int gJ[] = {5, 6};
    extern int gJ[];
    extern int gElsewhere[];
    extern int gCount;
    extern int gSet = 3;
    int size() { return sizeof(gK) / sizeof(int) + 10 * (sizeof(gJ) / sizeof(int)); }
    void t() { printf("%d %d %d %d\\n", size(), gK[3], gCount, gSet); }
    """
    text = translate(source, "t.C")
    assert text.count("gK = ") == 1 and text.count("gJ = ") == 1
    assert "gElsewhere" not in text
    assert "gCount = 0" in text
    assert output(capsys, source) == "24 4 0 3\n"


def test_an_array_with_no_size_and_nothing_to_size_it_is_still_refused() -> None:
    with pytest.raises(Refusal, match=r"the array fgX\[\] with no size to give it"):
        translate("class Hand { public: static float fgX[]; };", "t.C")
