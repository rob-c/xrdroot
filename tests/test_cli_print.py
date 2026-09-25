"""``xrdroot print``: objects drawn into picture files, or in characters without one.

matplotlib is not a dependency, so the drawing itself is stood in for - what
is checked is that each object is asked to draw and its figure is saved where
it was told - and one test at the end draws for real when matplotlib is there.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

from clisupport import FakeMatplotlib
from xrdroot import Graph, Histogram
from xrdroot.cli import main
from xrdroot.cli.render import options, render
from xrdroot.errors import UnsupportedFeatureError

DATA = pathlib.Path(__file__).parent / "data"


def data(name: str) -> str:
    return str(DATA / name)


@pytest.fixture
def fake_matplotlib(monkeypatch):
    return FakeMatplotlib().install(monkeypatch)


@pytest.fixture
def saved(monkeypatch, fake_matplotlib):
    """Where every histogram and graph was saved, and the style it was drawn in."""
    found: list = []
    for kind in (Histogram, Graph):
        monkeypatch.setattr(kind, "plot", lambda self, **style: fake_matplotlib.axes(found, style))
    return found


def printed(capsys, *argv: str) -> tuple[int, list[str]]:
    status = main(["print", *argv])
    return status, capsys.readouterr().out.splitlines()


def test_one_object_is_drawn_into_the_file_named(capsys, tmp_path, saved, fake_matplotlib):
    target = tmp_path / "h1d.png"
    status, out = printed(capsys, f"{data('gauss-h1.root')}:h1d", "-o", str(target))
    assert status == 0
    assert out == [f"wrote {target}"]
    assert saved == [(str(target), {})]
    assert fake_matplotlib.backends == ["Agg"]
    assert len(fake_matplotlib.closed) == 1


def test_options_are_handed_to_plot(capsys, tmp_path, saved):
    target = tmp_path / "g.pdf"
    printed(
        capsys,
        data("graphs.root"),
        "-k",
        "tg",
        "-o",
        str(target),
        "--option",
        "color=red",
        "--option",
        "ms=3",
    )
    assert saved == [(str(target), {"color": "red", "ms": 3})]


def test_a_directory_draws_everything_in_it_that_draws(capsys, tmp_path, saved):
    target = tmp_path / "all.svg"
    status, out = printed(capsys, data("dirs-6.14.00.root"), "-o", str(target))
    assert status == 0
    assert out == [f"wrote {tmp_path / 'all_dir1_dir11_h1.svg'}"]
    _, inner = printed(capsys, f"{data('dirs-6.14.00.root')}:dir1", "-o", str(target))
    assert inner == [f"wrote {tmp_path / 'all_dir1_dir11_h1.svg'}"]


def test_what_does_not_draw_is_skipped_with_the_reason(capsys, tmp_path, saved):
    _, out = printed(capsys, data("tformula.root"), "-o", str(tmp_path / "f.png"))
    assert out[0] == (
        "skipped func1 (TF1): a TF1 has no picture to draw; "
        "histograms, graphs, profiles, stacks and canvases do"
    )
    assert len(out) == 6


def test_a_directory_without_a_picture_file_is_refused(capsys):
    assert main(["print", data("graphs.root")]) == 2
    assert "say where with -o" in capsys.readouterr().err


def test_without_a_picture_file_one_object_is_drawn_in_characters(capsys):
    status, out = printed(capsys, f"{data('graphs.root')}:tg")
    assert status == 0
    assert any("*" in line for line in out)


def test_something_with_no_picture_in_characters_asks_for_a_file(capsys):
    assert main(["print", f"{data('tformula.root')}:func1"]) == 2
    assert "a TF1 has no picture in characters" in capsys.readouterr().err


def test_a_picture_file_must_end_in_a_picture_format(capsys):
    assert main(["print", f"{data('graphs.root')}:tg", "-o", "tg.txt"]) == 2
    assert "tg.txt does not end in a picture format" in capsys.readouterr().err


def test_options_are_literals_when_they_are_and_strings_otherwise():
    assert options(None) == {}
    assert options(["a=1", "b=[1, 2]", "c=red", "d=(1"]) == {
        "a": 1,
        "b": [1, 2],
        "c": "red",
        "d": "(1",
    }
    with pytest.raises(ValueError, match="an option is name=value, and 'bare' is not"):
        options(["bare"])
    with pytest.raises(ValueError):
        options(["=1"])


def test_something_that_saves_itself_is_asked_to(tmp_path):
    seen = []
    canvas = types.SimpleNamespace(save_as=lambda path, **style: seen.append((path, style)))
    assert render(canvas, str(tmp_path / "c.png"), {"dpi": 100}) == tmp_path / "c.png"
    assert seen == [(str(tmp_path / "c.png"), {"dpi": 100})]


def test_a_canvas_read_as_members_needs_the_canvas_module(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "xrdroot.canvas", None)
    with pytest.raises(UnsupportedFeatureError, match=r"drawing a canvas needs xrdroot\.canvas"):
        render({"fName": "c1"}, str(tmp_path / "c.png"), classname="TCanvas")


def test_a_canvas_module_without_a_renderer_is_refused_the_same_way(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "xrdroot.canvas", types.ModuleType("xrdroot.canvas"))
    with pytest.raises(UnsupportedFeatureError, match=r"needs xrdroot\.canvas"):
        render({"fName": "c1"}, str(tmp_path / "c.png"), classname="TPad")


def test_a_canvas_read_as_members_is_drawn_by_the_canvas_module(monkeypatch, tmp_path):
    seen = []
    module = types.ModuleType("xrdroot.canvas")
    module.render = lambda value, path, **style: seen.append((value, path, style))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xrdroot.canvas", module)
    target = str(tmp_path / "c.png")
    render({"fName": "c1"}, target, {"dpi": 50}, "TCanvas")
    assert seen == [({"fName": "c1"}, target, {"dpi": 50})]


def test_something_with_no_picture_is_refused_by_its_class(tmp_path):
    with pytest.raises(UnsupportedFeatureError, match="a dict has no picture to draw"):
        render({}, str(tmp_path / "x.png"))


def test_without_matplotlib_plot_refuses_naming_both_ways_out(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    h = Histogram.book("h", (4, 0.0, 1.0))
    with pytest.raises(UnsupportedFeatureError, match="pip install matplotlib"):
        render(h, str(tmp_path / "h.png"))


def test_the_real_matplotlib_writes_a_real_picture(capsys, tmp_path):
    pytest.importorskip("matplotlib")
    target = tmp_path / "h1d.png"
    assert main(["print", f"{data('gauss-h1.root')}:h1d", "-o", str(target)]) == 0
    assert target.read_bytes().startswith(b"\x89PNG")
