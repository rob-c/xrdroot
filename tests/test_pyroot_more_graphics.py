"""Smaller graphics classes the tutorials use: ``TPavesText`` and its kin.

``TPavesText`` is ROOT's ``archi.C`` and ``framework.C`` boxes: a pave of
text on ``npaves - 1`` paves stacked three borders apart behind it.
"""

from __future__ import annotations

from typing import Any

import xrdroot.pyroot as ROOT
from geomsupport import geometry_session  # noqa: F401
from pyrootsupport import expect


def stacked_boxes(tmp_path: Any, option: str) -> int:
    """How many box outlines a pad with one ``TPavesText`` of ``option`` is drawn with."""
    canvas = ROOT.TCanvas("c", "c", 200, 200)
    paves = ROOT.TPavesText(0.2, 0.2, 0.8, 0.8, 4, option)
    paves.AddText("stacked")
    paves.Draw()
    canvas.SaveAs(str(tmp_path / f"paves-{option}.png"))
    return paves.GetNpaves()


def test_a_stack_of_paves_is_drawn_behind_its_text(tmp_path: Any) -> None:
    for option in ("br", "tl"):
        assert stacked_boxes(tmp_path, option) == 4
    paves = ROOT.TPavesText(0, 0, 1, 1)
    paves.SetNpaves(2)
    expect((paves.GetNpaves(), 2), (paves.members["fBorderSize"], 1),
           (ROOT.TPavesText().ClassName(), "TPavesText"))  # fmt: skip


def test_a_colour_with_an_opacity_is_one_of_its_own_found_again_by_both() -> None:
    see_through = ROOT.TColor.GetColor(0.25, 0.25, 0.25, 0.55)
    again = ROOT.TColor.GetColor(0.25, 0.25, 0.25, 0.55)
    opaque = ROOT.TColor.GetColor(0.25, 0.25, 0.25, 1.0)
    expect((again, see_through), (opaque != see_through, True),
           (ROOT.gROOT.GetColor(see_through).GetAlpha(), 0.55))  # fmt: skip


def test_a_canvas_says_it_is_drawn_without_opengl_whatever_the_style_prefers() -> None:
    ROOT.gStyle.SetCanvasPreferGL(True)
    canvas = ROOT.TCanvas("c", "c", 100, 100)
    expect((ROOT.gStyle.GetCanvasPreferGL(), True), (canvas.UseGL(), False),
           (canvas.IsWeb(), False))  # fmt: skip
    ROOT.gStyle.SetCanvasPreferGL(False)


def test_groot_finds_a_colour_made_or_roots_own_and_none_for_one_not_there() -> None:
    red = ROOT.gROOT.GetColor(2)
    expect((red.GetRed(), 1.0), (red.GetNumber(), 2), (ROOT.gROOT.GetColor(2), red),
           (ROOT.gROOT.GetColor(99999), None))  # fmt: skip


#: A macro whose TExecs run its own functions, as ``multipalette.C`` and ``gr202`` do.
EXEC_MACRO = """
void mark() {
   TLatex l;
   l.SetTextColor(kBlue);
   l.PaintText(0.5, 0.5, "painted");
   l.PaintTextNDC(0.1, 0.1, "corner");
   l.PaintLatex(0.2, 0.2, 30, 0.05, "#alpha");
}
void texec_macro() {
   TCanvas *c = new TCanvas("c", "c", 200, 200);
   TGraph *g = new TGraph(2);
   g->SetPoint(0, 0, 0);
   g->SetPoint(1, 1, 1);
   g->GetListOfFunctions()->Add(new TExec("hung", "mark();"));
   g->Draw("AL");
   TExec *ex = new TExec("ex", "mark();");
   ex->Draw();
   c->AddExec("dynamic", "mark();");
}
"""


def test_a_texec_runs_the_macro_s_own_functions_each_time_its_pad_is_painted(tmp_path: Any) -> None:
    from xrdroot.cint.execute import run

    (tmp_path / "texec_macro.C").write_text(EXEC_MACRO)
    run(tmp_path / "texec_macro.C", use_cache=False)
    canvas = ROOT.gPad.GetCanvas()
    canvas.Update()
    canvas.Update()  # painting again replaces what the last painting made
    texts = [obj for obj, _ in canvas.primitives if obj.ClassName() == "TLatex"]
    execs = canvas.GetListOfExecs()
    canvas.DeleteExec("dynamic")
    expect((len(texts), 6), (texts[0].GetTitle(), "painted"), (execs[0].GetAction(), "mark();"),
           (canvas.GetListOfExecs(), []), (canvas.GetListOfPrimitives().FindObject("ex").GetName(),
                                           "ex"))  # fmt: skip


def test_text_painted_with_no_texec_running_is_drawn_in_the_pad() -> None:
    ROOT.TCanvas("c", "c", 100, 100)
    ROOT.TText().PaintText(0.1, 0.1, "here")
    execute = ROOT.TExec("e", "")
    execute.SetAction("1;")
    expect((ROOT.gPad.primitives[0][0].GetTitle(), "here"), (execute.Exec(), []))
