"""A local whose scope end C++ acts on, where the same is reached by other spellings.

``TFile f("x.root", "RECREATE")`` in a block closes the file as the block ends; so does
``auto p = T::Make()`` when ``T::Make`` is a *qualified static* call and the dialect says what
it gives back - which is how ``RNTupleWriter::Recreate``'s writer writes its file where the
macro's block ends. ``tests/test_pyroot_ntuple_destruct.py`` checks what that does.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from xrdroot.pycint import translate


def body(source: str) -> str:
    text = translate("void t() {\n" + source + "\n}\n", "t.C")
    compile(text, "t.C", "exec")
    return text


def test_a_block_scoped_file_is_closed_as_the_block_ends():
    text = body('{\nTFile f("x.root", "RECREATE");\n}\nprintf("after\\n");')
    expect(
        ("try:" in text, True),
        ("finally:" in text, True),
        ("f._destruct()" in text, True),
    )


def test_a_writer_a_qualified_static_call_gives_back_is_let_go_as_its_block_ends():
    text = body(
        '{\nauto w = RNTupleWriter::Recreate(RNTupleModel::Create(), "rs", "x.root");\n'
        "w->Fill();\n}\n"
    )
    expect(
        ("w._destruct()" in text, True),
        # the call is still made once, and the writer is not a value the translator copies
        (text.count("RNTupleWriter.Recreate"), 1),
    )


def test_the_translation_of_that_block_runs_and_closes_the_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = (
        "void t() {\n{\n"
        "auto model = RNTupleModel::Create();\n"
        'model->MakeField<std::int32_t>("value");\n'
        'auto w = RNTupleWriter::Recreate(std::move(model), "rs", "held.root");\n'
        "w->Fill();\n}\n}\n"
    )
    text = translate(source, "t.C")
    namespace: dict[str, object] = {}
    exec(compile(text, "t.C", "exec"), namespace)
    namespace["t"]()
    expect(((tmp_path / "held.root").stat().st_size > 0, True))


def test_calc_reads_a_value_out_without_printing_it(capsys):
    """ROOT's ``Calc`` is silent where ``ProcessLine`` echoes the value it comes to."""
    ROOT.gInterpreter.Calc("1+2")
    expect((capsys.readouterr().out, ""))
    ROOT.gInterpreter.ProcessLine("2+3")
    expect((capsys.readouterr().out, "(int) 5\n"))


@pytest.mark.parametrize("expression", ["1+2", "(Double_t) 3 / 2"])
def test_calc_gives_the_value_back(expression):
    expect((ROOT.gInterpreter.Calc(expression), eval(expression.replace("(Double_t) ", ""))))
