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


def test_an_array_of_objects_is_destroyed_as_its_scope_ends(tmp_path, monkeypatch):
    """``A a[3]`` in a block destroys three times at the closing brace, as C++ does."""
    source = (
        "struct A { static int n; A(){n++;} ~A(){n--;} };\n"
        "int A::n = 0;\n"
        "int count() {\n{\nA a[3];\n}\nreturn A::n;\n}\n"
    )
    text = translate(source, "a.C")
    namespace: dict[str, object] = {}
    exec(compile(text, "a.C", "exec"), namespace)
    expect((namespace["count"](), 0), ("a._destruct()" in text, True))


def test_a_pointer_owning_an_array_of_objects_destroys_them_as_its_scope_ends():
    """``std::make_unique<A[]>(n)`` builds n of them and the pointer destroys them."""
    text = body(
        '{\nstd::make_unique<A[]>(3);\n}\n'
    ) if False else translate(
        "struct A { static int n; A(){n++;} ~A(){n--;} };\n"
        "int A::n = 0;\n"
        "int count() {\n{\nauto a = std::make_unique<A[]>(3u);\n}\nreturn A::n;\n}\n",
        "b.C",
    )
    namespace: dict[str, object] = {}
    exec(compile(text, "b.C", "exec"), namespace)
    expect((namespace["count"](), 0), ("make=A" in text, True))


def test_an_array_of_numbers_has_no_destructor_to_run():
    text = body("{\nint a[3];\na[0] = 1;\n}\nprintf(\"after\\n\");")
    expect(("_destruct()" in text, False))


def test_an_array_of_arrays_is_destroyed_row_by_row():
    """``A a[2][2]`` destroys four times: each row is an array of objects in its own right."""
    text = translate(
        "struct A { static int n; A(){n++;} ~A(){n--;} };\n"
        "int A::n = 0;\n"
        "int count() {\n{\nA a[2][2];\n}\nreturn A::n;\n}\n",
        "c.C",
    )
    namespace: dict[str, object] = {}
    exec(compile(text, "c.C", "exec"), namespace)
    expect((namespace["count"](), 0), (text.count("a._destruct()"), 1))


def test_letting_an_array_of_objects_go_twice_leaves_it_empty():
    from xrdroot.cppstd.objects import ObjectArray

    made: list = []

    class Held:
        def _destruct(self) -> None:
            made.append(1)

    held = ObjectArray([Held(), Held()])
    held._destruct()
    held._destruct()
    # an element that says nothing of a destructor - a number, or one never built - is left
    # alone, and an array already emptied has nothing to walk
    ObjectArray([None, 7])._destruct()
    ObjectArray()._destruct()
    expect((len(made), 2), (len(held), 0))


def test_a_pointer_owning_an_array_of_numbers_is_an_array_of_zeros():
    """``std::make_unique<int[]>(2)`` is two ints, which the caller writes into."""
    text = translate(
        "int total() {\nauto a = std::make_unique<int[]>(2u);\na[0] = 42;\na[1] = 7;\n"
        "return a[0] + a[1];\n}\n",
        "d.C",
    )
    namespace: dict[str, object] = {}
    exec(compile(text, "d.C", "exec"), namespace)
    expect((namespace["total"](), 49), ("array('int', 2)" in text, True))


def test_a_smart_pointer_holding_a_file_closes_it_as_the_scope_ends():
    """``std::unique_ptr<TFile> f{TFile::Open(...)}`` is the ``TFile`` itself here, so the file
    is written where the scope ends - not when the process does, which would leave a program
    that reads the file back in the same run reading a file that is not there yet."""
    for declaration in (
        'std::unique_ptr<TFile> f{TFile::Open("x.root", "RECREATE")};',
        'std::unique_ptr<TFile> f(TFile::Open("x.root", "RECREATE"));',
        'auto f = std::unique_ptr<TFile>(TFile::Open("x.root", "RECREATE"));',
        'std::shared_ptr<TFile> f{TFile::Open("x.root", "RECREATE")};',
    ):
        text = body("{\n" + declaration + "\n}\nprintf(\"after\\n\");")
        assert "finally:" in text, declaration
        assert "f._destruct()" in text, declaration
