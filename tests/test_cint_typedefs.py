"""``typedef struct [Tag] {...} T;``: a class defined in a typedef, named by it."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.pycint import Refusal, translate
from xrdroot.pycint.execute import run_source


def test_an_unnamed_struct_takes_the_typedefs_name(capsys: pytest.CaptureFixture[str]) -> None:
    source = """
    typedef struct { float vect[3]; int nmec; } Gctrak_t;
    void t() { Gctrak_t g; g.nmec = 4; g.vect[1] = 2.5; printf("%d %g\\n", g.nmec, g.vect[1]); }
    """
    assert "class Gctrak_t:" in translate(source, "t.C")
    run_source(source, "t.C", root=fake())
    assert capsys.readouterr().out == "4 2.5\n"


def test_a_tagged_struct_is_its_tag_and_every_typedef_name_is_a_name_for_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    void t() {
      typedef struct _Chunk { unsigned short idnum; unsigned int len; } Chunk, *PChunk;
      Chunk c; c.len = 7; PChunk p = &c;
      printf("%u\\n", p->len);
    }
    """
    text = translate(source, "t.C")
    assert "class _Chunk:" in text and "c = _Chunk()" in text
    run_source(source, "t.C", root=fake())
    assert capsys.readouterr().out == "7\n"


def test_roots_opaque_handles_are_pointers_shared_not_copied_and_passed_by_reference() -> None:
    python = translate(
        """
    void walk(TXMLEngine &xml, XMLNodePointer_t node) {
       XMLNodePointer_t child = xml.GetChild(node);
       XMLNodePointer_t same = child;
       xml.ShiftToNext(child);
    }
    """
    )
    assert "value_copy" not in python
    assert "child = Cell(xml.GetChild(node))" in python and "xml.ShiftToNext(child)" in python


def test_a_using_inside_a_function_names_the_type_its_call_makes() -> None:
    """``using cv_milli_t = std::chrono::duration<double, std::milli>;`` in a body, then
    ``cv_milli_t(end - start)``: C++ makes one of the type the alias stands for, and a template
    instance's constructor is the runtime's own call of that template. ROOT's cross-validation
    test times its own run exactly so, and both of its programs passed only once this was here.
    """
    source = """
    #include <chrono>
    double t() {
      using cv_clock_t = std::chrono::high_resolution_clock;
      using cv_milli_t = std::chrono::duration<double, std::milli>;
      auto StartTime = cv_clock_t::now();
      auto EndTime = cv_clock_t::now();
      return cv_milli_t(EndTime - StartTime).count() / 1000.0;
    }
    """
    translated = translate(source, "t.C")
    assert "chrono.duration['double', 'std::milli'](EndTime - StartTime).count()" in translated
    assert "ROOT.cv_milli_t" not in translated


def test_a_name_two_functions_give_two_types_is_refused_by_name() -> None:
    """One alias table serves a program's functions, so two of them giving a name two types is
    refused with what to do - rather than the last one quietly winning in both."""
    source = """
    void f() { using Length = std::chrono::duration<double, std::milli>; Length l(1); (void)l; }
    void g() { using Length = std::chrono::duration<double, std::ratio<1>>; Length l(1); (void)l; }
    """
    with pytest.raises(Refusal, match="stands for two types in one program"):
        translate(source, "t.C")


def test_a_name_two_functions_spell_the_same_way_is_one_alias() -> None:
    """ROOT's own TMVA ConvNet tests declare ``using Net_t = TDeepNet<Architecture>;`` in each of
    their functions: the name, the type and the spelling are the same, so one table holds one
    alias - what is refused is one name given two *spellings*."""
    source = """
    template <typename A> void f() { using Net_t = TDeepNet<A>; Net_t n; (void)n; }
    template <typename A> void g() { using Net_t = TDeepNet<A>; Net_t n; (void)n; }
    """
    translated = translate(source, "t.C")
    assert translated.count("TDeepNet") >= 2
