"""Classes a macro derives from ROOT's, behaving as ROOT's: their bases' members, overrides
called through a base pointer with the base's defaults, ``ClassDef``'s reflection - and the
C strings and ``TString`` characters such macros walk.
"""

from __future__ import annotations

import pytest

from pyrootsupport import fresh
from xrdroot.pycint import translate
from xrdroot.pycint.execute import run_source
from xrdroot.pycxx.runtime import base_member, inherited
from xrdroot.pycxx.runtime.namespace import DECLARED


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    declared = dict(DECLARED)
    yield from fresh(tmp_path)
    DECLARED.clear()
    DECLARED.update(declared)


NAMED = r"""
#include "TNamed.h"
#include "TAttLine.h"
class MyObj : public TNamed {
public:
   int fN;
   MyObj() : fN(0) {}
   MyObj(const char *n, const char *t) : TNamed(n, t), fN(3) {}
   void Print(Option_t *opt) const override {
      TNamed::Print(opt);
      printf("MyObj %s %d [%s]\n", GetName(), fN, opt);
   }
   void Rename() { SetName("renamed"); }
   static TClass *Mine() { return Class(); }
   ClassDefOverride(MyObj, 2);
};
class Lined : public TAttLine {
public:
   void Warn() { SetLineColor(4); Warning("Lined::Warn", "colour %d", GetLineColor()); }
};
void named() {
   MyObj o("a", "b");
   TObject *p = &o;
   p->Print();
   o.Rename();
   o.Print("x");
   printf("%s %s %d %d\n", p->ClassName(), p->IsA()->GetName(), MyObj::Class()->GetClassVersion(),
          o.InheritsFrom("TNamed"));
   printf("%s %d\n", MyObj::Class_Name(), MyObj::Mine() == o.IsA());
   Lined l;
   l.Warn();
}
"""


def test_a_class_derived_from_roots_has_its_bases_members_and_classdefs_reflection(capfd):
    run_source(NAMED, "named.C")
    out, err = capfd.readouterr()
    assert out.splitlines() == [
        "OBJ: MyObj\ta\tb",
        "MyObj a 3 []",
        "OBJ: MyObj\trenamed\tb",
        "MyObj renamed 3 [x]",
        "MyObj MyObj 2 1",
        "MyObj 1",
    ]
    assert "Warning in <Lined::Warn>: colour 4" in err


def test_a_base_member_is_found_on_the_object_and_else_in_the_library():
    class Base:
        def f(self):
            return "base"

        @staticmethod
        def s():
            return "static"

    class Derived(Base):
        def f(self):
            return "derived"

    host = type("Host", (), {"g": staticmethod(lambda: "host")})
    made = Derived()
    assert inherited(made, "f", host)() == "derived" and inherited(made, "g", host)() == "host"
    assert base_member(Base, "f", made)() == "base" and base_member(Base, "s", made)() == "static"


def test_a_qualified_call_of_what_is_not_a_base_stays_the_librarys():
    python = translate(
        'class A : public TNamed { void f() { TMath::Abs(1); Form("%d", 1); } };', "a.C"
    )
    assert "ROOT.TMath.Abs(1)" in python and "inherited(self, 'Form'" not in python


STRINGS = r"""
#include <string>
void strings(const char *s) {
   TString t = s;
   if (t[0] == 'u') t = "unsigned";
   std::string a;
   char c = 'x';
   a += c;
   t += c;
   t += 5;
   int n = 0;
   const char *walk = s;
   while (*walk) { ++n; walk++; }
   walk = s;
   walk += 2;
   int m = 0;
   const char *again = s;
   while (*again++) ++m;
   const char *front = s;
   char first = *++front;
   int numbers[3] = {1, 2, 3};
   int *at = numbers;
   at += 1;
   printf("%s %s %d %s %c %d %c %d\n", t.Data(), a.c_str(), n, walk, *s, m, first, *at);
}
"""


def test_a_tstring_indexes_to_a_char_and_a_c_string_is_walked_along(capfd):
    run_source(STRINGS, "strings.C", ("uvwxyz",))
    assert capfd.readouterr().out == "unsignedx5 x 6 wxyz u 6 v 2\n"


def test_a_character_stored_through_a_char_pointer_is_its_first_element():
    python = translate("void f(char *p) { *p = 'x'; }", "f.C")
    assert "p[0] = ord('x')" in python


GLOBAL_STORE = r"""
class Quiet : public TNamed {
public:
   void hush() { gErrorIgnoreLevel = 1; }
   int level() { return gErrorIgnoreLevel; }
};
void quiet() {
   Quiet q;
   q.hush();
   printf("%d\n", q.level());
}
"""


def test_a_library_global_is_written_where_the_library_keeps_it(capfd):
    """``gErrorIgnoreLevel = 1`` in a method of a class derived from ROOT's: the object has
    no such member, so C++ writes ROOT's global - where ``inherited(...)``, a call, cannot."""
    import xrdroot.pyroot as ROOT

    python = translate(GLOBAL_STORE, "quiet.C")
    assert "ROOT.gErrorIgnoreLevel = 1" in python
    assert "inherited(self, 'gErrorIgnoreLevel', ROOT) =" not in python
    kept = ROOT.gErrorIgnoreLevel
    try:
        run_source(GLOBAL_STORE, "quiet.C")
        assert capfd.readouterr().out == "1\n"
    finally:
        ROOT.gErrorIgnoreLevel = kept


def test_a_library_global_in_a_fixtures_test_body_is_written_the_same_way():
    """The gtest programs write ROOT's globals from a ``TEST`` body, which is a method of a
    class derived from ``testing::Test`` too."""
    python = translate(
        '#include "gtest/gtest.h"\nTEST(Suite, Name) { gErrorIgnoreLevel = 2; }', "t.cxx"
    )
    assert "ROOT.gErrorIgnoreLevel = 2" in python


def test_a_base_reached_twice_and_a_members_call_keep_their_meaning():
    python = translate(
        "class A : public TNamed {}; class B : public A {}; class C : public A {};\n"
        "class D : public B, public C { void f() { TNamed::Clear(); fHelper(); } };",
        "d.C",
    )
    assert "base_member(ROOT.TNamed, 'Clear', self)()" in python
    assert "self.fHelper()" in python


def test_a_default_answered_through_self_is_worked_out_as_the_body_starts():
    """``Bool_t dir = kIterForward`` in a class deriving from ROOT's ``TListIter``: the name
    is ROOT's, which the translation answers through ``inherited(self, ...)`` - and Python
    builds a default list as the class is defined, where there is no ``self``. C++ reads the
    default in the class's own scope, so the body stands in for the list."""
    python = translate(
        "class Walk : public TListIter {\n"
        "public:\n"
        "   Walk(TList *l, Bool_t dir = kIterForward) : TListIter(l, dir) {}\n"
        "};\n",
        "walk.C",
    )
    assert "dir_=UNGIVEN" in python or "dir=UNGIVEN" in python
    assert "is UNGIVEN" in python
    assert "inherited(self, 'kIterForward'" in python.split("is UNGIVEN")[1]


def test_an_override_takes_its_bases_defaults_where_it_has_none_of_its_own():
    import xrdroot.pyroot as ROOT

    class Mine(ROOT.TNamed):
        _cxx_layout_ = ("Mine", ("TNamed",), ())

        def Print(self, opt):
            return f"[{opt}]"

        def SetName(self, name="kept"):
            return name

        def Draw(self, a, b):
            return a + b

        def ls(self, option="own"):
            return option

    made = Mine()
    assert (made.Print(), made.SetName(), made.Draw(1, 2), made.ls()) == ("[]", "kept", 3, "own")
