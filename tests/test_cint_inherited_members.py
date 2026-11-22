"""A class a macro declares, deriving from ROOT's, holds its base's members as its own.

C++ gives an object of ``class Mine : public TNamed`` one set of ``TNamed``'s data
members, the base subobject's, which the base's own methods and the derived class's
both read and write. The translation derives the Python class from ROOT's, runs the
base's constructor before the class's own members are set, and writes a base member
as the object's attribute (``self.fTitle = 'x'``) rather than as a name of the
macro module - so a write in a constructor is the write ``GetTitle`` later reads.
A member the base does not declare is the macro class's own, unchanged.
"""

from __future__ import annotations

import pytest

from pyrootsupport import fresh
from xrdroot.pycint import translate
from xrdroot.pycint.execute import run_source
from xrdroot.pycxx.runtime.namespace import DECLARED

NAMED = r"""
#include "TNamed.h"
class Base : public TNamed {
public:
   int fCount;
   Base(const char *n) : TNamed(n, "basetitle"), fCount(1) {}
   void Bump() { fCount++; }
};
class Mid : public Base {
public:
   Mid(const char *n) : Base(n) { fCount = 2; }
};
class Mine : public Mid {
public:
   Mine(const char *n) : Mid(n), fOwn(7) { fCount = 5; }
   int fOwn;
   int Total() const { return fCount + fOwn; }
};
void named() {
   Mine m("hello");
   m.Bump();
   printf("%d %d %d\n", m.fCount, m.fOwn, m.Total());
   printf("[%s] [%s] %s\n", m.GetName(), m.GetTitle(), m.ClassName());
   TNamed *n = &m;
   printf("[%s] %d\n", n->GetName(), n->InheritsFrom("TNamed"));
}
"""


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    declared = dict(DECLARED)
    yield from fresh(tmp_path)
    DECLARED.clear()
    DECLARED.update(declared)


def test_a_member_inherited_through_two_macro_classes_is_the_objects_one(capfd):
    """``Base``'s ``fCount``, set by ``Base``'s constructor and by both classes below it,
    is one member: the last write wins, and the base's own method changes the same one."""
    run_source(NAMED, "named.C")
    assert capfd.readouterr().out.splitlines() == [
        "6 7 13",
        "[hello] [basetitle] Mine",
        "[hello] 1",
    ]


def test_a_roots_base_members_are_written_on_the_object_as_cpp_writes_them():
    """The brief's macro, before anything of ROOT's class is asked: the base is constructed
    first, and ``fTitle``/``fName`` are the object's attributes - not names of the module,
    which nothing would read back."""
    python = translate(
        '#include "TNamed.h"\nclass Mine : public TNamed { public:\n'
        '   Mine(const char *n) { fName.Form("%s_%d", n, 7); fTitle = "x"; } };',
        "mine.C",
    )
    lines = [line.strip() for line in python.splitlines() if "TNamed.__init__" in line
             or "fName" in line or "fTitle" in line]  # fmt: skip
    assert lines == [
        "ROOT.TNamed.__init__(self)",
        "self.fName.Form('%s_%d', n, 7)",
        "self.fTitle = 'x'",
    ]


def test_a_member_the_base_does_not_have_is_the_macro_classs_own(capfd):
    """``fOwn`` is declared by ``Mine`` alone: it is set from the constructor's initialiser
    list and read back as the class's own member, with the base's members untouched."""
    run_source(
        '#include "TNamed.h"\n'
        "class Mine : public TNamed { public:\n"
        "   Mine() : fOwn(3) { fTitle = \"t\"; }\n"
        "   int fOwn; int Own() const { return fOwn; } };\n"
        "void own() { Mine m; printf(\"%d %d\\n\", m.Own(), m.fOwn); }\n",
        "own.C",
    )
    assert capfd.readouterr().out == "3 3\n"


def test_a_base_member_reached_from_an_out_of_line_method_is_the_objects(capfd):
    """``void Mine::Dump() const`` is still ``Mine``'s method: its ``fOwn`` is the object's
    member, and the base's ``GetName`` is the one the object has."""
    python = translate(
        '#include "TNamed.h"\nclass Mine : public TNamed { public:\n'
        "   Mine() { fOwn = 4; }\n   int fOwn; void Dump() const; };\n"
        'void Mine::Dump() const { printf("%d [%s]\\n", fOwn, GetName()); }\n'
        "void out() { Mine m; m.Dump(); }\n",
        "out.C",
    )
    assert "self.fOwn = 4" in python
    run_source(
        '#include "TNamed.h"\nclass Mine : public TNamed { public:\n'
        "   Mine() { fOwn = 4; SetName(\"named\"); }\n   int fOwn; void Dump() const; };\n"
        'void Mine::Dump() const { printf("%d [%s]\\n", fOwn, GetName()); }\n'
        "void out() { Mine m; m.Dump(); }\n",
        "out.C",
    )
    assert capfd.readouterr().out == "4 [named]\n"


STATICS = r"""
class Base {
public:
   TString fLabel;
   Base() { fLabel = "kept"; }
   const char *Label() const { return fLabel.Data(); }
};
class Mine : public Base {
public:
   static int fSeen;
   void Touch() { fSeen = 2; fLabel = "touched"; }
   int Stored() const { return fSeen; }
};
int Mine::fSeen = 0;
void statics() {
   Mine m;
   m.Touch();
   printf("%d %d [%s]\n", m.Stored(), Mine::fSeen, m.Label());
}
"""


def test_a_static_of_the_class_does_not_take_a_base_member_of_the_same_object(capfd):
    """A class with a static data member writes instances through its own ``__setattr__``,
    which looks for the static first and leaves every other name to the object - so the
    base's ``fLabel`` a method writes is the one the base's own ``Label`` reads."""
    run_source(STATICS, "statics.C")
    assert capfd.readouterr().out == "2 2 [touched]\n"


def test_a_roots_base_member_written_beside_a_static_is_still_the_objects():
    """The same write where the base is ROOT's: ``fTitle`` is the object's attribute even
    though the class writes its instance through ``__setattr__`` for the static's sake."""
    python = translate(
        '#include "TNamed.h"\nclass Mine : public TNamed { public:\n'
        "   static int fSeen;\n   void Touch() { fSeen = 2; fTitle = \"x\"; } };\n"
        "int Mine::fSeen = 0;\n",
        "statics.C",
    )
    assert "self.fTitle = 'x'" in python and "Mine.fSeen = 2" in python
    assert "_cxx_statics_ = ['fSeen']" in python
