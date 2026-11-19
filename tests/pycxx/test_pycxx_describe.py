"""A class described for ``.Class``: its size, its place, its bases and its signatures.

ROOT's description comes out of a dictionary; this one is read back off the C++
the session translated, so what it can describe is what a declaration wrote.
"""

from __future__ import annotations

from pathlib import Path

from xrdroot.pycxx.describe import described, program_of

#: ``roottest/cling/operator/equal.C``'s classes: a private assignment operator that takes
#: an ``int``, a const member, and a base whose member is inherited.
EQUAL = """\
class privateOp2 {
private:
   privateOp2& operator=(const int&);
public:
   int c;
};

class withConstMember {
public:
   withConstMember(int val = 0) : a(val) {};
   const int a;
};

class base {
public:
   void f();
};

class withHiddenRef : public base {
   withHiddenRef &operator=(const withHiddenRef&);
   int a;
public:
   ~withHiddenRef();
   void Print(Option_t *option) const;
};
"""

#: A member template, as ``roottest/cling/templateMembers/operatorEqual.C`` declares one.
TEMPLATES = """\
class Holder {
public:
    template<class X> Holder& operator+= (const Holder&) { return *this; };
};
"""


def test_a_private_assignment_operator_is_described_as_its_declaration_spelled_it() -> None:
    program = program_of(EQUAL, "equal.C")
    found = described(program, "privateOp2")
    assert found is not None
    assert found.name == "privateOp2"
    assert found.size == 4  # one int, as clang lays it out
    assert found.file == "equal.C"
    assert found.line == 1
    assert [(each.access, each.signature) for each in found.methods] == [
        ("private", "privateOp2 &operator=(const int &)")
    ]


def test_a_constructor_and_a_const_member_function_say_so() -> None:
    program = program_of(EQUAL, "equal.C")
    found = described(program, "withConstMember")
    assert found is not None
    assert [(each.access, each.signature) for each in found.methods] == [
        ("public", "withConstMember(int)")
    ]
    hidden = described(program, "withHiddenRef")
    assert hidden is not None
    assert [(each.access, each.signature) for each in hidden.methods] == [
        ("private", "withHiddenRef &operator=(const withHiddenRef &)"),
        ("public", "~withHiddenRef()"),
        ("public", "void Print(Option_t *) const"),
    ]
    assert [each.name for each in hidden.bases] == ["base"]


def test_a_member_of_a_class_no_layout_is_known_for_leaves_the_size_unknown() -> None:
    program = program_of("class E { public: TGraph g; };\n", "E.hh")
    found = described(program, "E")
    assert found is not None and found.size is None


def test_a_class_nothing_declares_is_not_described() -> None:
    program = program_of(EQUAL, "equal.C")
    assert described(program, "Nowhere") is None


def test_a_member_template_is_described_with_its_own_parameters() -> None:
    program = program_of(TEMPLATES, "operatorEqual.C")
    found = described(program, "Holder")
    assert found is not None
    assert [each.template for each in found.methods] == ["template <class X> "]
    assert found.methods[0].signature == "Holder &operator+=(const Holder &)"


def test_the_directories_a_source_reads_its_headers_from_are_given_it(tmp_path: Path) -> None:
    (tmp_path / "a.h").write_text("class A { public: void f(); };\n")
    program = program_of('#include "a.h"\n', str(tmp_path / "a.cxx"), [tmp_path])
    assert described(program, "A") is not None
