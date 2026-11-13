"""``typeid`` and ``std::type_info``: identities, comparisons, and the names compilers mangle."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from cintfake import fake
from xrdroot.pycint import Refusal, translate
from xrdroot.pycint.execute import run_source
from xrdroot.pycint.runtime import bad_typeid, type_info, typeid_dynamic, typeid_of
from xrdroot.pycxx.runtime.typeinfo import _base36, mangle

CLASSES = """
#include <typeinfo>
#include <map>
class A {};
class B { public: virtual ~B() {} };
class C : public B {};
class D : public A {};
"""


def output(capsys: pytest.CaptureFixture[str], source: str, root: object = None) -> str:
    run_source(source, "t.C", root=fake() if root is None else root)
    return capsys.readouterr().out


@pytest.mark.parametrize(
    ("spelled", "mangled"),
    [
        ("int", "i"),
        ("unsigned int", "j"),
        ("unsigned", "j"),
        ("signed", "i"),
        ("short int", "s"),
        ("unsigned short", "t"),
        ("long", "l"),
        ("unsigned long", "m"),
        ("long long", "x"),
        ("unsigned long long", "y"),
        ("char", "c"),
        ("signed char", "a"),
        ("unsigned char", "h"),
        ("bool", "b"),
        ("float", "f"),
        ("double", "d"),
        ("long double", "e"),
        ("void*", "Pv"),
        ("const char*", "PKc"),
        ("char const*", "PKc"),
        ("volatile int", "i"),
        ("const int &", "i"),
        ("int* const", "Pi"),
        ("int[3]", "A3_i"),
        ("int[2][3]", "A2_A3_i"),
        ("std::nullptr_t", "Dn"),
        ("A", "1A"),
        ("A**", "PP1A"),
        ("TH1F", "4TH1F"),
        ("ns::A::B", "N2ns1A1BE"),
        ("ROOT::Math::Cartesian3D<double>", "N4ROOT4Math11Cartesian3DIdEE"),
        (
            "ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>, "
            "ROOT::Math::DefaultCoordinateSystemTag>",
            "N4ROOT4Math20DisplacementVector3DINS0_11Cartesian3DIdEENS0_26DefaultCoordinateSystemTagEEE",
        ),
        ("ns::A<ns::B>", "N2ns1AINS_1BEEE"),
        ("std::pair<A, A>", "St4pairI1AS0_E"),
        ("std::vector<int>", "St6vectorIiE"),
        ("std::chrono::seconds", "NSt6chrono7secondsE"),
        ("std::string", "NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"),
        ("N<3>", "1NILi3EE"),
        ("B<A*, A*>", "1BIP1AS1_E"),
    ],
)
def test_a_types_name_is_mangled_as_the_itanium_abi_mangles_it(spelled: str, mangled: str) -> None:
    assert mangle(spelled) == mangled


def test_a_substitution_past_the_tenth_is_numbered_in_base_36() -> None:
    assert [_base36(n) for n in (0, 9, 10, 35, 36, 40)] == ["0", "9", "A", "Z", "10", "14"]


def test_a_type_info_is_one_object_per_type_compared_ordered_and_hashed_by_its_name() -> None:
    one, same, other = typeid_of("int"), typeid_of("const int"), typeid_of("double")
    assert one is same and one == same and not one != same
    assert one != other and one != "i"
    assert other.before(one) and not one.before(other)
    assert one.hash_code() == same.hash_code() != other.hash_code()
    assert hash(one) == hash(same)
    assert str(typeid_of("std::vector<int>")) == "vector<int>"
    assert str(typeid_of("const A*")) == "const A*"
    assert repr(one) == "<type_info i>"
    assert isinstance(one, type_info)


def test_the_dynamic_type_is_the_objects_own_class_or_the_static_one_given() -> None:
    class Own:
        _cxx_layout_ = ("Own", (), ())

    assert typeid_dynamic(Own()).name() == "3Own"
    assert typeid_dynamic(ROOT.TH1F("h", "h", 1, 0, 1), "TObject").name() == "4TH1F"
    assert typeid_dynamic([1, 2], "std::vector<int>").name() == "St6vectorIiE"
    values = [True, 3, 2.5, "s", [1]]
    assert [typeid_dynamic(v).name() for v in values] == [
        "b", "i", "d", "NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", "4list"]
    with pytest.raises(bad_typeid, match="null pointer"):
        typeid_dynamic(None, "B")


def test_typeid_of_a_polymorphic_object_is_its_dynamic_type_and_else_its_static_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = CLASSES + """
    void t() {
      A *a = nullptr;
      A *a2 = new D;
      B *b = new B;
      B *b2 = new C;
      map<const type_info*, string> m;
      m[&typeid(A)] = "A"; m[&typeid(B)] = "B"; m[&typeid(C)] = "C"; m[&typeid(D)] = "D";
      cout << m[&typeid(*a)] << m[&typeid(*a2)] << m[&typeid(*b)] << m[&typeid(*b2)] << endl;
      void *p = b2;
      cout << typeid(b2).name() << " " << typeid(*b2).name() << " " << typeid(p).name() << endl;
      try { B *nb = nullptr; typeid(*nb); } catch (std::bad_typeid &e) { cout << "bad" << endl; }
    }
    """
    text = translate(source, "t.C")
    assert "typeid_of('A')" in text and "typeid_dynamic(b2, 'B')" in text
    assert output(capsys, source, ROOT) == "AABC\nP1B 1C Pv\nbad\n"


def test_typeid_names_the_builtin_types_and_compares_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    #include <typeinfo>
    void t() {
      int i = 3; double d = 1; unsigned int u = 2; float f = 1; char c = 'x'; long l = 1;
      unsigned long ul = 1; bool bb = true; Long64_t big = 1; char *s = nullptr;
      cout << typeid(i).name() << typeid(d).name() << typeid(u).name() << typeid(f).name()
           << typeid(c).name() << typeid(l).name() << typeid(ul).name() << typeid(bb).name()
           << typeid(big).name() << typeid(s).name() << typeid(i + d).name() << endl;
      cout << (typeid(i) == typeid(int)) << (typeid(i) != typeid(d))
           << typeid(d).before(typeid(i)) << (typeid(i).hash_code() == typeid(int).hash_code())
           << endl;
      const std::type_info &ti = typeid(Double_t);
      cout << ti.name() << endl;
    }
    """
    assert output(capsys, source) == "idjfclmbxPcd\n1111\nd\n"


def test_typeid_spells_a_class_with_its_namespaces_and_the_class_it_is_in(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    namespace ns { struct E { struct In {}; }; namespace { struct Hidden {}; } }
    enum Color { kRed };
    void t() {
      ns::E e; Color c = kRed;
      cout << typeid(e).name() << " " << typeid(ns::E::In).name() << " "
           << typeid(ns::Hidden).name() << " " << typeid(c).name() << " "
           << typeid(std::array<int, 3>).name() << " " << typeid(TH1F).name() << endl;
    }
    """
    names = "N2ns1EE N2ns1E2InE N2ns6HiddenE 5Color St5arrayIiLi3EE 4TH1F\n"
    assert output(capsys, source) == names


def test_typeid_finds_roots_class_of_an_object_for_tclass(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    void t() {
      TH1F *h = new TH1F("h", "h", 10, 0, 1);
      TObject *o = h;
      cout << typeid(*o).name() << " " << TClass::GetClass(typeid(*o))->GetName() << endl;
    }
    """
    assert output(capsys, source, ROOT) == "4TH1F TH1F\n"


def test_typeid_in_a_template_of_an_object_of_its_parameter_is_its_dynamic_type(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = CLASSES + """
    template <class T> void show(const T *obj) { cout << typeid(*obj).name() << endl; }
    void t() { B *b2 = new C; show(b2); }
    """
    assert output(capsys, source, ROOT) == "1C\n"


def test_typeid_of_a_type_that_is_not_known_is_refused() -> None:
    generic = translate("template <class T> void f() { auto &t = typeid(T*); }", "t.C")
    assert "uninstantiated('the function template f', " in generic
    assert "typeid of the template parameter T" in generic
    with pytest.raises(Refusal, match="typeid of an array whose size is not a number"):
        translate("void f(int n) { double a[n]; auto &t = typeid(a); }", "t.C")


def test_a_type_infos_root_name_drops_the_const_typeid_ignores() -> None:
    assert str(typeid_of("const long double &")) == "long double"
