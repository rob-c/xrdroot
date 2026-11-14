"""Objects of the classes a macro declares, written as ROOT writes them and read back.

``declared_objects.C`` declares classes of every kind of member ROOT's I/O
streams - numbers and packed floats, fixed and counted arrays, strings, the
standard containers, objects by value and by pointer, bases - and writes one
object of each. ROOT 6.40.04 ran it through ACLiC to make
``declared-objects-6.40.root``; here xrdroot translates and runs the same
macro, and each key's bytes are ROOT's. ROOT's file is then read back into the
translated classes, member by member.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.file import open_root
from xrdroot.pycint.execute import run_source
from xrdroot.pycxx.runtime.namespace import DECLARED
from xrdroot.pyroot.core.declared import find_class, made_from, payload_of
from xrdroot.wdeclared import Catalogue, Declaration, Field, normalised, parse_type
from xrdroot.wdobjects import Instance, write_instance
from xrdroot.writer import WBuffer

DATA = pathlib.Path(__file__).parent / "data"
SOURCE = (DATA / "declared_objects.C").read_text()
ORACLE = DATA / "declared-objects-6.40.root"
#: Where ``derived``'s map of strings says which pair it holds: ROOT's in-memory checksum of
#: ``pair<int,string>``, which cling works out from libc++'s own declaration of the pair, where
#: xrdroot gives the checksum of the pair's streamer description - the one ROOT also writes
#: into the file, so ROOT reads either.
PAIR_CHECKSUM = slice(164, 168)


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    declared = dict(DECLARED)
    yield from fresh(tmp_path)
    DECLARED.clear()
    DECLARED.update(declared)


def _payloads(path):
    with open_root(path) as found:
        return {key.name: key.payload(found._source) for key in found.all_keys()}


def _streamers(path):
    with open_root(path) as found:
        return found._source.streamers()


def test_each_object_is_written_byte_for_byte_as_root_writes_it():
    run_source(SOURCE, "declared_objects.C", ("mine.root",))
    mine, roots = _payloads("mine.root"), _payloads(ORACLE)
    assert mine["plain"] == roots["plain"]
    assert mine["a"] == roots["a"]
    ours, theirs = bytearray(mine["derived"]), bytearray(roots["derived"])
    assert theirs[PAIR_CHECKSUM].hex() == "0b5fb752" and ours[PAIR_CHECKSUM].hex() == "24ed66ba"
    ours[PAIR_CHECKSUM] = theirs[PAIR_CHECKSUM]
    assert ours == theirs


def test_the_file_describes_each_class_reached_with_roots_checksum():
    run_source(SOURCE, "declared_objects.C", ("mine.root",))
    infos = _streamers("mine.root")
    assert {"Plain", "Inner", "Derived", "Base1", "ns::Deep", "Versioned", "Obj", "Named",
            "TNamed", "pair<int,double>", "pair<int,string>"} <= set(infos)  # fmt: skip
    assert list(infos["Plain"])[:3] == ["b", "c", "s"] and "trans" not in infos["Plain"]
    assert infos["Plain"]["var"].count == "n" and infos["Plain"]["d32r"].title == "[0,20,10]"


def _read_back(name):
    run_source(SOURCE, "declared_objects.C", call=False)
    return ROOT.TFile(str(ORACLE)).Get(name)


def test_roots_file_reads_back_into_the_classes_the_macro_declares():
    plain = _read_back("plain")
    expect(
        (type(plain).__name__, "Plain"),
        ((plain.b, plain.c, plain.ll, plain.uc, plain.f16r), (True, 50, 6, 7, 4.4921875)),
        ((list(plain.arr), list(plain.var)), ([1, 2, 3], [7.0, 8.0])),
        ((plain.str, str(plain.tstr), type(plain.tstr).__name__), ("hello", "tee", "TString")),
        ((list(plain.vi), list(plain.vs), plain.mid[3]), ([10, 11], ["ab"], 4.5)),
        ((plain.inner.fI, plain.vinner[0].fD, plain.pinner.fD), (1, 2.0, 2.0)),
    )


def test_containers_objects_and_pointers_read_back_into_their_declared_kinds():
    derived = _read_back("derived")
    expect(
        ((derived.b1, derived.deep.x, derived.arr2[1][2]), (1, 5, 5)),
        ([list(v) for v in derived.vvf], [[1.5], [2.5, 3.5]]),
        ((derived.vpinner[0].fI, derived.vpinner[1]), (1, None)),
        ((derived.vver[0].v, derived.mis[2], sorted(derived.si)), (7, "two", [4, 5])),
        ((list(derived.ld), [bool(b) for b in derived.vb]), ([0.25], [True, False])),
        ((derived.obj.o, derived.pobj.o, derived.pnull, derived.ver.v), (9, 9, None, 7)),
    )
    named = _read_back("a")
    expect(
        ((named.GetName(), named.GetTitle(), named.fN, named.ClassName()), ("a", "b", 3, "Named"))
    )


EVOLVED = r"""
class Extra { public: int x = 8; };
class Evolved : public Extra {   // a base added since: left as the constructor made it
public:
   float i = 0;        // an int on file, a float now
   unsigned char s = 0;
   bool d = false;
   int added = 42;     // not on file: left as the constructor made it
   Double32_t f = 0;
};
"""

WRITTEN = r"""
class Evolved {
public:
   int i = -7;
   short s = -1;
   double d = 2.5;
   int dropped = 3;    // on file, gone now
   float f = 1.25;
};
void evolved(const char *name) {
   TFile f(name, "RECREATE");
   Evolved e;
   f.WriteObject(&e, "e");
}
"""


def test_a_member_whose_type_changed_is_converted_as_cpp_converts_it():
    run_source(WRITTEN, "evolved.C", ("evolved.root",))
    DECLARED.pop("Evolved")
    run_source(EVOLVED, "now.C", call=False)
    e = ROOT.TFile("evolved.root").Get("e")
    assert (e.i, e.s, e.d, e.added, e.f, e.x) == (-7.0, 255, True, 42, 1.25, 8)
    assert isinstance(e.i, float) and not hasattr(e, "dropped")


def test_a_record_of_a_class_no_one_declared_is_not_made_into_one():
    assert made_from("NoSuchClass", {"x": 1}) is None
    assert find_class("NoSuchClass") is None
    assert payload_of(3) is None


CONTAINERS = r"""
#include <list>
#include <map>
#include <deque>
class Holder {
public:
   std::list<std::pair<int,double> > lp;
   std::multimap<int,double> mm;
   std::deque<short> dq;
   std::unordered_map<int,int> um;
   std::pair<int,float> p;
};
void containers(const char *name) {
   Holder h;
   h.lp.push_back(std::make_pair(1, 2.0));
   h.mm.insert(std::make_pair(2, 4.0));
   h.mm.insert(std::make_pair(2, 5.0));
   h.dq.push_back(3);
   h.dq.push_front(1);
   h.um[1] = 2;
   h.p = std::make_pair(5, 0.5f);
   TFile f(name, "RECREATE");
   f.WriteObject(&h, "h");
}
"""


def test_lists_multimaps_deques_and_pairs_go_out_and_come_back():
    run_source(CONTAINERS, "containers.C", ("containers.root",))
    h = ROOT.TFile("containers.root").Get("h")
    assert [(p.first, p.second) for p in h.lp] == [(1, 2.0)]
    assert h.mm.items() == [(2, 4.0), (2, 5.0)] and h.mm.count(2) == 2
    assert list(h.dq) == [1, 3] and dict(h.um.items()) == {1: 2}
    assert (h.p.first, h.p.second) == (5, 0.5)
    infos = _streamers("containers.root")
    assert infos["Holder"]["mm"].typename == "multimap<int,double>"


def test_a_type_is_named_as_a_streamer_names_it():
    assert normalised("std::vector<std::vector<Int_t> >") == "vector<vector<int> >"
    assert normalised("const std::map<long long, std::string>&") == "map<Long64_t,string>"
    assert parse_type("ns::A**").stars == 2


def test_what_this_writer_cannot_lay_out_is_refused_by_name():
    def catalogue(*fields, bases=()):
        return Catalogue({"C": Declaration("C", 1, True, bases, fields)})

    with pytest.raises(UnsupportedFeatureError, match="C::x is a long double"):
        catalogue(Field("x", "long double", "", ())).info("C")
    with pytest.raises(UnsupportedFeatureError, match="derives from Unknown"):
        catalogue(bases=("Unknown",)).info("C")
    with pytest.raises(UnsupportedFeatureError, match="Thing is not a class"):
        catalogue(Field("v", "std::vector<Thing>", "", ())).info("C")
    with pytest.raises(UnsupportedFeatureError, match="Thing is not a class"):
        catalogue(Field("v", "std::vector<Thing*>", "", ())).info("C")


def test_a_counted_array_shorter_than_its_count_is_an_error():
    counter, counted = Field("n", "int", "", ()), Field("v", "float*", "[n]", ())
    fields = (counter, counted, Field("a", "int", "", (2,)))
    decl = Declaration("C", 2, False, (), fields)
    catalogue = Catalogue({"C": decl})
    with pytest.raises(ValueError, match="v holds 1 values where n says 2"):
        write_instance(WBuffer(), Instance("C", {"n": 2, "v": [1.0], "a": [1, 2]}), catalogue)
    with pytest.raises(ValueError, match="a holds 1 values where 2 were declared"):
        write_instance(WBuffer(), Instance("C", {"n": 0, "v": None, "a": [1]}), catalogue)
    buf = WBuffer()
    write_instance(buf, Instance("C", {"n": 0, "v": None, "a": np.array([1, 2])}), catalogue)
    assert bytes(buf.data).hex() == "4000000f0002000000000000000001" + "00000002"


def test_a_container_of_containers_of_objects_is_refused():
    decl = Declaration("C", 1, True, (), (Field("v", "std::vector<std::vector<Inner> >", "", ()),))
    inner = Declaration("Inner", 1, True, (), (Field("x", "int", "", ()),))
    catalogue = Catalogue({"C": decl, "Inner": inner})
    with pytest.raises(UnsupportedFeatureError, match="member-wise inside its own record"):
        write_instance(WBuffer(), Instance("C", {"v": [[Instance("Inner", {"x": 1})]]}), catalogue)


MIXED = r"""
#include "TAttLine.h"
#include "TNamed.h"
class Part { public: int p = 1; };
class Whole : public Part { public: Part part; float w = 2; };
class NoDefault {
public:
   NoDefault(int a) : x(a) {}
   int x;
};
class Lined : public TAttLine {
public:
   Part *always; //-> never null
   std::vector<Whole> wholes;
   TNamed named;
   NoDefault *nd;
   Lined() : always(new Part()), nd(new NoDefault(4)) {
      SetLineColor(3);
      wholes.push_back(Whole());
   }
};
void mixed(const char *name) {
   TFile f(name, "RECREATE");
   Lined l;
   l.named.SetName("inside");
   f.WriteObject(&l, "l");
}
"""


def test_roots_bases_objects_and_pointers_never_null_are_written_in_place():
    run_source(MIXED, "mixed.C", ("mixed.root",))
    infos = _streamers("mixed.root")
    assert infos["Lined"]["always"].stype == 68 and infos["Lined"]["named"].stype == 61
    lined = ROOT.TFile("mixed.root").Get("l")
    assert lined.GetLineColor() == 3 and lined.always.p == 1 and lined.nd.x == 4
    assert (lined.wholes[0].p, lined.wholes[0].part.p, lined.wholes[0].w) == (1, 1, 2.0)
    assert lined.named.GetName() == "inside"


def test_a_class_of_roots_that_nothing_derives_from_is_a_base_by_its_own_description():
    decl = Declaration("C", 1, True, ("TObjString",), (Field("c", "char", "", ()),))
    catalogue = Catalogue({"C": decl})
    assert catalogue.info("C")[2][0][:4] == ("TStreamerBase", "TObjString", "", 0)
    buf = WBuffer()
    write_instance(buf, Instance("C", {"TObjString": {"fString": "s"}, "c": "A"}), catalogue)
    assert bytes(buf.data).endswith(b"\x01s\x41")
    assert Catalogue({}).tobject("TNamed") and not Catalogue({}).tobject("TAttLine")


def test_an_enumeration_is_an_int_by_its_own_name():
    decl = Declaration("C", 1, True, (), (Field("k", "EKind", "", ()),))
    elements = Catalogue({"C": decl}, frozenset({"EKind"})).info("C")[2]
    assert elements[0][3] == 3 and elements[0][8] == "EKind"


def test_a_value_that_is_no_object_where_one_should_be_is_refused():
    decl = Declaration("C", 1, True, (), (Field("i", "Inner", "", ()),))
    inner = Declaration("Inner", 1, True, (), (Field("x", "int", "", ()),))
    with pytest.raises(UnsupportedFeatureError, match="a int stands where an object"):
        write_instance(WBuffer(), Instance("C", {"i": 3}), Catalogue({"C": decl, "Inner": inner}))


def test_a_pointer_read_to_a_class_no_one_declared_is_kept_as_read():
    from xrdroot.pyroot.core.declared import converted

    assert converted({"x": 1}, parse_type("Unknown*"), None) == {"x": 1}
    assert converted({"x": 1}, parse_type("Unknown"), None) == {"x": 1}
    assert converted(3, parse_type("Unknown"), None) == 3
    assert converted([1, 2], parse_type("int"), None).tolist() == [1, 2]
    assert converted(-1.5, parse_type("short"), 0) == -1


def test_roots_members_without_a_setter_are_left_as_they_are():
    from xrdroot.pyroot.core.declared import fill_root

    named = ROOT.TNamed("n", "t")
    fill_root(named, {"TObject": {"fUniqueID": 4}, "fTitle": "u", "fNoSetter": 1})
    assert (named.GetName(), named.GetTitle(), named.GetUniqueID()) == ("n", "u", 4)


LINKED = r"""
class Node {
public:
   int v = 0;
   Node *next = nullptr;
};
void linked(const char *name) {
   Node first;
   first.v = 1;
   first.next = new Node();
   first.next->v = 2;
   TFile f(name, "RECREATE");
   f.WriteObject(&first, "n");
}
"""


def test_a_class_that_points_at_its_own_kind_is_written_and_read_link_by_link():
    run_source(LINKED, "linked.C", ("linked.root",))
    first = ROOT.TFile("linked.root").Get("n")
    assert (first.v, first.next.v, first.next.next) == (1, 2, None)
