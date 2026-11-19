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
import re

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.cppstd.stdlib import complex_type
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.file import open_root
from xrdroot.pycint.execute import run_source
from xrdroot.pycxx.runtime.namespace import DECLARED
from xrdroot.pyroot.core.declared import find_class, made_from, payload_of
from xrdroot.wdeclared import (
    NO_DIMS,
    Catalogue,
    Declaration,
    Field,
    normalised,
    parse_type,
)
from xrdroot.wdobjects import Instance, write_instance, write_members, write_value
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
    with pytest.raises(UnsupportedFeatureError, match="C::a is a int whose array length"):
        catalogue(Field("a", "int", "", (None,))).info("C")


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
   f.WriteObjectAny(&l, "Lined", "l");
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


TREE = r"""
#include <vector>
#include <string>
class Track { public: float pt = 1; Track() {} Track(float p) : pt(p) {} };
class Event : public TObject {
public:
   int n = 0;
   std::string tag;
   std::vector<Track> tracks;
   ClassDefOverride(Event, 3);
};
class Flat { public: double x = 0; };
void tree(const char *name) {
   {
      TFile f(name, "RECREATE");
      TTree t("T", "events");
      Event *e = new Event;
      Flat flat;
      t.Branch("event", &e);
      t.Branch("flat", &flat, 32000, 0);
      for (int i = 0; i < 3; ++i) {
         e->n = i;
         e->tag = Form("evt%d", i);
         e->tracks.clear();
         for (int j = 0; j < i; ++j) e->tracks.push_back(Track(j + 0.5));
         flat.x = i / 2.0;
         t.Fill();
      }
      t.Write();
   }
   TFile f(name);
   TTree *t = (TTree *)f.Get("T");
   Event *e = nullptr;
   Flat *flat = nullptr;
   t->SetBranchAddress("event", &e);
   t->SetBranchAddress("flat", &flat);
   for (int i = 0; i < t->GetEntries(); ++i) {
      t->GetEntry(i);
      printf("%d %s %d %g %g\n", e->n, e->tag.c_str(), (int)e->tracks.size(),
             e->tracks.size() ? e->tracks.back().pt : -1.0, flat->x);
   }
}
"""


def test_a_branch_of_a_class_this_writer_does_not_split_is_written_whole(capfd):
    run_source(TREE, "tree.C", ("tree.root",))
    assert capfd.readouterr().out.splitlines() == [
        "0 evt0 0 -1 0",
        "1 evt1 1 0.5 0.5",
        "2 evt2 2 1.5 1",
    ]
    with open_root("tree.root") as f:
        infos = f._source.streamers()
    assert {"Event", "Track", "Flat"} <= set(infos)


ZERO = r"""
struct MyS0 { static constexpr short Class_Version() { return 0; } int x = 4; };
struct MyS1 { static constexpr short Class_Version() { return 1; } };
void zero() {
   MyS0 s0;
   MyS1 s1;
   TMemFile f("mem19371.root", "RECREATE");
   f.WriteObject(&s0, "s0");
   f.WriteObject(&s1, "s1");
   printf("%d %d %d %d\n", f.Get<MyS0>("s0")->x, f.Get<MyS1>("s1") != nullptr,
          f.GetKey("s0")->GetObjlen(), f.GetKey("s1")->GetObjlen());
}
"""


def test_a_class_with_a_version_and_no_classdef_is_foreign_and_reads_back_while_written(capfd):
    run_source(ZERO, "zero.C")
    assert capfd.readouterr().out == "4 1 14 10\n"


RESIZED = r"""
#include <vector>
class Slot { public: int v = 6; };
class Slots { public: std::vector<Slot> slots; Slots() { slots.resize(2); } };
void resized(const char *name) {
   Slots s;
   TFile f(name, "RECREATE");
   f.WriteObject(&s, "s");
}
"""


def test_the_slots_resize_made_are_written_as_default_made_objects():
    run_source(RESIZED, "resized.C", ("resized.root",))
    back = ROOT.TFile("resized.root").Get("s")
    assert [slot.v for slot in back.slots] == [6, 6]
    from xrdroot.pyroot.core.declared import _present, catalogue_of, find_class

    catalogue = catalogue_of(find_class("Slots"))
    assert _present(None, parse_type("Slot"), catalogue).v == 6  # a slot not made yet


EVENT = r"""
#include "TClonesArray.h"
#include "TRef.h"
#include "TBits.h"
#include "TH1F.h"
#include <vector>
class Hit : public TObject {
public:
   float fE = 0;
   Hit() {}
   Hit(float e) : fE(e) {}
   Hit(const Hit &other) : TObject(other), fE(other.fE + 100) {}
   ClassDefOverride(Hit, 2);
};
class Evt : public TObject {
public:
   char fType[8];
   char *fName;
   int fN = 0;
   Double32_t *fValues; //[fN][0,10,16]
   std::vector<Double32_t> fMore;
   TClonesArray *fHits; //->
   TRef fLast;
   TBits fBits;
   TH1F *fH;
   Evt() : fName(nullptr), fValues(0), fHits(new TClonesArray("Hit", 10)), fH(0) {}
   void Build(int n) {
      strcpy(fType, "kind");
      fName = (char *)"evt";
      fN = n;
      fValues = new Double32_t[n];
      for (int i = 0; i < n; ++i) { fValues[i] = i; new ((*fHits)[i]) Hit(i + 0.5); }
      if (fValues) fMore.push_back(2.5);
      Hit *first = (Hit *)fHits->At(0);
      fLast = first;
      fBits.SetBitNumber(3);
      fH = new TH1F("h", "h", 4, 0, 4);
      fH->Fill(1);
   }
   ClassDefOverride(Evt, 1);
};
void event(const char *name) {
   TFile f(name, "RECREATE");
   TTree t("T", "events");
   Evt *e = new Evt;
   t.Branch("event", &e);
   e->Build(3);
   t.Fill();
   t.Write();
}
"""


def test_an_event_of_objects_references_and_histograms_goes_whole_into_a_tree():
    run_source(EVENT, "event.C", ("event.root",))
    with open_root("event.root") as f:
        entry = f.tree("T")["event"].array(0, 1)[0]
    expect(
        ((entry["fName"], entry["fN"], list(entry["fValues"])), ("evt", 3, [0.0, 1.0, 2.0])),
        (bytes(entry["fType"][:4]), b"kind"),
        ((len(entry["fHits"]), entry["fHits"][2]["fE"]), (3, 2.5)),
        (list(entry["fMore"]), [2.5]),
    )


def test_an_overload_taking_an_object_is_not_chosen_for_a_number(capfd):
    run_source(
        "class P { public: float x; P(float v) : x(v) {} P(const P &o) : x(o.x + 1) {} };\n"
        'void pick() { P a(2.5); P b(a); printf("%g %g\\n", a.x, b.x); }',
        "pick.C",
    )
    assert capfd.readouterr().out == "2.5 3.5\n"


def test_roots_objects_among_members_are_handed_to_the_writer_as_it_takes_them():
    from xrdroot.pyroot.core.declared import _clones, _root_object
    from xrdroot.wdobjects import Clones

    catalogue = Catalogue({})
    line = _root_object(ROOT.TLine(0, 0, 1, 1), "TLine", catalogue)
    assert line["fX2"] == 1.0 and _root_object(ROOT.TObject(), "TNoSuch", catalogue) is not None
    named = ROOT.TClonesArray("TNamed")
    assert _clones(named, catalogue) == Clones("TNamed", "TNameds", [])
    assert _clones(named, catalogue).classname == "TNamed"
    decl = Declaration("Slot", 1, True, (), (Field("v", "int", "", ()),))
    buf = WBuffer()
    write_instance(buf, Instance("Slot", {"v": 1}), Catalogue({"Slot": decl}))
    assert _clones(ROOT.TClonesArray("Slot"), Catalogue({"Slot": decl})).name == "Slots"
    holder = Declaration("H", 1, True, (), (Field("c", "TClonesArray*", "->", ()),))
    write_instance(
        WBuffer(), Instance("H", {"c": Clones("Slot", "Slots", [None])}),
        Catalogue({"H": holder, "Slot": decl}),
    )  # fmt: skip


def test_an_overload_taking_a_string_class_takes_a_c_string():
    out = run_source(
        "int pick(double x) { return 3; }\nint pick(const TString &s) { return 1; }\n"
        'int pick(int n[2]) { return 2; }\nint which() { return pick("abc") * 10 + pick(1.5); }',
        "which.C",
    )
    assert out == 13


def test_a_std_array_member_is_the_fixed_array_root_lays_out():
    """ROOT 6.40 writes ``std::array<int,4>`` as the ``int a[4]`` it is - a
    ``TStreamerBasicType`` whose type carries the array offset, its size four values' worth
    and four as the length - which is what a member of one is read back by."""

    def catalogue(*fields, bases=()):
        return Catalogue({"C": Declaration("C", 1, True, bases, fields)})

    info = catalogue(Field("a0", "std::array<int,4>", "", ()),
                     Field("a1", "std::array<float,2>", "", ())).info("C")
    assert info[2] == (("TStreamerBasicType", "a0", "", 23, 16, 4, 1, (4, 0, 0, 0, 0), "int", ()),
                       ("TStreamerBasicType", "a1", "", 25, 8, 2, 1, (2, 0, 0, 0, 0), "float", ()))
    # An array of objects is not a basic type's array, and is refused rather than guessed.
    with pytest.raises(UnsupportedFeatureError, match=r"C::v is a std::array<Thing,2>, which"):
        catalogue(Field("v", "std::array<Thing,2>", "", ())).info("C")
    # ... and so is a length that is no number anything can count at compile time.
    with pytest.raises(UnsupportedFeatureError, match=r"C::v is a std::array<int,n>, which"):
        catalogue(Field("v", "std::array<int,n>", "", ())).info("C")


def test_the_length_of_a_std_array_must_be_a_comment_free_whole_number():
    """``std::array<int,n>`` of a length nothing can count is refused, and so is one written
    with stars or another template's arguments."""
    from xrdroot.wdeclared import _whole, parse_type

    assert _whole(parse_type("4")) == 4
    assert _whole(parse_type("std::vector<int>")) is None
    assert _whole(parse_type("T*")) is None
    assert _whole(parse_type("n")) is None  # a name, not a number


def test_a_fixed_array_is_taken_apart_into_what_it_holds_and_its_lengths():
    """``std::array`` of ``std::array`` is one fixed array of the innermost type, as many
    dimensions as the nesting asks for, outermost first - and anything else is no fixed
    array at all, so that another member rule may take it."""
    from xrdroot.wdeclared import _array_shape, parse_type

    assert _array_shape(parse_type("std::array<std::array<int,4>,3>")) == (
        parse_type("int"), (3, 4))  # fmt: skip
    assert _array_shape(parse_type("std::array<int,4>")) == (parse_type("int"), (4,))
    for nothing in ("int", "std::array<int,4>*", "std::array<int*>",
                    "std::array<std::array<int*>,3>,3", "std::array<int,0>",
                    "std::array<std::array<int>,3>", "std::array<std::array<int,3>,n>",
                    "std::array<T*,3>"):
        assert _array_shape(parse_type(nothing)) is None, nothing


def test_a_std_array_of_a_std_array_is_the_two_dimensional_array_root_lays_out():
    """ROOT 6.40 resolves ``std::array<std::array<int,3>,3>`` to the element it gives
    ``int a[3][3]``: one ``int`` element, nine values' worth of room, a length of nine, two
    dimensions outermost first - the oracle's answer for the stdarray roottest's ``a2``,
    which prints ``Is Basic: 23`` and two maximum indices of three, as its ``a3`` does."""

    def catalogue(*fields):
        return Catalogue({"C": Declaration("C", 1, True, (), fields)})

    info = catalogue(Field("a2", "std::array<std::array<int,3>,3>", "", ()),
                     Field("a3", "int", "", (3, 3)),
                     Field("a8", "std::array<std::array<float,3>,3>", "", ()),
                     Field("b", "std::array<std::array<int,4>,3>", "", ()),
                     Field("c", "std::array<std::array<std::array<int,2>,4>,3>", "", ()),
                     ).info("C")  # fmt: skip
    assert info[2] == (
        ("TStreamerBasicType", "a2", "", 23, 36, 9, 2, (3, 3, 0, 0, 0), "int", ()),
        ("TStreamerBasicType", "a3", "", 23, 36, 9, 2, (3, 3, 0, 0, 0), "int", ()),
        ("TStreamerBasicType", "a8", "", 25, 36, 9, 2, (3, 3, 0, 0, 0), "float", ()),
        ("TStreamerBasicType", "b", "", 23, 48, 12, 2, (3, 4, 0, 0, 0), "int", ()),
        ("TStreamerBasicType", "c", "", 23, 96, 24, 3, (3, 4, 2, 0, 0), "int", ()))


def test_a_std_array_of_a_std_array_writes_its_values_row_by_row_as_its_plain_array_does():
    """The values of a nested array go as the flattened nine ints it holds, in the order
    ``std::array`` holds them - the very bytes its plain ``int a[3][3]`` twin writes, which
    is what a reader of the superposed layout expects."""
    made = Catalogue({"C": Declaration("C", 1, True, (), (
        Field("a2", "std::array<std::array<int,3>,3>", "", ()),
        Field("a3", "int", "", (3, 3))))})  # fmt: skip
    rows = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    buf = WBuffer()
    write_members(buf, Instance("C", {"a2": rows, "a3": rows}), made)
    nine = b"".join(value.to_bytes(4, "big") for row in rows for value in row)
    assert bytes(buf.data) == nine + nine  # the nested member's nine ints, then the plain's


def test_a_std_array_nested_deeper_than_root_records_is_refused_by_name():
    """ROOT's ``TStreamerElement`` records five array dimensions; a member nested deeper is
    refused rather than written with a shape a reader cannot recover."""
    deep = "std::array<" * 6 + "int,2" + ">,2" * 5 + ">"
    made = Catalogue({"C": Declaration("C", 1, True, (), (Field("v", deep, "", ()),))})
    with pytest.raises(UnsupportedFeatureError, match=r"C::v is a std::array.*nested 6 arrays"):
        made.info("C")


def test_a_std_array_of_a_std_array_of_objects_is_one_element_of_two_dimensions():
    """An array of objects nested in another is still one element - the object's type with
    the array offset on it, every value's worth of room, two dimensions outermost first."""
    def catalogue(*fields):
        return Catalogue({"C": Declaration("C", 1, True, (), fields),
                          "A": Declaration("A", 1, True, (), (Field("x", "int", "", ()),))})

    info = catalogue(Field("v", "std::array<std::array<A,3>,3>", "", ()),
                     Field("w", "std::array<A,9>", "", ())).info("C")  # fmt: skip
    assert info[2] == (("TStreamerObjectAny", "v", "", 82, 72, 9, 2, (3, 3, 0, 0, 0), "A", ()),
                       ("TStreamerObjectAny", "w", "", 82, 72, 9, 1, (9, 0, 0, 0, 0), "A", ()))


def test_an_array_of_objects_whose_value_is_not_there_is_no_element_at_all():
    """A member of an array of objects that holds nothing writes nothing, which is what an
    empty array of them is."""
    from xrdroot.pyroot.core.declared import _plain_member

    fields = (Field("v", "std::array<A,2>", "", ()),)
    made = Catalogue({"C": Declaration("C", 1, True, (), fields),
                      "A": Declaration("A", 1, True, (), (Field("x", "int", "", ()),))})
    field = made.declared("C").fields[0]
    assert _plain_member(None, field, made) == []


def test_an_array_of_objects_is_written_one_record_after_another():
    """``std::array<A,2>`` and ``A a[2]`` are the same layout - ``kOffsetL + kObject``, two
    values' worth of room, two as the length - and the values go in one record each."""
    def catalogue(*fields, bases=()):
        return Catalogue({"C": Declaration("C", 1, True, bases, fields),
                           "A": Declaration("A", 1, True, (), (Field("x", "int", "", ()),))})

    info = catalogue(Field("v", "std::array<A,2>", "", ()),
                     Field("w", "A", "", (2,))).info("C")
    assert info[2] == (("TStreamerObjectAny", "v", "", 82, 16, 2, 1, (2, 0, 0, 0, 0), "A", ()),
                       ("TStreamerObjectAny", "w", "", 82, 16, 2, 1, (2, 0, 0, 0, 0), "A", ()))


def test_an_array_of_objects_writes_one_record_per_element_and_refuses_one_it_cannot():
    """The values of an array of objects go in one record each, and one whose class nothing
    describes is refused rather than laid out by guesswork."""
    def catalogue(*fields, bases=()):
        return Catalogue({"C": Declaration("C", 1, True, bases, fields),
                           "A": Declaration("A", 1, True, (), (Field("x", "int", "", ()),))})

    with pytest.raises(UnsupportedFeatureError, match="C::v is a std::array<Unknown,2>"):
        catalogue(Field("v", "std::array<Unknown,2>", "", ())).info("C")
    # The writer's own path: one record per element, in place, each its member's bytes.
    made = catalogue(Field("v", "std::array<A,2>", "", ()))
    buf = WBuffer()
    write_members(buf, Instance("C", {"v": [Instance("A", {"x": 1}),
                                            Instance("A", {"x": 2})]}), made)  # fmt: skip
    data = bytes(buf.data)
    assert len(data) == 28  # two records of one member each, 14 bytes apiece
    assert data.endswith(b"\x00\x00\x00\x02") and b"\x00\x00\x00\x01" in data


#: ROOT 6.40.04's bytes for ``class V : public TObject { std::vector<double> fVectorD[4];
#: std::vector<int> fVint[2]; }`` - written by the oracle through ACLiC with 1.5 in ``[0]``,
#: 2.5 and 3.5 in ``[3]``, and 7 in the second int vector. The array of containers is one
#: record holding every element's count and values in turn, not a record apiece.
ARRAY_OF_CONTAINERS_BYTES = (
    "4000004c0001"
    "00010000000000000000"
    "4000002a000a000000013ff80000000000000000000000000000000000024004000000000000"
    "400c000000000000"
    "4000000e000a000000000000000100000007"
)


def test_an_array_of_containers_is_roots_stl_element_with_the_array_offset():
    """ROOT 6.40 describes ``vector<double> f[4]`` as one ``TStreamerSTL`` whose type carries
    the array offset, room for four containers and four as the length - the oracle's answer
    for the bigevent roottest's ``Event::fVectorD``, which prints ``type=320 stl=1 ctype=8``,
    a size of 96, a length of 4, one dimension and a maximum index of 4 - and a
    two-dimensional one as one element of six, two dimensions outermost first."""

    def catalogue(*fields):
        return Catalogue({"C": Declaration("C", 1, True, (), fields)})

    info = catalogue(Field("fVectorD", "vector<double>", "array of STL vectors of doubles", (4,)),
                     Field("v", "vector<int>", "", (2, 3)),
                     Field("one", "vector<int>", "", ())).info("C")  # fmt: skip
    assert info[2] == (
        ("TStreamerSTL", "fVectorD", "array of STL vectors of doubles", 320, 96, 4, 1,
         (4, 0, 0, 0, 0), "vector<double>", (1, 8)),
        ("TStreamerSTL", "v", "", 320, 144, 6, 2, (2, 3, 0, 0, 0), "vector<int>", (1, 3)),
        ("TStreamerSTL", "one", "", 300, 24, 0, 0, NO_DIMS, "vector<int>", (1, 3)))


def test_an_array_of_containers_writes_every_element_into_its_one_record():
    """The values of such an array go as one record's worth: the version a container carries,
    then each element's count and what it holds, in the order the array holds them - ROOT's
    own bytes for a class of an array of doubles' vectors and an array of ints'."""
    made = Catalogue({"V": Declaration("V", 1, False, ("TObject",), (
        Field("fVectorD", "vector<double>", "array of STL vectors of doubles", (4,)),
        Field("fVint", "vector<int>", "", (2,))))})  # fmt: skip
    buf = WBuffer()
    write_instance(buf, Instance("V", {"TObject": {}, "fVectorD": [[1.5], [], [], [2.5, 3.5]],
                                        "fVint": [[], [7]]}), made)  # fmt: skip
    assert bytes(buf.data).hex() == ARRAY_OF_CONTAINERS_BYTES


def test_an_array_of_containers_of_objects_is_refused_by_name():
    """ROOT writes a container of objects member by member, and how it lays one of those out
    column by column inside an array's single record is not a shape this writer knows: the
    member is refused with a sentence naming it rather than written on a guess."""
    made = Catalogue({"C": Declaration("C", 1, True, (), (
        Field("v", "vector<TLine>", "", (2,)),))})
    with pytest.raises(UnsupportedFeatureError,
                       match=r"a vector<TLine> array holds objects, which ROOT writes"):
        write_members(WBuffer(), Instance("C", {"v": [[], []]}), made)  # fmt: skip


#: ROOT 6.40.04's bytes for ``class P : public TObject { std::vector<int> *fVint;
#: std::string *fS; std::vector<int> *fA[2]; }``, streamed by the oracle through ACLiC with every
#: pointer null and then with ``fVint`` {1,2}, ``fS`` "hi" and ``fA[1]`` {7}. A pointer to a
#: container is written as the container itself - an empty one where the pointer is null, an
#: empty string for a ``std::string`` - and an array of pointers as the one record round all
#: of its elements, each whose count follows the one before.
POINTER_TO_CONTAINER_BYTES = {
    "null": "4000002b00010001000000000000000040000006000a0000000040000003000a"
            "004000000a000a0000000000000000",
    "set": "400000390001000100000000000000004000000e000a000000020000000100000002"
           "40000005000a0268694000000e000a000000000000000100000007",
}  # fmt: skip


def _a_class_of_pointers_to_containers():
    return Catalogue({"P": Declaration("P", 1, False, ("TObject",), (
        Field("fVint", "vector<int>*", "", ()),
        Field("fS", "string*", "", ()),
        Field("fA", "vector<int>*", "", (2,))))})  # fmt: skip


def test_a_pointer_to_a_container_is_roots_stl_pointer_element():
    """ROOT describes ``vector<T> *f`` as ``kSTLp``, 71, its container number offset by
    ``kOffsetP`` - ``stl=41`` for a vector, 365 for the ``kSTLstring`` a pointer to a string is
    - and the room a pointer takes; an array of such pointers as ``kSTLp + kOffsetL``, 91,
    with its length and maximum index. The oracle says so for the bigevent roottest's
    ``fVectorTobject``, ``fVectorTnamed``, ``fListString``, ``fMapTAxisp``, ``fSetTAxisp``
    and ``fMultiSetTAxisp``."""
    elements = _a_class_of_pointers_to_containers().info("P")[2]
    assert elements[1:] == (
        ("TStreamerSTL", "fVint", "", 71, 8, 0, 0, NO_DIMS, "vector<int>*", (41, 3)),
        ("TStreamerSTL", "fS", "", 71, 8, 0, 0, NO_DIMS, "string*", (365, 365)),
        ("TStreamerSTL", "fA", "", 91, 16, 2, 1, (2, 0, 0, 0, 0), "vector<int>*", (41, 3)))


def test_a_pointer_to_a_container_is_written_as_the_container_root_writes():
    """ROOT streams a pointer to a container exactly as the container it points at, so a null
    pointer is the empty container - a count of nothing, or a string of no characters - and
    nothing marks the pointer itself."""
    made = _a_class_of_pointers_to_containers()
    cases = {
        "null": {"fVint": None, "fS": None, "fA": [None, None]},
        "set": {"fVint": [1, 2], "fS": "hi", "fA": [[], [7]]},
    }
    for case, members in cases.items():
        buf = WBuffer()
        write_instance(buf, Instance("P", {"TObject": {}, **members}), made)
        assert bytes(buf.data).hex() == POINTER_TO_CONTAINER_BYTES[case], case


def test_a_container_of_pointers_to_strings_holds_pointers_to_objects():
    """``list<string *>`` is ``kObjectp``, 63, as any container of pointers is - ROOT needs no
    class of the kit for the thing pointed at, so the bigevent roottest's ``fListStringp`` is
    laid out rather than refused."""
    made = Catalogue({"C": Declaration("C", 1, True, (), (
        Field("fListStringp", "list<string *>", "STL list of pointers to strings", ()),))})
    assert made.info("C")[2] == (
        ("TStreamerSTL", "fListStringp", "STL list of pointers to strings", 300, 24, 0, 0,
         NO_DIMS, "list<string*>", (2, 63)),)


def test_a_counted_array_of_strings_is_refused_as_the_streamer_loop_it_is():
    """``TString *f //[n]`` is ``kStreamLoop``: as many strings as the count says, one after
    another, with no record round them. This writer has not been taught that layout, and the
    member is refused by name rather than written as a pointer to one string."""
    made = Catalogue({"C": Declaration("C", 1, True, (), (
        Field("fTstringp", "TString*", "[fNvertex] array of TString", ()),))})
    with pytest.raises(UnsupportedFeatureError,
                       match=r"fTstringp is a TString\* counted by a //\[n\] comment"):
        made.info("C")  # fmt: skip


def test_a_class_spelled_with_a_space_where_a_streamer_names_it_is_the_same_class():
    """A declaration spells a template instance as C++ writes it,
    ``template2<template1<int> >``, while a streamer names the very class
    ``template2<template1<int>>``: the catalogue answers to either spelling, so the bigevent
    roottest's ``Event::fTemplateMember`` - an empty class - is laid out, not refused."""
    spelled = "template2<template1<int> >"
    made = Catalogue({spelled: Declaration(spelled, 1, False, (), ())})
    assert made.known("template2<template1<int>>")
    assert made.info("template2<template1<int>>")[2] == ()
    assert set(made.entries()) == {spelled}


#: ROOT 6.40.04's ``TBufferFile::WriteObjectAny`` of a ``classWithComplex(1, 2, 3, 4)`` - a
#: ``std::complex<float> fcf`` and a ``std::complex<double> fcd`` - as the oracle writes it: the
#: record's byte count, the version a class of no ``ClassDef`` carries (0) and its checksum,
#: then each member as a record of its own, ``_real`` before ``_imag``.
COMPLEX_BYTES = (
    "400000320000ef61cc47"
    "4000000e00002e5c46083f80000040000000"
    "400000160000db2371a140080000000000004010000000000000"
)
#: The oracle's bytes for a class of every shape a complex comes in: the two of them by value,
#: ``std::complex<double> arr[3]``, a pointer that may be null and one declared ``//->`` never
#: to be, and a ``vector<complex<double> >``, which ROOT writes member-wise.
COMPLEX_SHAPES_BYTES = (
    "400000ce0000b0127892"
    "4000000e00002e5c46083f80000040000000"
    "400000160000db2371a140080000000000004010000000000000"
    "400000160000db2371a140240000000000004034000000000000"
    "400000160000db2371a140260000000000004035000000000000"
    "400000160000db2371a140280000000000004036000000000000"
    "00000000"
    "400000160000db2371a140140000000000004018000000000000"
    "4000002c400a0000db2371a100000002"
    "401c000000000000402200000000000040200000000000004024000000000000"
)


def test_a_complex_member_is_the_two_numbers_root_streams_in_place():
    """ROOT's dictionary describes ``std::complex<T>`` as the pair of ``T`` it is - ``_real``
    then ``_imag``, and no version, so its objects carry that description's checksum - and
    streams a member of one in place: twice a number's room, not the pointer a member of a
    class held by value takes."""
    decl = Declaration("classWithComplex", 1, True, (),
                       (Field("fcf", "std::complex<float>", "", ()),
                        Field("fcd", "std::complex<double>", "", ())))  # fmt: skip
    catalogue = Catalogue({"classWithComplex": decl})
    assert catalogue.info("classWithComplex")[2] == (
        ("TStreamerObjectAny", "fcf", "", 62, 8, 0, 0, (0, 0, 0, 0, 0), "complex<float>", ()),
        ("TStreamerObjectAny", "fcd", "", 62, 16, 0, 0, (0, 0, 0, 0, 0), "complex<double>", ()),
    )
    # ROOT's checksum of each complex' description, which a file carries as its class version is
    # 1 - and of the class holding them, which is what its objects name instead of a version.
    described = {name: (hex(sum_), version)
                 for name, (sum_, version, _elements) in catalogue.entries().items()}  # fmt: skip
    assert described == {"complex<float>": ("0x2e5c4608", 1),
                        "complex<double>": ("0xdb2371a1", 1),
                        "classWithComplex": ("0xef61cc47", 1)}  # fmt: skip
    buf = WBuffer()
    write_instance(buf, Instance("classWithComplex", {
        "fcf": complex_type("float")(1, 2), "fcd": complex_type("double")(3, 4),
    }), catalogue)  # fmt: skip
    assert bytes(buf.data).hex() == COMPLEX_BYTES


def test_a_complex_in_an_array_behind_a_pointer_or_in_a_vector_is_roots_layout():
    """Every other shape ROOT has of one: ``kOffsetL + kAny`` for an array - three values'
    room and three as the length - the pointer's two spellings, and the container of one, which
    is written member-wise as ROOT writes a container of objects."""
    decl = Declaration("Shapes", 1, True, (), (
        Field("fcf", "std::complex<float>", "", ()),
        Field("fcd", "std::complex<double>", "", ()),
        Field("arr", "std::complex<double>", "", (3,)),
        Field("pnull", "std::complex<float>*", "", ()),
        Field("palways", "std::complex<double>*", "-> never null", ()),
        Field("vc", "std::vector<std::complex<double> >", "", ()),
    ))  # fmt: skip
    catalogue = Catalogue({"Shapes": decl})
    sum_, _version, elements = catalogue.info("Shapes")
    assert sum_ == 0xB0127892
    assert elements[2:5] == (
        ("TStreamerObjectAny", "arr", "", 82, 48, 3, 1, (3, 0, 0, 0, 0), "complex<double>", ()),
        ("TStreamerObjectAnyPointer", "pnull", "", 69, 8, 0, 0, NO_DIMS, "complex<float>*", ()),
        ("TStreamerObjectAnyPointer", "palways", "-> never null", 68, 8, 0, 0, NO_DIMS,
         "complex<double>*", ()),
    )
    assert elements[5][8:] == ("vector<complex<double> >", (1, 61))
    double = complex_type("double")
    buf = WBuffer()
    write_instance(buf, Instance("Shapes", {
        "fcf": complex_type("float")(1, 2), "fcd": double(3, 4),
        "arr": [double(10 + i, 20 + i) for i in range(3)],
        "pnull": None, "palways": double(5, 6), "vc": [double(7, 8), double(9, 10)],
    }), catalogue)  # fmt: skip
    assert bytes(buf.data).hex() == COMPLEX_SHAPES_BYTES


def test_a_vector_of_complex_is_described_whichever_member_comes_first():
    """A container of objects asks the catalogue for the layout of what it holds before the
    member's own rule is tried, so a ``vector<complex<T> >`` is described even when no member
    of the plain complex came first to describe it."""
    decl = Declaration("C", 1, True, (),
                       (Field("vc", "std::vector<std::complex<float> >", "", ()),))  # fmt: skip
    catalogue = Catalogue({"C": decl})
    assert catalogue.info("C")[2][0][8:] == ("vector<complex<float> >", (1, 61))
    assert catalogue.entries()["complex<float>"][:2] == (0x2E5C4608, 1)


def test_a_complex_named_on_its_own_is_written_as_root_writes_the_object():
    """``WriteObjectAny`` hands the writer a complex by its class name alone, with nothing
    holding one to describe it: the value writer describes it then, and writes the object
    ROOT does - its record, the version a class of no ``ClassDef`` carries and its checksum."""
    catalogue = Catalogue({})
    buf = WBuffer()
    write_value(buf, "complex<double>", complex_type("double")(3, 4), catalogue)
    assert bytes(buf.data).hex() == (
        "400000160000db2371a140080000000000004010000000000000"
    )
    assert catalogue.entries()["complex<double>"][:2] == (0xDB2371A1, 1)


def test_a_complex_of_a_width_this_writer_has_no_number_for_is_refused():
    """A complex is two numbers of one width, so one of a type that is not a number this
    writer packs - or written without its width at all - is refused rather than guessed at."""
    def catalogue(*fields):
        return Catalogue({"C": Declaration("C", 1, True, (), fields)})

    for spelled in ("std::complex<long double>", "std::complex<float*>", "std::complex"):
        with pytest.raises(UnsupportedFeatureError, match=re.escape(f"C::c is a {spelled}, which")):
            catalogue(Field("c", spelled, "", ())).info("C")


DOXYGEN = r'''
class Documented {
public:
   Int_t fPlain; ///< A plain persistent member.
   Int_t fDoxygen; //!< A persistent member, documented in doxygen's way.
   Int_t fCachePlain; ///<! A transient member.
   Int_t fCacheDoxygen; //!<! A transient member, documented in doxygen's way.
   Int_t fBang; //! A member marked transient the plain way;

   ClassDef(Documented, 1);
};
'''


def test_a_doxygen_trailing_comment_documents_a_member_it_does_not_hide_it():
    """ROOT reads ``///<`` and ``//!<`` as doxygen's trailing marker - the comment is the text
    after it, and the member is persistent - where ``///<!`` and ``//!<!`` mark it transient
    and keep the ``!`` their comment starts with. Checked against ROOT 6.40.04, whose file
    lists ``fPlain`` and ``fDoxygen`` with those titles and no other member.
    """
    from xrdroot.pyroot.core.declared import catalogue_of, declaration_of

    run_source(DOXYGEN, "doxygen.cxx", call=False)
    cls = find_class("Documented")
    fields = {field.name: field.comment for field in declaration_of(cls).fields}
    elements = catalogue_of(cls).info("Documented")[2]
    expect(
        (fields, {"fPlain": "A plain persistent member.",
                  "fDoxygen": "A persistent member, documented in doxygen's way.",
                  "fCachePlain": "! A transient member.",
                  "fCacheDoxygen": "! A transient member, documented in doxygen's way.",
                  "fBang": "! A member marked transient the plain way;"}),
        ([element[1] for element in elements], ["fPlain", "fDoxygen"]),
        ([element[2] for element in elements],
         ["A plain persistent member.", "A persistent member, documented in doxygen's way."]),
    )  # fmt: skip


def test_a_class_a_pointer_member_points_at_is_described_with_it():
    """A ``Base *f`` member may point at a ``Derived``, and ROOT's streamer record names every
    class it wrote; so a class deriving from one already described is described too, one step
    at a time down a chain of bases."""
    from xrdroot.pycxx.runtime.namespace import DECLARED
    from xrdroot.pyroot.core.declared import declaration_of
    from xrdroot.pyroot.trees.objects import _describe_pointed_at

    run_source(
        """
        class Root1 { public: int a = 1; };
        class Middle1 : public Root1 { public: int b = 2; };
        class Leaf1 : public Middle1 { public: int c = 3; };
        """, "pointed.cxx", call=False)  # fmt: skip
    catalogue = Catalogue({"Root1": declaration_of(find_class("Root1"))})
    _describe_pointed_at(catalogue)
    assert {"Root1", "Middle1", "Leaf1"} <= set(catalogue.declarations)
    assert DECLARED  # the classes stay declared, nothing is unregistered
