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


def test_the_length_of_a_std_array_must_be_a_comment_free_whole_number():
    """``std::array<int,n>`` of a length nothing can count is refused, and so is one written
    with stars or another template's arguments."""
    from xrdroot.wdeclared import _whole, parse_type

    assert _whole(parse_type("4")) == 4
    assert _whole(parse_type("std::vector<int>")) is None
    assert _whole(parse_type("T*")) is None
    assert _whole(parse_type("n")) is None  # a name, not a number
