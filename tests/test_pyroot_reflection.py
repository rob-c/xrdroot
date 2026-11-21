"""``TClass`` of the classes a macro declares and of ROOT's: versions, members, bases and
layouts as ROOT 6.40 reports them, ``TBufferFile`` round trips, and collection proxies.

Each listing below is what ROOT 6.40.04 printed for the same classes, compiled
by ACLiC from ``declared_objects.C``, with the offsets ROOT's compiler chose
left out, as roottest's own comparisons leave them out.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot import streamers
from xrdroot.pycint.execute import run_source
from xrdroot.pycxx.runtime.namespace import DECLARED
from xrdroot.pyroot.core import reflection, troot

DATA = pathlib.Path(__file__).parent / "data"
SOURCE = (DATA / "declared_objects.C").read_text()


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    declared = dict(DECLARED)
    infos = {name: list(found) for name, found in reflection.FILE_INFOS.items()}
    enums = dict(troot._ENUMS)
    troot._ENUMS.clear()
    yield from fresh(tmp_path)
    DECLARED.clear()
    DECLARED.update(declared)
    reflection.FILE_INFOS.clear()
    reflection.FILE_INFOS.update(infos)
    reflection.DISAGREED.clear()
    reflection.GENERATED.clear()
    reflection.INTERPRETED_METHODS.clear()
    troot._ENUMS.clear()
    troot._ENUMS.update(enums)


def _names(cls):
    return {member.GetName() for member in cls.GetListOfAllPublicDataMembers()}


def _declared():
    run_source(SOURCE, "declared_objects.C", call=False)


def _listing(text):
    """A listing with the offsets left out, as roottest leaves them out, and each line's padding
    after the title - twenty characters of it, which the test that prints it holds to."""
    assert all(
        len(line) >= 59 for line in text.splitlines() if " type=" in line and "i=" not in line
    )
    return "\n".join(
        line.rstrip() for line in re.sub(r"offset=\s*\d+", "offset=na", text).split("\n")
    )


NAMED = """
StreamerInfo for class: Named, version=2, checksum=0xa3400129
  TNamed         BASE            offset=na type=67 The basis for a named object (name, title)
  int            fN              offset=na type= 3
   i= 0, TNamed          type= 67, offset=na, len=1, method=0
   i= 1, fN              type=  3, offset=na, len=1, method=0
"""

DERIVED = """
StreamerInfo for class: Derived, checksum=0xb0a379ae
  Base1          BASE            offset=na type= 0
  ns::Deep       deep            offset=na type=62
  int            arr2[2][3]      offset=na type=23
  vector<vector<float> > vvf             offset=na type=300 stl=1 ctype=61
  vector<Inner*> vpinner         offset=na type=300 stl=1 ctype=63
"""


def test_a_declared_classs_layout_lists_as_roots_does(capsys):
    _declared()
    ROOT.TClass.GetClass("Named").GetStreamerInfo().ls()
    ROOT.TClass.GetClass("Derived").GetStreamerInfo().ls()
    printed = _listing(capsys.readouterr().out)
    assert printed.startswith(NAMED)
    assert DERIVED in printed


def test_a_class_reports_its_version_checksum_and_kind_as_root_does():
    _declared()
    found = {name: ROOT.TClass.GetClass(name) for name in ("Base1", "Versioned", "Named")}
    named = found["Named"]
    expect(
        ([c.GetClassVersion() for c in found.values()], [-1, 3, 2]),
        ([c.IsForeign() for c in found.values()], [True, False, False]),
        ([c.IsTObject() for c in found.values()], [False, False, True]),
        ((named.GetCheckSum(), named.GetState()), (0xA3400129, 4)),
        ((named.InheritsFrom("TObject"), named.IsLoaded()), (True, True)),
        ((ROOT.TClass.GetClass("int"), ROOT.TClass.GetClass("Some").GetName()), (None, "Some")),
        (ROOT.gROOT.GetClass("vector<int>").GetName(), "vector<int>"),
        (ROOT.TClass.GetClass(ROOT.TH1F).GetName(), "TH1F"),
        (ROOT.TClass.GetClass(named) is named, True),
    )


def test_data_members_are_described_as_the_dictionary_describes_them():
    _declared()
    plain = ROOT.TClass.GetClass("Plain")
    members = {m.GetName(): m for m in plain.GetListOfDataMembers()}
    ll, var, trans, mid, arr = (members[name] for name in ("ll", "var", "trans", "mid", "arr"))
    expect(
        ((ll.GetTypeName(), ll.GetTrueTypeName(), ll.IsBasic()), ("Long64_t", "Long64_t", True)),
        ((var.GetTypeName(), var.GetFullTypeName(), var.IsaPointer()), ("double", "double*", True)),
        ((var.GetArrayIndex(), var.GetUnitSize(), var.GetOffset()), ("n", 8, 0)),
        ((trans.IsPersistent(), trans.GetComment()), (False, "!")),
        (
            (mid.GetFullTypeName(), mid.IsSTLContainer(), mid.IsBasic()),
            ("map<int,double>", 4, False),
        ),
        (
            (arr.GetArrayDim(), arr.GetMaxIndex(0), arr.GetMaxIndex(1), arr.GetUnitSize()),
            (1, 3, -1, 4),
        ),
        ((arr.IsEnum(), arr.GetClass()), (False, plain)),
        (plain.GetDataMember("arr[3]").GetName(), "arr"),
        (plain.GetDataMember("nothing"), None),
    )


#: A class template, whose instance a macro's linkdef asks for by the spelling it was written
#: with - ``Holder<Double32_t>`` - while the declaration settles on the typedef's type.
HOLDER = """
template <typename T>
class Holder { public: T value; double reg; };
Holder<Double32_t> held;
"""


def test_a_template_instance_is_found_by_the_double32_spelling_its_linkdef_asked_for():
    """roottest's ``io/double32`` asks ``TClass::GetClass("MyVector<Double32_t>")`` for the
    class a ``#pragma link C++ class MyVector<Double32_t>+`` selected, and ROOT answers it:
    the dictionary was generated under that spelling. The translation makes ``Double32_t`` a
    ``double``, so the class here is ``Holder<double>``, and the typedef's spelling has to
    find it or the class is one nothing describes and ``GetStreamerInfo`` is ``nullptr``.
    """
    run_source(HOLDER, "holder.C", call=False)
    found = ROOT.TClass.GetClass("Holder<Double32_t>")
    members = found.GetListOfDataMembers()
    expect(
        (found.GetName(), "Holder<Double32_t>"),
        (found.HasDictionary(), True),
        (found.GetState(), reflection.K_HAS_TCLASS_INIT),
        (found.GetStreamerInfo() is not None, True),
        ([(m.GetName(), m.GetTypeName()) for m in members],
         [("value", "Double32_t"), ("reg", "double")]),
    )  # fmt: skip
    assert found.GetStreamerInfo().GetCheckSum() == ROOT.TClass.GetClass(
        "Holder<double>"
    ).GetCheckSum()


def test_a_name_that_is_not_a_template_instance_finds_nothing_by_its_plain_spelling():
    """Only a ``Double32_t``/``Float16_t`` argument is looked for under the type it spells, so
    a name nothing declares is still a class with no dictionary rather than a cousin's."""
    _declared()
    assert reflection._plain_spelling("Named") == "Named"
    expect(
        (reflection._plain_spelling("Holder<Float16_t>"), "Holder<float>"),
        (ROOT.TClass.GetClass("Holder<Float16_t>").GetStreamerInfo(), None),
        (ROOT.TClass.GetClass("Nothing<Double32_t>").HasDictionary(), False),
    )


def test_bases_are_listed_and_found_at_any_depth():
    _declared()
    named = ROOT.TClass.GetClass("Named")
    bases = list(named.GetListOfBases())
    expect(
        ([b.GetName() for b in bases], ["TNamed"]),
        (bases[0].GetClassPointer().GetName(), "TNamed"),
        ((bases[0].GetDelta(), bases[0].IsSTLContainer()), (0, 0)),
        (named.GetBaseClass("TObject").GetName(), "TObject"),
        (named.GetBaseClass("TH1"), None),
        ({"fN", "fName", "fUniqueID"} <= _names(named),
         True),
    )  # fmt: skip


def test_a_layout_a_file_holds_is_listed_beside_the_one_in_memory(capsys):
    _declared()
    ROOT.TFile(str(DATA / "declared-objects-6.40.root"))
    infos = ROOT.TClass.GetClass("Derived").GetStreamerInfos()
    assert infos.GetEntries() == 1  # the file's layout is the one in memory
    pair = ROOT.TClass.GetClass("pair<int,string>")
    assert pair.GetStreamerInfo(1).GetCheckSum() == 0xB5FB752
    assert pair.FindStreamerInfo(0xB5FB752).GetClassVersion() == 1
    assert pair.GetStreamerInfo(7) is None and pair.FindStreamerInfo(1) is None
    pair.GetStreamerInfos().ls()
    printed = capsys.readouterr().out
    assert printed.startswith("OBJ: TObjArray\tTObjArray\tAn array of objects : 0\n")
    assert (
        "string         second          offset=  0 type=300 stl=365 ctype=365 Emulation" in printed
    )
    assert "   i=" not in printed


def test_a_class_only_files_describe_is_built_from_the_first_layout_read(capfd):
    """ROOT emulates a class nothing in memory describes from the first layout a file read
    gave it, and ``GetStreamerInfo`` hands that one back built - each element listed with the
    action it is read with - and a later file's layout still by its version. A listing of
    the layouts as the file holds them stays unbuilt, so it names no action."""
    _evo("int", "one.root")
    _evo("float", "two.root")
    ROOT.TFile("one.root")
    ROOT.TFile("two.root")
    evo = ROOT.TClass.GetClass("Evo")
    current = evo.GetStreamerInfo()
    assert (current.IsBuilt(), current.GetClassVersion()) == (True, 1)
    assert current.GetElement(0).GetTypeName() == "int"
    assert evo.GetStreamerInfo(2).GetElement(0).GetTypeName() == "float"
    assert evo.GetStreamerInfo(2).IsBuilt() and evo.GetStreamerInfo(3) is None
    current.ls()
    printed = capfd.readouterr().out
    # A class only a file describes is versioned: ROOT prints the version the file gave it.
    assert "StreamerInfo for class: Evo, version=1, checksum=" in printed
    assert "   i= 0, a               type=  3, offset=  0, len=1, method=0" in printed


def test_a_class_only_files_describe_takes_the_version_its_files_give_it():
    """``TClass::GetClassVersion`` answers the version the file gave a class nothing in memory
    describes - ``roottest``'s ``io/tree/read`` unloads ``classes_C`` and asks what its file
    records for ``TNonPhysObj``, which is version 0 - where an in-memory answer would be no
    answer at all. A class the session does describe keeps the version it is declared with.
    """
    _declared()
    reflection.note_file_infos([("FromFile", (0x2468, 0, (LONG64,)))], "one.root")
    reflection.note_file_infos([("Later", (0x9876, 7, (LONG64,)))], "two.root")
    reflection.note_file_infos([("Named", (0x1357, 9, (LONG64,)))], "three.root")
    from_file = ROOT.TClass.GetClass("FromFile")
    expect(
        ((from_file.GetClassVersion(), from_file.IsForeign()), (0, False)),
        (from_file.GetState(), 2),
        (ROOT.TClass.GetClass("Later").GetClassVersion(), 7),
        (ROOT.TClass.GetClass("Named").GetClassVersion(), 2),  # its own, not the file's 9
        (ROOT.TClass("TNoSuch").GetClassVersion(), 1),  # no file describes it
    )


def test_a_pair_names_no_version_where_a_plain_class_names_the_files_own(capsys):
    """ROOT tells the layouts of a ``pair<A,B>`` apart by checksum - it is never given a
    ``ClassDef`` - so a listing of one names no version, and the emulated layout is built."""
    ROOT.TFile(str(DATA / "declared-objects-6.40.root"))
    pair = ROOT.TClass.GetClass("pair<int,string>")
    info = pair.GetStreamerInfo()
    assert info is not None and info.IsBuilt()
    info.ls()
    printed = capsys.readouterr().out
    assert "StreamerInfo for class: pair<int,string>, checksum=" in printed
    assert "version=" not in printed and "   i= 0, first" in printed
    found = pair.FindStreamerInfo(info.GetCheckSum())
    assert (found.GetCheckSum(), found.IsBuilt()) == (info.GetCheckSum(), False)
    found.ls()
    printed = capsys.readouterr().out
    assert "version=" not in printed and "   i=" not in printed


def test_a_streamer_info_hands_out_its_elements_by_their_kinds():
    _declared()
    info = ROOT.TClass.GetClass("Plain").GetStreamerInfo()
    kinds = {e.GetName(): e.ClassName() for e in info.GetElements()}
    vi, arr = info.GetElement(19), info.GetElement(14)
    expect(
        ((kinds["var"], kinds["str"]), ("TStreamerBasicPointer", "TStreamerSTLstring")),
        ((info.GetElement(16).GetCountName(), info.GetElement(99)), ("n", None)),
        ((vi.GetSTLtype(), vi.GetCtype(), vi.GetSize()), (1, 3, 24)),
        (vi.GetTypeNameBasic(), "vector<int>"),
        ((arr.GetFullName(), arr.GetArrayLength(), arr.GetArrayDim()), ("arr[3]", 3, 1)),
        ((arr.GetMaxIndex(0), arr.GetMaxIndex(9)), (3, 0)),
        ((arr.IsBase(), arr.IsaPointer()), (False, False)),
        (info.GetElement(24).GetClassPointer().GetName(), "Inner"),
        ((info.GetNelement(), info.IsBuilt(), info.IsCompiled()), (25, True, True)),
        ((info.GetClass().GetName(), info.GetClassVersion()), ("Plain", 1)),
    )  # fmt: skip
    base = ROOT.TClass.GetClass("Named").GetStreamerInfo().GetElement(0)
    empty = ROOT.TStreamerInfo()
    empty.Build()
    expect(
        ((base.IsBase(), base.GetBaseVersion()), (True, 1)),
        ((empty.IsBuilt(), ROOT.TStreamerElement().GetType()), (True, 0)),
    )


def test_a_class_is_made_and_unmade_and_its_methods_listed():
    _declared()
    named = ROOT.TClass.GetClass("Named")
    made = named.New()
    named.Destructor(made)
    named.Destructor(ROOT.TH1F("h", "h", 1, 0, 1))
    unknown = ROOT.TClass("TNoSuch")
    expect(
        ((made.ClassName(), named.GetActualClass(made)), ("Named", named)),
        (ROOT.TClass.GetClass("Plain").GetActualClass(3).GetName(), "Plain"),
        ((named.GetBaseClassOffset("TNamed"), named.GetBaseClassOffset("TH1")), (0, -1)),
        (named.GetMethodAny("Class_Name").GetName(), "Class_Name"),
        (named.GetMethodAllAny("Nope"), None),
        ((named.Size() > 0, named.GetClassSize() == named.Size(), named.CanSplit()), (True,) * 3),
        ((named.GetDeclFileName(), named.GetImplFileName(), named.GetDeclFileLine()), ("", "", -1)),
        ((named.Property(), ROOT.TClass("A").Property()), (ROOT.TClass.kIsTObject, 0)),
        ((unknown.GetState(), unknown.GetCheckSum(), unknown.GetClassVersion()), (2, 0, 1)),
        (list(unknown.GetListOfMethods()), []),
        ([b.GetName() for b in ROOT.TClass("TList").GetListOfBases()], ["TSeqCollection"]),
        (ROOT.TClass.GetClass("TH1F").GetStreamerInfo().GetClassVersion(), 3),
    )  # fmt: skip


def test_a_container_is_interpreted_until_a_dictionary_is_generated_for_it():
    """ROOT's ``ReuseMethod``: a standard container is a class the interpreter describes, so
    ``kInterpreted`` - 3, ROOT's number for it - until ``GenerateDictionary`` gives it one, and
    then ``kHasTClassInit``. The ``TMethod`` found before the dictionary is still the list's."""
    container = ROOT.TClass.GetClass("vector<TObject*>")
    size = container.GetMethodWithPrototype("size", "")
    assert (container.GetState(), ROOT.TClass.kInterpreted) == (3, 3)
    assert ROOT.gInterpreter.GenerateDictionary("vector<TObject*>") == 0
    again = ROOT.TClass.GetClass("vector<TObject*>")
    expect(
        (again.GetState(), ROOT.TClass.kHasTClassInit),
        (again.GetState(), ROOT.TClass.kLoaded),
        ([m.GetName() for m in again.GetListOfMethods(False)], ["size"]),
        (again.GetListOfMethods(False).At(0) is size, True),
        ((ROOT.TClass.GetClass("TH1F").GetState(), ROOT.TClass("TNoSuch").GetState()), (4, 2)),
    )  # fmt: skip


def test_a_containers_method_list_holds_the_functions_looked_up_in_it():
    """ROOT lists none of a class's functions until something asks for one, and the list holds
    what the interpreter was asked to make from then on. Only a member a container declares
    with no argument is here to be found."""
    container = ROOT.TClass.GetClass("vector<int>")
    before = list(container.GetListOfMethods(False))
    found = container.GetMethodWithPrototype("size", "")
    expect(
        (before, []),
        ([m.GetName() for m in container.GetListOfMethods(False)], ["size"]),
        (container.GetMethodWithPrototype("size", "") is found, True),
        (container.GetMethodAny("size") is found, True),
        (container.GetMethodWithPrototype("size", "int"), None),
        (container.GetMethodWithPrototype("push_back", ""), None),
        (container.GetMethodWithPrototype("grow", ""), None),
    )


def test_a_declared_classs_method_is_found_by_name_and_prototype():
    _declared()
    named = ROOT.TClass.GetClass("Named")
    expect(
        (named.GetMethodWithPrototype("Class_Name", "").GetName(), "Class_Name"),
        (named.GetMethodWithPrototype("Class_Name", "int"), None),
        (named.GetMethodWithPrototype("NoSuchMethod", ""), None),
    )


def test_a_dictionary_generated_for_several_classes_is_noted_for_each():
    assert ROOT.gInterpreter.GenerateDictionary("vector<int>; std::vector<double>; ") == 0
    expect(
        (ROOT.TClass.GetClass("vector<int>").GetState(), ROOT.TClass.kHasTClassInit),
        (ROOT.TClass.GetClass("std::vector<double>").GetState(), ROOT.TClass.kHasTClassInit),
        (ROOT.TClass.GetClass("vector<float>").GetState(), ROOT.TClass.kInterpreted),
    )


def test_the_class_table_answers_for_the_classes_a_dictionary_describes():
    # A program asks this to decide whether it must load a library first, so a class of
    # ROOT's answers by its qualified name too, and one nobody describes does not.
    assert ROOT.TClassTable.GetDict("TH1F").GetName() == "TH1F"
    assert ROOT.TClassTable.GetDictAny("ROOT::TNamed").GetName() == "TNamed"
    assert ROOT.gClassTable.GetDict("TNamed").GetName() == "TNamed"
    assert ROOT.TClassTable.GetDict("NoSuchClassIsHere") is None
    assert ROOT.TClassTable.GetDict("TH1F").IsLoaded()
    ROOT.TClassTable.Add("NotedByHand", 1)  # a dictionary's class, noted as one from now on
    assert ROOT.TClassTable.GetDict("NotedByHand").GetName() == "NotedByHand"


def test_the_class_table_lists_the_classes_it_holds(capsys):
    ROOT.TClassTable.Add("NotedByHand", 1)
    names = ROOT.TClassTable.GetTable()
    assert names == sorted(names) and "TH1F" in names and "NotedByHand" in names
    assert ROOT.TClassTable.Classes() == len(names)
    assert ROOT.TClassTable.At(0) == names[0]
    assert ROOT.gClassTable.GetName() == "TClassTable"
    ROOT.TClassTable.Print("TH1F")
    assert "TH1F\n" in capsys.readouterr().out


def test_tvirtualstreamerinfos_switches_are_kept():
    previous = ROOT.TVirtualStreamerInfo.SetStreamMemberWise(False)
    assert previous and not ROOT.TVirtualStreamerInfo.GetStreamMemberWise()
    ROOT.TVirtualStreamerInfo.SetStreamMemberWise(True)
    ROOT.TVirtualStreamerInfo.SetCanDelete(False)
    ROOT.TVirtualStreamerInfo.SetFactory(None)


BUFFERS = r"""
#include "TBufferFile.h"
#include <vector>
#include <map>
class Hits {
public:
   int n = 1;
   float x = 2;
   std::vector<double> v;
   std::map<int, int> m;
};
void buffers() {
   std::vector<int> v{1, 2, 3, 4, 5, 6, 7};
   TBufferFile *buff = new TBufferFile(TBuffer::kWrite);
   buff->WriteObjectAny(&v, TClass::GetClass("vector<int>"));
   buff->WriteObjectAny(nullptr, TClass::GetClass("vector<int>"));
   buff->SetReadMode();
   buff->Reset();
   void *read = buff->ReadObjectAny(TClass::GetClass("vector<float>"));
   std::vector<float> &w = *reinterpret_cast<std::vector<float> *>(read);
   void *none = buff->ReadObjectAny(TClass::GetClass("vector<int>"));
   printf("%g %g %d %d\n", w[0], w[6], (int)w.size(), none == nullptr);

   TBufferFile b(TBuffer::kWrite, 32000);
   Hits a;
   a.v.push_back(3.5);
   a.m[1] = 2;
   TClass *cl = TClass::GetClass(typeid(a));
   for (int i = 0; i < 3; ++i) cl->Streamer(&a, b);
   printf("%d %d %d\n", b.Length(), b.IsWriting(), b.BufferSize() >= b.Length());
   b.SetReadMode();
   b.Reset();
   Hits c;
   c.n = 0;
   cl->Streamer(&c, b);
   printf("%d %g %d %g %d %d\n", c.n, c.x, (int)c.v.size(), c.v[0], c.m[1], b.IsReading());
}
"""


def test_objects_go_into_a_tbufferfile_and_come_back_as_the_class_asked_for(capfd):
    run_source(BUFFERS, "buffers.C")
    assert capfd.readouterr().out.splitlines() == ["1 7 7 1", "180 1 1", "1 2 1 3.5 2 1"]


def test_a_tbufferfile_reads_and_writes_numbers_strings_and_objects_of_roots():
    buf = ROOT.TBufferFile(ROOT.TBuffer.kWrite)
    buf.WriteInt(7)
    buf.WriteDouble(2.5)
    buf.WriteString("hello")
    buf.WriteTString("x" * 300)
    named = ROOT.TNamed("n", "t")
    buf.WriteObject(named)
    assert buf.Length() == len(buf.Buffer()) and buf.GetBufferOffset() == buf.Length()
    buf.SetReadMode()
    buf.SetBufferOffset(0)
    assert (buf.ReadInt(), buf.ReadDouble(), buf.ReadString(), len(buf.ReadTString())) == (
        7,
        2.5,
        "hello",
        300,
    )
    read = buf.ReadObject(ROOT.TNamed.Class())
    assert (read.GetName(), read.GetTitle()) == ("n", "t")
    with pytest.raises(AttributeError, match="TBuffer has WriteShout"):
        buf.WriteShout(1)
    buf.SetWriteMode()
    buf.Reset()
    buf.StreamObject(named, "TNamed")
    assert buf.Length() > 0


def test_a_collection_proxy_says_what_a_container_is_and_holds():
    proxy = ROOT.TClass.GetClass("vector<int>").GetCollectionProxy()
    v = ROOT.std.vector["int"]()
    v.push_back(4)
    proxy.PushProxy(v)
    expect(
        ((proxy.GetCollectionType(), proxy.GetType(), proxy.HasPointers()), (1, 3, False)),
        ((proxy.GetValueClass(), proxy.GetCollectionClass().GetName()), (None, "vector<int>")),
        ((proxy.GetProperties(), proxy.Sizeof()), (proxy.kIsInitialized, 24)),
        (proxy.GetIncrement(), 8),
        ((proxy.Size(), proxy.At(0), proxy.At(3)), (1, 4, None)),
    )  # fmt: skip
    proxy.Clear()
    proxy.PopProxy()
    proxy.PopProxy()
    expect((proxy.Size(), 0), (v.size(), 0))


def test_a_proxy_of_a_map_holds_pairs_and_one_of_pointers_says_so():
    mapped = ROOT.TClass.GetClass("map<int,double>").GetCollectionProxy().Generate()
    m = ROOT.std.map["int", "double"]()
    m[1] = 2.0
    mapped.PushProxy(m)
    pointers = ROOT.TVirtualCollectionProxy("vector<TH1F*>")
    bare = ROOT.TVirtualCollectionProxy()
    expect(
        (mapped.GetValueClass().GetName(), "pair<int,double>"),
        (bool(mapped.GetProperties() & mapped.kIsAssociative), True),
        ((mapped.GetType(), mapped.At(0)), (ROOT.EDataType.kInt_t, (1, 2.0))),
        (mapped.New().size(), 0),
        ((pointers.HasPointers(), pointers.GetValueClass().GetName()), (True, "TH1F")),
        ((bare.GetCollectionType(), bare.GetType()), (0, ROOT.EDataType.kOther_t)),
        (bare.New(), None),
        (ROOT.TClass.GetClass("TH1F").GetCollectionProxy(), None),
    )  # fmt: skip


VERSIONS = r"""
class Evo {
public:
   %s a = 1;
};
void evo(const char *name) {
   TFile f(name, "RECREATE");
   Evo e;
   f.WriteObject(&e, "e");
}
"""


def _evo(kind, name):
    run_source(VERSIONS % kind, "evo.C", (name,))
    DECLARED.pop("Evo")


VERSED = """
class Versed {
public:
   int a = 1;
   ClassDef(Versed, 1);
};
"""

#: A file's layout of that member, as ``TStreamerInfo`` records one - its kind, name, title,
#: ROOT's type code, its width, and the array bookkeeping after them - recorded at eight bytes.
LONG64 = ("TStreamerBasicType", "a", "", 16, 8, 1, 0, (0, 0, 0, 0, 0), "Long64_t", ())
#: The same member spelled ``Int_t`` at four bytes, under a type code today's ROOT no longer
#: writes for it: what an old file records, and not a difference.
AS_WIDE = ("TStreamerBasicType", "a", "", 6, 4, 1, 0, (0, 0, 0, 0, 0), "Int_t", ())


def test_a_file_layout_differing_from_the_one_in_memory_names_the_file(capfd):
    """ROOT's ``TStreamerInfo::BuildCheck``, which names the file when a layout of a version
    the class has in memory records a member at another width, and ``CompareContent``, which
    sets that member's two declarations out - once for the class, not once for each file."""
    run_source(VERSED, "versed.C", call=False)
    reflection.note_file_infos([("Versed", (0x1234567, 1, (LONG64,)))], "one.root")
    reflection.note_file_infos([("Versed", (0x7654321, 1, (LONG64,)))], "two.root")
    printed = capfd.readouterr().err
    assert printed.count("Warning in <TStreamerInfo::BuildCheck>:") == 1
    assert (
        "   The StreamerInfo for version 1 of class Versed read from the file one.root\n"
        "   has a different checksum than the previously loaded StreamerInfo." in printed
    )
    assert (
        "Warning in <TStreamerInfo::CompareContent>: The following data member of\n"
        "the on-file layout version 1 of class 'Versed' differs from\n"
        "the in-memory layout version 1:\n   Long64_t a; //\nvs\n   int a; //" in printed
    )


def test_a_member_as_wide_either_way_is_not_reported(capfd):
    """A number is compared by its width, so a member the file records at the width the class
    in memory gives it is read across and ROOT says nothing - an old file spells ``Int_t``
    under a type code today's ROOT no longer writes for it, and ``io/evolution/ClonesArrayEvo``
    reads such a file and prints nothing - and a class with no layout in memory is not
    reported either."""
    run_source(VERSED, "versed.C", call=False)
    reflection.note_file_infos([("Versed", (0x1234, 1, (AS_WIDE,)))], "one.root")
    assert "BuildCheck" not in capfd.readouterr().err
    reflection.note_file_infos([("Elsewhere", (0x5678, 1, (LONG64,)))], "one.root")
    assert "BuildCheck" not in capfd.readouterr().err


def test_a_layout_whose_file_is_unknown_names_no_file(capfd, monkeypatch):
    """``BuildCheck`` names the file a layout came from, so a layout noted when nothing says
    which file it was read from is answered with the members alone."""
    monkeypatch.setattr(streamers, "LAST_SOURCE", "")
    run_source(VERSED, "versed.C", call=False)
    reflection.note_file_infos([("Versed", (0x1234, 1, (LONG64,)))])
    printed = capfd.readouterr().err
    assert "BuildCheck" not in printed and "   Long64_t a; //" in printed


def test_the_last_file_read_names_the_layout_it_described(capfd, monkeypatch):
    """Opening a file notes its layouts without naming it, so the name the warning gives is
    the one :mod:`xrdroot.streamers` read those layouts from."""
    monkeypatch.setattr(streamers, "LAST_SOURCE", "read.root")
    run_source(VERSED, "versed.C", call=False)
    reflection.note_file_infos([("Versed", (0x1234, 1, (LONG64,)))])
    assert "class Versed read from the file read.root" in capfd.readouterr().err


def test_a_files_pair_base_is_no_missing_dictionary(capfd):
    """``__pair_base<A,B>`` is not a class a program can ask for - ROOT's ``TClass::Init``
    skips the warning for a ``pair``'s base - so the file ``roottest/root/io/stdpair`` reads,
    which describes one beside its pair, opens without a word about it."""
    reflection.note_file_infos(
        [("__pair_base<edm::Hash<1>,edm::ParameterSetBlob>", (0x2829819F, 1, ()))]
    )
    assert capfd.readouterr().err == ""
    reflection.note_file_infos([("std::__pair_base<int,int>", (0x2829819F, 1, ()))])
    assert capfd.readouterr().err == ""


def test_a_files_instance_of_a_dictionarys_template_needs_no_dictionary(capfd):
    """ROOT asks the interpreter for the class of a name a file describes, and the interpreter
    answers a template's instance from the template: with ``Ref``'s dictionary in the session,
    ``edm::Ref<vector<int>>`` is a type cling can declare, so ``TClass::Init`` makes no
    complaint about it - and without that dictionary it does, as ``pairEnumEvo``'s runs that
    read the file before loading the library print."""
    described = [("edm::Ref<vector<int> >", (0x251EA0D5, 11, ()))]
    reflection.note_file_infos(described)
    assert (
        "no dictionary for class edm::Ref<vector<int> > is available"
        in capfd.readouterr().err
    )
    ROOT.TClassTable.Add("Ref", 11)
    reflection.note_file_infos(described)
    assert capfd.readouterr().err == ""


def test_a_name_that_is_no_instance_of_anything_needs_its_dictionary(capfd):
    """``<`` alone is no template's instance - a name without one, or whose head is empty, is
    judged by its own dictionary alone."""
    reflection.note_file_infos([("Nowhere", (0x1, 1, ())), ("<odd>", (0x2, 1, ()))])
    printed = capfd.readouterr().err
    assert "no dictionary for class Nowhere is available" in printed
    assert "no dictionary for class <odd> is available" in printed


def test_a_user_class_named_like_roots_own_needs_its_dictionary(capfd):
    """A name's spelling says nothing about whose class it is: ``roottest/root/io/tree``'s
    ``missingClass`` reads a file describing ``TEmcl``, ``TPhysObj``, ``TNonEmcl`` and
    ``TNonPhysObj``, a program's own classes all four, and ROOT warns of each beside the
    ``TTree`` its file describes without a word."""
    reflection.note_file_infos([("TEmcl", (0x7D5A2E11, 1, ()))])
    assert (
        "Warning in <TClass::Init>: no dictionary for class TEmcl is available"
        in capfd.readouterr().err
    )


def test_a_root_class_the_kit_streams_needs_no_dictionary(capfd):
    """ROOT's own classes have the dictionaries its libraries were built with, and the kit
    knows them by what it streams: the layout ``TH1F`` is written with, ``TBasket``'s reader,
    ``THnSparseD``'s branches and ``TRef``'s."""
    described = [(name, (index, 1, ())) for index, name in enumerate(
        ("TH1F", "TBasket", "THnSparseD", "TRef"))]
    reflection.note_file_infos(described)
    assert capfd.readouterr().err == ""


def test_a_root_class_the_kit_implements_without_a_tobject_needs_no_dictionary(capfd):
    """``TStyle``, ``TCanvas`` and ``TColor`` are ROOT's own classes and have their
    dictionaries, though the namespace's implementations of them derive from no ``TObject``
    - a file describing one opens as quietly as ROOT opens it."""
    reflection.note_file_infos([("TStyle", (0x11, 1, ())), ("TColor", (0x12, 1, ()))])
    assert capfd.readouterr().err == ""


def test_files_layouts_are_numbered_as_met_while_no_class_is_in_memory(capfd):
    _evo("int", "one.root")
    _evo("float", "two.root")
    ROOT.TFile("one.root")
    ROOT.TFile("two.root")
    ROOT.TFile("two.root")
    evo = ROOT.TClass.GetClass("Evo")
    infos = list(evo.GetStreamerInfos())
    assert [(i.GetClassVersion(), i.GetElement(0).GetTypeName()) for i in infos] == [
        (1, "int"),
        (2, "float"),
    ]
    assert evo.GetStreamerInfo(2).GetElement(0).GetTypeName() == "float"
    assert (
        "Warning in <TClass::Init>: no dictionary for class Evo is available"
        in capfd.readouterr().err
    )
    run_source(VERSIONS % "float", "evo.C", call=False)
    evo.GetStreamerInfos().ls()
    printed = capfd.readouterr().out
    assert (
        printed.count("StreamerInfo for class: Evo, checksum=") == 3 and "version=" not in printed
    )
    assert evo.FindStreamerInfo(evo.GetCheckSum()).IsBuilt()


def test_a_versioned_classs_layouts_are_listed_by_version():
    """ROOT lists a versioned class's layouts in the order of their versions, which is not the
    order the files were read in: ``roottest``'s ``AddVersion-1-4`` reads a file holding
    versions 1 and 3 beside a library at version 2 and prints them 1, 2, 3. A class told apart
    by checksum has no version to be sorted by, so it keeps the order it met them in - and the
    layout in memory stays built where a file's does not."""
    _evo("int", "one.root")
    _evo("float", "two.root")
    ROOT.TFile("one.root")
    ROOT.TFile("two.root")
    run_source(
        VERSIONS.replace("public:", "public: ClassDef(Evo, 3)") % "double", "evo.C", call=False
    )
    infos = list(ROOT.TClass.GetClass("Evo").GetStreamerInfos())
    assert [(i.GetClassVersion(), i.GetElement(0).GetTypeName()) for i in infos] == [
        (1, "int"),
        (2, "float"),
        (3, "double"),
    ]
    assert [i.IsBuilt() for i in infos] == [False, False, True]


def test_a_layout_the_class_in_memory_has_is_not_noted_again():
    run_source(
        VERSIONS.replace("public:", "public: ClassDef(Evo, 1)") % "short", "evo.C", ("v.root",)
    )
    ROOT.TFile("v.root")
    assert ROOT.TClass.GetClass("Evo").GetStreamerInfos().GetEntries() == 1


def test_a_files_description_that_cannot_be_read_notes_nothing(monkeypatch):
    from xrdroot.errors import FormatError
    from xrdroot.pyroot.core import files

    def broken(source):
        raise FormatError("no")

    monkeypatch.setattr("xrdroot.streamers.read_infos", broken)
    files._note_layouts(type("Reading", (), {"_source": None})())


def test_a_file_without_a_description_describes_nothing():
    from xrdroot.streamers import read_infos

    assert read_infos(type("Source", (), {"info": (0, 0)})()) == []


def test_elements_answer_for_their_type_and_place_and_objects_are_destroyed():
    element = ROOT.TStreamerElement(
        ("TStreamerBasicType", "x", "", 3, 4, 0, 0, (0,) * 5, "int", ())
    )
    assert (element.GetTypeName(), element.GetOffset()) == ("int", 0)
    closed = []
    held = type("Held", (), {"_destruct": lambda self: closed.append(1)})()
    ROOT.TClass("Held").Destructor(held)
    assert closed == [1]


def test_a_value_of_a_type_nothing_makes_is_none_and_a_proxy_of_nothing_clears_nothing():
    from xrdroot.pyroot.core.declared import new_value

    assert new_value("NoSuchClassAtAll") is None and new_value("TNamed").GetName() == ""
    proxy = ROOT.TVirtualCollectionProxy("vector<int>")
    proxy.Clear()
    proxy.PushProxy((1, 2))
    proxy.Clear()
    assert proxy.Size() == 2


MAPS = r"""
#include <map>
#include "TBufferFile.h"
class Leaf : public TObject {
public:
   int v = 3;
   ClassDefOverride(Leaf, 1);
};
class Tree {
public:
   std::map<std::string, Leaf*> leaves;
   std::map<int, std::string> empty;
};
void maps(const char *name) {
   Tree t;
   t.leaves["a"] = new Leaf();
   TFile f(name, "RECREATE");
   f.WriteObject(&t, "t");
   TBufferFile b(TBuffer::kWrite);
   std::map<int, int> m;
   m[1] = 2;
   b.WriteObjectAny(&m, TClass::GetClass("map<int,int>"));
   b.SetReadMode();
   b.Reset();
   void *read = b.ReadObjectAny(TClass::GetClass("map<int,int>"));
   std::map<int, int> *back = (std::map<int, int> *)read;
   printf("%d\n", (*back)[1]);
}
"""


def test_a_map_of_pointers_to_objects_goes_out_and_comes_back(capfd):
    run_source(MAPS, "maps.C", ("maps.root",))
    assert capfd.readouterr().out == "2\n"
    tree = ROOT.TFile("maps.root").Get("t")
    assert tree.leaves["a"].v == 3 and tree.empty.size() == 0


def test_cling_s_optimisation_level_is_no_command_to_refuse():
    assert ROOT.gROOT.ProcessLine(".O0") == 0
    assert (ROOT.TObject.kSingleKey, ROOT.TObject.kOverwrite, ROOT.TNamed.kCanDelete) == (1, 2, 1)


def test_a_map_of_pointers_written_object_wise_reads_pair_by_pair():
    from xrdroot.buffer import Buffer
    from xrdroot.interp import _pointer_map

    step = _pointer_map("map<int,TNamed*>", type("Source", (), {"streamers": dict})(), ())
    raw = bytes.fromhex("4000000e" "000a" "00000001" "00000007" "00000000")
    assert step(Buffer(raw), {}) == [(7, None)]
    assert _pointer_map("map<int,TNamed>", None, ()) is None


def test_an_array_lists_what_it_holds_and_passes_over_its_empty_slots(capsys):
    held = ROOT.TObjArray()
    held.AddAt(ROOT.TNamed("n", "t"), 2)
    held.ls()
    assert capsys.readouterr().out.count("OBJ: TNamed") == 1
    _evo_declared()
    from xrdroot.pyroot.core.declared import new_value

    assert ROOT.TClass("Evo").New().a == 1 and new_value("Evo").a == 1


def _evo_declared():
    run_source(VERSIONS % "int", "evo.C", call=False)



STREAMED = r"""
#include "TBufferFile.h"
class simple {
public:
   int i = 0;
   float f = 0;
   simple() {}
   simple(int) : i(3), f(1.5) {}
   ClassDef(simple, 2);
};
void streamed() {
   TBufferFile buf(TBuffer::kWrite);
   simple out(3);
   out.Streamer(buf);
   buf.SetReadMode();
   buf.Reset();
   simple in;
   in.Streamer(buf);
   printf("%d %g %s\n", in.i, in.f, in.IsA()->GetName());
}
"""


def test_classdef_gives_a_class_its_own_streamer_and_isa(capfd):
    run_source(STREAMED, "streamed.C")
    assert capfd.readouterr().out == "3 1.5 simple\n"


def test_get_class_info_answers_for_a_class_something_describes_and_null_otherwise():
    """A macro asks ``cl->GetClassInfo() == 0`` to decide whether to load a dictionary first,
    so a class that is here answers with one and a class nobody describes with nothing."""
    _declared()
    known = ROOT.TClass.GetClass("Named").GetClassInfo()
    expect(
        ((known.IsValid(), known.GetName(), known.Tagnum()), (True, "Named", 0)),
        (known.Size() > 0, True),
        (repr(known), "<ClassInfo_t of Named>"),
        (ROOT.TClass.GetClass("vector<int>").GetClassInfo().GetName(), "vector<int>"),
        (ROOT.TClass.GetClass("NobodyKnowsThis").GetClassInfo(), None),
    )


def test_has_interpreter_info_answers_for_a_class_the_interpreter_describes():
    """``HasInterpreterInfo`` tells ROOT whether the class has a declaration to build a
    ``TStreamerInfo`` from: one PyROOT has, one C++ declared and a container do, a class
    nobody describes does not."""
    _declared()
    expect(
        (ROOT.TClass.GetClass("Named").HasInterpreterInfo(), True),
        (ROOT.TClass.GetClass("TNamed").HasInterpreterInfo(), True),
        (ROOT.TClass.GetClass("vector<int>").HasInterpreterInfo(), True),
        (ROOT.TClass.GetClass("NobodyKnowsThis").HasInterpreterInfo(), False),
    )


def test_an_enum_constant_is_one_of_a_class_s_data_members_as_cpp_lists_it():
    """A dictionary lists ``TObject::kZombie`` among ``TNamed``'s public data members, so a
    macro looking for it finds it - and a class nothing describes has no constants."""
    from xrdroot.pyroot.core.reflection import _is_constant

    members = ROOT.gROOT.GetClass("TNamed").GetListOfAllPublicDataMembers()
    named = [member.GetName() for member in members]
    assert "kZombie" in named and "kIsOnHeap" in named and "fName" in named
    assert ROOT.gROOT.GetClass("TNamed").GetDataMember("kZombie").GetName() == "kZombie"
    assert ROOT.gROOT.GetClass("TNamed").GetDataMember("notAMember") is None
    assert [_is_constant(name, value) for name, value in
            (("kZombie", 1), ("fName", 1), ("kThing", "text"), ("ALLCAPS", 2), ("CamelCase", 3))
            ] == [True, False, False, True, False]  # fmt: skip
    assert ROOT.TObject.kZombie == 0x04000000 and ROOT.TH1F.kIsOnHeap == 0x01000000
    assert list(ROOT.TClass("NobodyKnowsThis").GetListOfDataMembers()) == []


def test_a_class_lists_the_enums_it_declares_and_an_empty_list_for_none():
    """``TClass::GetListOfEnums`` is the session's table filtered by owner, as ROOT's is: a
    class declaring no enumeration answers with an empty list rather than ``nullptr``, which
    is what roottest's ``execEnumsTest`` walks."""
    nested = troot.note_enum("A::B::C::enumC1", [("kClassEnumC1A", 0)])
    sibling = troot.note_enum("A::B::C::enumC2", [("kClassEnumC2A", 0)])
    troot.note_enum("A::B::enumB", [("kClassEnumB1", 0)])
    troot.note_enum("ETest", [("kA", 3)])
    listed = ROOT.TClass.GetClass("A::B::C").GetListOfEnums()
    assert listed.ClassName() == "TListOfEnums"
    assert list(listed) == [nested, sibling]
    assert listed.FindObject("enumC1") is nested
    # The flag ROOT reads the class's payload for fills the same table in, so it adds nothing.
    assert list(ROOT.TClass.GetClass("A::B::C").GetListOfEnums(True)) == [nested, sibling]
    assert ROOT.TClass.GetClass("TNamed").GetListOfEnums().GetSize() == 0
    assert list(ROOT.TClass("NobodyKnowsThis").GetListOfEnums()) == []


def test_a_namespace_answers_with_the_enums_declared_in_it():
    """A namespace is a ``TClass`` too, and ``TClass::GetClass("N")->GetListOfEnums()`` is
    how ``runTEnum`` reaches ``N::namespaceEnum`` - the enum's owner is what owns it."""
    namespaced = troot.note_enum("N::namespaceEnum", [("kNamespace", 1)])
    troot.note_enum("N2::otherEnum", [("kOther", 1)])
    assert namespaced.GetClass() is not None
    assert list(ROOT.TClass.GetClass("N", True).GetListOfEnums()) == [namespaced]


def test_a_file_written_before_a_class_had_a_version_is_listed_above_the_librarys():
    """ROOT records version 1 for a class whose library had none when the file was written, so
    that layout belongs above the in-memory one rather than at 1 where ``ClassDef`` put it -
    ``AddVersion-1-3`` reads a version-1 file beside a library at version 3."""
    from xrdroot.pyroot.core.reflection import _free_version

    run_source(
        VERSIONS.replace("public:", "public: ClassDef(Evo, 3)") % "long", "evo.C", call=False
    )
    assert _free_version("Evo", 1, {3}) == 1


def test_an_emulated_classs_second_layout_takes_the_next_number():
    """A class nothing in memory describes is emulated from its files and ROOT numbers the
    layouts as they come, so a version already taken gets the next number."""
    from xrdroot.pyroot.core.reflection import _free_version

    _evo("int", "one.root")
    ROOT.TFile("one.root")
    assert _free_version("Evo", 1, {1}) == 2


#: A file's layout of a versioned class, naming the member ``int myvar_one`` where the class
#: in memory declares ``double mydouble_two`` - ``roottest``'s ``checkWarn``, whose class is
#: given a version of its own in each macro.
ON_FILE_INT = ("TStreamerBasicType", "myvar_one", "", 3, 4, 1, 0, (0, 0, 0, 0, 0), "int", ())


def _renamed_class(name, spelling):
    run_source(VERSED.replace("Versed", name).replace("int a = 1", f"{spelling} = 1"),
               f"{name}.C", call=False)  # fmt: skip


def test_a_member_the_file_names_differently_is_set_against_the_class_s_own(capfd):
    """``roottest``'s ``checkWarn`` opens a file recording ``int myvar_one`` where the class in
    memory declares ``double mydouble_two``: ROOT pairs the two members by where each comes,
    not by name, and sets the two declarations out."""
    _renamed_class("Renamed", "double mydouble_two")
    reflection.note_file_infos([("Renamed", (0x1234567, 1, (ON_FILE_INT,)))], "data1.root")
    printed = capfd.readouterr().err
    assert "Warning in <TStreamerInfo::BuildCheck>:" in printed
    assert "   int myvar_one; //\nvs\n   double mydouble_two; //" in printed


def test_a_member_named_in_both_layouts_is_never_paired_by_position(capfd):
    """A file that names a member the class has is set against that member, so a layout with
    one member the class shares and one it does not names only the one it cannot read."""
    run_source(VERSED, "versed.C", call=False)
    somewhere = (*ON_FILE_INT[:1], "b", *ON_FILE_INT[2:])
    reflection.note_file_infos([("Versed", (0x1234567, 1, (LONG64, somewhere)))], "one.root")
    printed = capfd.readouterr().err
    assert "   Long64_t a; //\nvs\n   int a; //" in printed
    assert "myvar_one" not in printed


def test_a_dictionary_loaded_after_the_file_is_a_post_load_check(capfd):
    """``roottest``'s ``checkWarn0`` reads ``data1.root`` and then loads the class: the file's
    layout is the only description of the class there was, so ROOT says the version should be
    updated and lists the members the class in memory does not declare."""
    reflection.note_file_infos([("Renamed", (0x1234567, 1, (ON_FILE_INT,)))], "data1.root")
    _renamed_class("Renamed", "double mydouble_two")
    capfd.readouterr()
    reflection.note_generated_dictionary("Renamed")
    printed = capfd.readouterr().err
    assert (
        "Warning in <TClass::PostLoadCheck>: \n"
        "   The StreamerInfo version 1 for the class Renamed which was read\n"
        "   from a file previously opened has the same version as the active class\n"
        "   but a different checksum. You should update the version to ClassDef(Renamed,2).\n"
        "   Do not try to write objects with the current class definition,\n"
        "   the files will not be readable.\n" in printed
    )
    assert (
        "Warning in <TStreamerInfo::CompareContent>: The following data member of\n"
        "the in-memory layout version 1 of class 'Renamed' is missing from\n"
        "the on-file layout version 1:\n   int myvar_one; //" in printed
    )


def test_a_dictionary_is_checked_against_the_file_once(capfd):
    """ROOT warns once per class however many dictionaries a session loads of it, so a second
    load of the class says nothing."""
    reflection.note_file_infos([("Renamed", (0x1234567, 1, (ON_FILE_INT,)))], "data1.root")
    _renamed_class("Renamed", "double mydouble_two")
    reflection.note_generated_dictionary("Renamed")
    capfd.readouterr()
    reflection.note_generated_dictionary("Renamed")
    assert "PostLoadCheck" not in capfd.readouterr().err


def test_a_dictionary_of_a_class_no_file_disagrees_with_says_nothing(capfd):
    """A class whose layout in memory nothing disagrees with - no file read it, or a file whose
    checksum is the class's own - is loaded without a word."""
    run_source(VERSED, "versed.C", call=False)
    reflection.note_generated_dictionary("Versed;Nonesuch")
    assert "PostLoadCheck" not in capfd.readouterr().err
    checksum = reflection.TClass("Versed")._layout()[0]
    reflection.note_file_infos([("Versed", (checksum, 1, (LONG64,)))], "same.root")
    reflection.note_generated_dictionary("Versed")
    assert "PostLoadCheck" not in capfd.readouterr().err


def test_a_file_layout_of_a_class_with_no_classdef_is_no_post_load_check(capfd):
    """A class declared without ``ClassDef`` is told apart by checksum rather than by version,
    so a file's layout of it is no version the class failed to update."""
    run_source("class Forin {\npublic:\n   int a = 1;\n};\n", "forin.C", call=False)
    reflection.note_file_infos([("Forin", (0x1234567, 1, (ON_FILE_INT,)))], "data1.root")
    reflection.note_generated_dictionary("Forin")
    assert "PostLoadCheck" not in capfd.readouterr().err


def test_a_layout_the_class_in_memory_disowns_is_listed_as_roots_file_layout(capfd):
    """A file's layout of a version the class in memory also has, with a checksum of its own,
    stays the class's current one - ROOT warns the class should be given the next version
    rather than replacing it - and is listed with each element set against nothing: a type a
    hundred above the file's own, no new type, and the missing offset."""
    reflection.note_file_infos([("Owned", (0x589374F7, 1, (ON_FILE_INT,)))], "data1.root")
    _renamed_class("Owned", "double mydouble_two")
    info = ROOT.TClass.GetClass("Owned").GetStreamerInfo()
    assert (info.GetCheckSum(), info.GetClassVersion()) == (0x589374F7, 1)
    info.ls()
    printed = capfd.readouterr().out
    assert "checksum=0x589374f7" in printed
    assert "  int            myvar_one       offset=99999 type= 3 newtype=-1" in printed
    assert "   i= 0, myvar_one       type=103, offset=99999, len=1, method=0" in printed


def test_a_class_whose_members_a_file_names_alike_lists_them_at_their_offsets(capfd):
    """The class in memory's own layout - the one ROOT builds - is listed with each element's
    offset and the action it is read with, and a file's layout that agrees with it changes
    nothing."""
    run_source(VERSED, "versed.C", call=False)
    reflection.note_file_infos([("Versed", (0x1234, 1, (AS_WIDE,)))], "one.root")
    ROOT.TClass.GetClass("Versed").GetStreamerInfo().ls()
    printed = capfd.readouterr().out
    assert "offset=  0 type= 3" in printed and "newtype" not in printed


def test_an_element_a_file_spelled_another_way_names_the_type_in_memory(capsys):
    """ROOT lists a file's element with the *new* type the class in memory gives that member,
    and reads it with an action numbered two hundred above the file's - ``kHasNewType``. An
    enum a file wrote as the ``int`` it is declared as is the case the roottest pair shows."""
    spelled = [("TStreamerBasicType", "first", "", 3, 4, 0, 0, (0,) * 5, "AnEnum", ())]
    today = [("TStreamerBasicType", "first", "", 13, 4, 0, 0, (0,) * 5, "AnEnum", ())]
    from xrdroot.pyroot.core.reflection import TStreamerInfo

    TStreamerInfo("Spelled", (0xB15D1241, 2, tuple(spelled)), built=True, from_file=True,
                  memory=tuple(today)).ls()
    printed = capsys.readouterr().out
    assert "  AnEnum         first           offset=  0 type= 3 newtype=13" in printed
    assert "   i= 0, first           type=203, offset=  0, len=1, method=0" in printed


def test_a_member_the_two_layouts_type_alike_names_no_new_type(capsys):
    """The same type in both layouts is no new type: ROOT's listing is the file's own."""
    same = [("TStreamerBasicType", "first", "", 3, 4, 0, 0, (0,) * 5, "int", ())]
    extra = [("TStreamerBasicType", "gone", "", 3, 4, 0, 0, (0,) * 5, "int", ())]
    from xrdroot.pyroot.core.reflection import TStreamerInfo

    info = TStreamerInfo("Alike", (1, 2, tuple(same + extra)), built=True, memory=tuple(same))
    info.ls()
    printed = capsys.readouterr().out
    assert "newtype" not in printed
    assert "   i= 0, first           type=  3, offset=  0, len=1, method=0" in printed
    # A member the layout in memory does not have is the file's own, new type or no.
    assert "   i= 1, gone            type=  3, offset=  0, len=1, method=0" in printed
