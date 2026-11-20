"""RNTuple models by ROOT's names: fields declared by C++ type, and the entries holding values."""

from __future__ import annotations

import numpy
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot import stl
from xrdroot.pyroot.core.objects import typed
from xrdroot.pyroot.ntuple import fields
from xrdroot.pyroot.ntuple.fields import RFieldPtr, _default_object, field_type
from xrdroot.pyroot.ntuple.model import RFieldBase, RResult
from xrdroot.rntuple.writer import RecordSpec


@pytest.mark.parametrize(
    ("cxx", "kept", "spec"),
    [
        ("int", "std::int32_t", "std::int32_t"),
        ("unsigned long", "std::uint64_t", "std::uint64_t"),
        ("std::uint16_t", "std::uint16_t", "std::uint16_t"),
        ("float", "float", "float"),
        ("bool", "bool", "bool"),
        ("std::string", "std::string", str),
        ("std::vector<float>", "std::vector<float>", "std::vector<float>"),
        ("ROOT::RVec<int>", "std::vector<std::int32_t>", "std::vector<std::int32_t>"),
    ],
)
def test_a_field_type_is_kept_by_the_name_roots_files_use(cxx, kept, spec) -> None:
    found = field_type(cxx)
    assert (found.cxx, found.spec) == (kept, spec)


@pytest.mark.parametrize("cxx", ["TVector3*", "std::vector<std::string>", "auto", "foo<int>"])
def test_a_field_of_a_record_or_a_nested_collection_is_refused(cxx) -> None:
    with pytest.raises(UnsupportedFeatureError, match="RNTuple field of type"):
        field_type(cxx)


def test_a_bitset_is_held_as_its_bits_and_declared_by_how_many_there_are() -> None:
    """``std::bitset<16>`` is one field type of its own, whose entry holds the bits."""
    found = field_type("std::bitset<16>")
    assert (found.cxx, found.bits, found.spec) == ("std::bitset<16>", 16, "std::bitset<16>")
    held = found.holder()
    assert isinstance(held, RFieldPtr) and held.value.to_ulong() == 0
    assert found.zero().to_string() == "0" * 16
    assert field_type("int").bits == 0 and field_type("std::vector<int>").bits == 0
    model = ROOT.RNTupleModel.Create()
    model.AddField(RFieldBase.Create("foo", "std::bitset<16>").Unwrap())
    entry = model.GetDefaultEntry()
    entry["foo"] = 0xFA2
    assert entry["foo"].to_ulong() == 0xFA2
    entry["foo"] = ROOT.std.bitset[16](0x0F)
    assert entry["foo"].to_ulong() == 0x0F


def test_a_record_is_held_as_the_object_its_class_default_constructs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MakeField<T>(name)`` hands out what a class's default constructor makes, so a fill
    of nothing writes the class's default member initialisers - which is why roottest's
    makeproject reads back ``bar == {1, 2}`` and ``foo == 0xfa2``."""

    class Point:
        def __init__(self) -> None:
            self.fX, self.fY, self.fZ = 1.0, 2.0, 3.0

    class Event:
        def __init__(self) -> None:
            self.fP = Point()
            self.fE = 4.0

    monkeypatch.setattr(fields, "_default_object", lambda cxx: Event())
    held = field_type("TLorentzVector").holder()
    assert held["fE"].value == 4.0
    assert (held["fP"]["fX"].value, held["fP"]["fZ"].value) == (1.0, 3.0)


def test_a_default_object_is_the_class_a_dictionary_declares_or_there_is_none() -> None:
    """The object a record's holder is filled from exists only where a dictionary here
    declares the class; a name none does has its members' zeros alone."""
    assert isinstance(_default_object("TLorentzVector"), ROOT.TLorentzVector)
    assert _default_object("NoClassIsDeclaredUnderThisName") is None


def test_a_record_no_dictionary_declares_is_read_as_its_members_alone() -> None:
    """A class nobody declared here has no object to make, so a view hands out the
    members the file keeps, as a ``dict``."""
    assert field_type("NoSuchEventClass").value({}) == {}


def test_a_class_with_no_default_constructor_makes_no_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``find_class`` gives a class, and ``cls()`` is what may fail: a class whose only
    constructor takes an argument is one ROOT could not make for a file either."""

    class NoDefault:
        def __init__(self, required: int) -> None:
            self.required = required

    monkeypatch.setattr(stl, "_resolve_root_type", lambda cxx: NoDefault)
    assert _default_object("NoDefault") is None


def test_a_member_of_a_container_kind_is_the_objects_own_container() -> None:
    """A vector, a set or a map member is held as the object holds it, not as a copy, so
    that what a macro puts in the object is what a fill writes."""

    class Event:
        def __init__(self) -> None:
            self.v = ROOT.std.vector["int"]([1, 2])

    made = Event()
    assert fields._member_holder("v", field_type("std::vector<int>"), made) is made.v


def test_a_value_from_a_file_is_held_as_c_holds_the_fields_type() -> None:
    """A view hands out a bitset of its bits, a vector of its items and a set or a map of
    its own, rather than the columns' arrays."""
    bits = numpy.array([bool(0xFA2 >> bit & 1) for bit in range(16)])
    assert field_type("std::bitset<16>").value(bits).to_ulong() == 0xFA2
    vector = field_type("std::vector<int>").value(numpy.array([1, 2], dtype="int32"))
    assert (vector.size(), list(vector)) == (2, [1, 2])
    items = field_type("std::unordered_set<double>").value(numpy.array([3.5, 1.5]))
    assert sorted(items) == [1.5, 3.5]
    kept = field_type("std::unordered_map<int,std::string>").value({2: "two"})
    assert dict(kept.items()) == {2: "two"}
    assert field_type("std::string").value("text") == "text"


def test_a_number_or_string_is_held_by_a_pointer_and_a_vector_is_itself() -> None:
    held = field_type("int").holder()
    assert isinstance(held, RFieldPtr) and held.value == 0 and held[0] == 0
    held[0] = 4
    assert held.value == 4 and "holding 4" in repr(held)
    assert field_type("std::string").holder().value == ""
    vector = field_type("std::vector<double>").holder()
    vector.push_back(1.5)
    assert list(vector) == [1.5]


def test_a_template_method_takes_its_type_in_brackets_or_finds_it_itself() -> None:
    class Thing:
        @typed
        def Make(self, kind, name):
            return kind, name

    assert isinstance(Thing.__dict__["Make"].__get__(None, Thing), typed)
    assert Thing().Make["int"]("n") == ("int", "n")
    assert Thing().Make("n") == (None, "n")


def test_a_model_hands_out_the_holders_its_default_entry_keeps() -> None:
    model = ROOT.RNTupleModel.Create()
    n = model.MakeField["int"]("n")
    pt = model.MakeField["std::vector<float>"]("pt", "momenta")
    assert model.GetDefaultEntry().GetPtr["int"]("n") is n
    assert model.GetFieldNames() == ["n", "pt"] and "n, pt" in repr(model)
    assert model.GetField("pt").GetDescription() == "momenta"
    assert [value.GetField().GetFieldName() for value in model.GetDefaultEntry()] == ["n", "pt"]
    assert next(iter(model.GetDefaultEntry())).GetPtr["int"]() is n
    assert pt is not model.CreateEntry().GetPtr("pt")
    with pytest.raises(KeyError, match="invalid field: x"):
        model.GetField("x")
    with pytest.raises(ValueError, match="already exists"):
        model.MakeField["int"]("n")
    model.Freeze()
    assert model.IsFrozen()
    with pytest.raises(RuntimeError, match="frozen model"):
        model.MakeField["int"]("m")


def test_a_bare_model_has_no_default_entry_but_makes_entries() -> None:
    model = ROOT.RNTupleModel.CreateBare()
    assert model.MakeField["float"]("x") is None and "no fields" not in repr(model)
    with pytest.raises(RuntimeError, match="bare model"):
        model.GetDefaultEntry()
    entry = model.CreateBareEntry()
    assert entry.GetPtr("x").value == 0.0
    copied = model.Clone()
    assert copied.GetFieldNames() == ["x"] and "no fields" in repr(ROOT.RNTupleModel())


def test_an_entry_binds_a_value_and_names_its_fields_when_one_is_missing() -> None:
    entry = ROOT.RNTupleModel.Create().CreateEntry()
    with pytest.raises(KeyError, match="no fields"):
        entry.GetPtr("x")
    model = ROOT.RNTupleModel.Create()
    model.MakeField["int"]("a")
    entry = model.CreateEntry()
    mine = field_type("int").holder()
    entry.BindValue["int"]("a", mine)
    assert entry.GetPtr("a") is mine and entry.values() == {"a": mine}
    with pytest.raises(KeyError, match="has a"):
        entry.BindValue("b", mine)


def test_a_field_made_apart_is_cloned_under_another_name() -> None:
    field = ROOT.RField["float"]("x")
    assert field.GetTypeName() == "float" and "'x' of float" in repr(field)
    assert ROOT.RField("y").GetTypeName() == "double"
    field.SetDescription("an x")
    clone = field.Clone("y")
    assert (clone.GetFieldName(), clone.GetQualifiedFieldName(), clone.GetDescription()) == (
        "y", "y", "an x")  # fmt: skip
    field.SetHalfPrecision()
    assert repr(field.kind().spec) == "<LowPrecisionSpec half keeping a half of a float32>"


def test_a_field_made_from_a_type_name_comes_back_in_a_result() -> None:
    result = RFieldBase.Create("pt", "std::vector<float>", "momenta")
    assert isinstance(result, RResult) and result.IsValid() and result.GetError() == ""
    field = result.Unwrap()
    assert "RField" in result.Inspect() and "RResult" in repr(result)
    assert type(field) is ROOT.RField and isinstance(field, RFieldBase)
    assert (field.GetFieldName(), field.GetTypeName(), field.GetDescription()) == (
        "pt", "std::vector<float>", "momenta")  # fmt: skip
    model = ROOT.RNTupleModel.Create()
    model.AddField(RFieldBase.Create("n", "int").Unwrap())
    assert model.GetFieldNames() == ["n"] and model.GetDefaultEntry().GetPtr("n").value == 0


def test_a_set_or_a_map_is_a_collection_field_over_the_item_under_it() -> None:
    """roottest's makeproject header holds a ``std::unordered_set<double>``, a
    ``std::unordered_map<int, std::string>`` and a ``std::unordered_multimap<std::string,
    std::vector<float>>``; ROOT writes each as a collection - a set over its item, a map
    over the record of a key and a value - so each is a field of its item's type."""
    spam = field_type("std::unordered_set<double>")
    assert (spam.cxx, spam.collection, spam.record, spam.item.cxx) == (
        "std::unordered_set<double>", True, False, "double")  # fmt: skip
    assert repr(spam.spec) == "<CollectionSpec std::unordered_set<double>>"
    eggs = field_type("std::unordered_map<int, std::string>")
    assert eggs.cxx == "std::unordered_map<std::int32_t,std::string>"
    assert eggs.item.cxx == "std::pair<std::int32_t,std::string>"
    assert [name for name, _ in eggs.item.members] == ["_0", "_1"]
    assert isinstance(eggs.spec.item, RecordSpec) and eggs.spec.keyed
    strange = field_type("std::unordered_multimap<std::string, std::vector<float>>")
    assert strange.cxx == "std::unordered_multimap<std::string,std::vector<float>>"
    assert [member.cxx for _, member in strange.item.members] == [
        "std::string", "std::vector<float>"]  # fmt: skip
    assert not field_type("std::vector<double>").collection
    with pytest.raises(UnsupportedFeatureError, match="RNTuple field of type"):
        field_type("std::map<int>")


def test_a_set_or_a_map_is_held_in_the_container_of_its_kinds() -> None:
    """An entry holds a set or a map in the C++ container itself, which cppstd has for
    every one of them but the unordered multimap and multiset: those keep their items in
    the ordered container of the same kinds, which iterates them the same way."""
    spam = field_type("std::unordered_set<double>").holder()
    assert type(spam).__name__ == "unordered_set<double>" and list(spam) == []
    eggs = field_type("std::unordered_map<int,std::string>").holder()
    eggs[2] = "two"
    assert eggs.items() == [(2, "two")]
    strange = field_type("std::unordered_multimap<std::string,std::vector<float>>").holder()
    assert type(strange).__name__ == "multimap<string,vector<float>>"
    assert field_type("std::multiset<int>").holder().__cpp_name__ == "multiset<int>"


def test_an_entry_of_a_set_or_a_map_is_filled_with_its_items() -> None:
    """C++ has no assignment of a list to a container either: an entry is filled the way
    a macro fills one, item by item, after clearing what the holder held."""
    model = ROOT.RNTupleModel.Create()
    model.MakeField["std::unordered_set<double>"]("spam")
    model.MakeField["std::unordered_map<int,std::string>"]("eggs")
    entry = model.GetDefaultEntry()
    entry["spam"] = [1.0, 3.0]
    entry["eggs"] = {2: "two"}
    assert list(entry["spam"]) == [1.0, 3.0]
    assert entry["eggs"].items() == [(2, "two")]
    entry["spam"] = [5.0]
    assert list(entry["spam"]) == [5.0]
    with pytest.raises(TypeError, match="so it holds the items of a list"):
        entry["spam"] = 1.5
    with pytest.raises(TypeError, match="so it holds the items of a list"):
        entry["eggs"] = "two"


def test_a_field_of_a_class_type_is_a_record_of_the_members_its_dictionary_lists() -> None:
    """ROOT reads the class's dictionary to lay a record out, and writes the field under
    the class's name; ``write.cxx`` of roottest's atlas-datavector asks for exactly this."""
    field = RFieldBase.Create("my_field", "AtlasLikeDataVector<CustomStruct>").Unwrap()
    assert field.GetTypeName() == "AtlasLikeDataVector<CustomStruct>"
    kind = field.kind()
    assert kind.record and kind.holder() == {}
    model = ROOT.RNTupleModel.Create()
    model.AddField(field)
    entry = model.GetDefaultEntry()
    entry["my_field"] = {}  # a record's value is its members', by name
    assert entry["my_field"] == {}
    with pytest.raises(TypeError, match="a dict of its members"):
        entry["my_field"] = 3
    declared = kind.spec
    assert declared.cxx == "AtlasLikeDataVector<CustomStruct>"


def test_a_record_is_declared_to_the_writer_as_the_fields_of_its_members() -> None:
    """ROOT lays a class out as a record of the members its dictionary lists, so the
    writer is told the class's name and each member's own declaration under it."""
    declared = field_type("TLorentzVector").spec
    assert repr(declared) == "<RecordSpec TLorentzVector of fP, fE>"
    assert [name for name, _ in declared.members] == ["fP", "fE"]
    assert repr(declared.members[0][1]) == "<RecordSpec TVector3 of fX, fY, fZ>"
    assert declared.members[1][1] == "double"


def test_a_record_holds_the_members_of_the_classes_inside_it() -> None:
    kind = field_type("TLorentzVector")
    assert kind.record and [name for name, _ in kind.members] == ["fP", "fE"]
    assert [member.cxx for _, member in kind.members] == ["TVector3", "double"]
    assert [name for name, _ in kind.members[0][1].members] == ["fX", "fY", "fZ"]
    assert kind.holder()["fE"].value == 0.0


def test_a_name_that_is_no_class_is_refused_rather_than_taken_for_a_record() -> None:
    """``flaot`` is a misspelt number: ROOT refuses a type no dictionary describes, and
    so does xrdroot here. A capitalised name is a class, and a class is a record - one
    with no members when nothing describes it."""
    with pytest.raises(UnsupportedFeatureError, match="is not supported"):
        field_type("flaot")
    assert field_type("Vector3").members == ()


def test_a_record_whose_dictionary_the_kit_cannot_read_has_no_members(monkeypatch) -> None:
    from xrdroot.pyroot.core.reflection import TClass

    def refuse(name: str) -> None:
        raise RuntimeError("a library the kit cannot translate")

    monkeypatch.setattr(TClass, "GetClass", staticmethod(refuse))
    assert field_type("TLorentzVector").members == ()


def test_a_record_that_holds_itself_is_refused() -> None:
    with pytest.raises(UnsupportedFeatureError, match="holds a Node of itself"):
        field_type("Node", frozenset({"Node"}))
