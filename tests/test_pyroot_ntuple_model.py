"""RNTuple models by ROOT's names: fields declared by C++ type, and the entries holding values."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.core.objects import typed
from xrdroot.pyroot.ntuple.fields import RFieldPtr, field_type


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


@pytest.mark.parametrize("cxx", ["Vector3", "std::vector<std::string>", "auto", "foo<int>"])
def test_a_field_of_a_record_or_a_nested_collection_is_refused(cxx) -> None:
    with pytest.raises(UnsupportedFeatureError, match="RNTuple field of type"):
        field_type(cxx)


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
    with pytest.raises(UnsupportedFeatureError, match="low precision"):
        field.SetHalfPrecision()
