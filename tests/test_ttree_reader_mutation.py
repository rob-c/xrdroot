"""Writable reader branch buffers retain native entry and scalar-lvalue semantics."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import xrdroot
from xrdroot import pyroot as ROOT
from xrdroot.cint import Refusal, translate
from xrdroot.cint.execute import run_source
from xrdroot.cint.runtime import deref, store_through
from xrdroot.pyroot.trees import wrap


def _tree() -> tuple[Any, Any, Any]:
    tree = ROOT.TTree("events", "")
    scalar, values = np.zeros(1, dtype="int32"), np.zeros(3, dtype="int32")
    tree.Branch("x", scalar, "x/I")
    tree.Branch("a", values, "a[3]/I")
    for number in range(3):
        scalar[0] = 10 * (number + 1)
        values[:] = np.arange(1, 4) + 3 * number
        tree.Fill()
    scalar[0], values[:] = -999, [-999] * 3
    return tree, scalar, values


def _readers(tree: Any) -> tuple[Any, Any, Any, Any]:
    reader = ROOT.TTreeReader(tree)
    value = ROOT.TTreeReaderValue["Int_t"](reader, "x")
    other = ROOT.TTreeReaderValue["Int_t"](reader, "x")
    values = ROOT.TTreeReaderArray["Int_t"](reader, "a")
    assert reader.Next()
    return reader, value, other, values


def test_numeric_reader_pointers_share_the_leaf_and_original_fill_storage() -> None:
    tree, scalar, array = _tree()
    reader, value, other, values = _readers(tree)
    pointer = value.Get()
    assert pointer is value.Get() is other.Get() is tree.GetBranch("x").GetAddress()
    assert pointer[0] == scalar[0] == tree.GetLeaf("x").GetValue() == 10
    assert array.tolist() == [-999] * 3  # An unread array branch is not loaded eagerly.
    pointer[0] = 88
    other.Get()[0] -= 3
    assert value.__deref__() == scalar[0] == tree.GetLeaf("x").GetValue() == 85
    assert values.GetSize() == 3 and array.tolist() == [1, 2, 3]
    values[1] += 100
    assert values.At(1) == tree.GetLeaf("a").GetValue(1) == array[1] == 102
    assert reader.SetEntry(0) == 0 and value.Get() is pointer
    assert pointer[0] == 85 and list(values) == [1, 102, 3]


def test_reader_entry_transitions_reload_rows_but_same_entry_restart_keeps_mutations() -> None:
    tree, scalar, array = _tree()
    reader, value, _, values = _readers(tree)
    pointer = value.Get()
    pointer[0] = 85
    values[1] += 100
    assert reader.Next() and value.Get() is pointer and pointer[0] == 20
    assert array.tolist() == [1, 102, 3]  # Scalar access has not loaded the array branch.
    assert list(values) == [4, 5, 6] and array.tolist() == [4, 5, 6]
    assert reader.SetEntry(0) == 0 and value.__deref__() == scalar[0] == 10
    assert array.tolist() == [4, 5, 6] and list(values) == [1, 2, 3]
    _check_restart_and_fresh_reader(tree, reader, value, values, pointer)


def _check_restart_and_fresh_reader(
    tree: Any, reader: Any, value: Any, values: Any, pointer: Any,
) -> None:
    pointer[0] = 99
    reader.Restart()
    assert reader.Next() and value.__deref__() == 99 and list(values) == [1, 2, 3]
    assert reader.SetEntry(2) == 0 and value.__deref__() == 30
    # The native fixture constructs this reader after the old one loaded entry two.
    fresh = ROOT.TTreeReader(tree)
    fresh_value = ROOT.TTreeReaderValue["Int_t"](fresh, "x")
    assert fresh.Next() and fresh_value.__deref__() == 10


def test_cpp_reader_assignment_and_prefix_results_bind_the_actual_branch_lvalue() -> None:
    tree, scalar, array = _tree()
    cpp = r"""
    std::vector<int> f(TTree* tree) {
        TTreeReader reader(tree);
        TTreeReaderValue<Int_t> value(reader, "x"), other(reader, "x");
        TTreeReaderArray<Int_t> values(reader, "a");
        reader.Next();
        Int_t& assigned = (*value = 41);
        Int_t& plus = (*value += 5);
        Int_t old = (*value)--;
        Int_t& before = --(*value);
        bool same = &assigned == &*value && &plus == &before;
        Int_t* pointer = value.Get();
        *pointer = 88;
        *other -= 3;
        Int_t& element = (values[1] += 100);
        element += 2;
        return {old, *value, same, pointer == &*value, before, values[1]};
    }
    """
    result = run_source(cpp, "f.C", (tree,), root=ROOT)
    assert list(result) == [46, 85, 1, 1, 85, 104]
    assert scalar[0] == 85 and array.tolist() == [1, 104, 3]
    python = translate(cpp, "f.C")
    assert "store_reference" in python and "increment_reference" in python
    assert "._cint_ref(" in python
    expected = ast.dump(ast.parse("pointer[0] = 88").body[0])
    assert any(ast.dump(node) == expected for node in ast.walk(ast.parse(python)))


def test_cpp_reader_pointer_and_array_at_references_write_their_captured_storage() -> None:
    tree, scalar, array = _tree()
    cpp = r"""
    void add(Int_t& value) { value += 7; }
    int f(TTree* tree) {
        TTreeReader reader(tree);
        TTreeReaderValue<Int_t> value(reader, "x");
        TTreeReaderArray<Int_t> values(reader, "a");
        reader.Next();
        Int_t* pointer = value.Get();
        Int_t& direct = *pointer;
        Int_t& element = values.At(1);
        add(*value);
        direct += 1;
        values.At(1) += 100;
        element -= 1;
        return *pointer + values[1];
    }
    """
    assert run_source(cpp, "f.C", (tree,), root=ROOT) == 119
    assert scalar[0] == 18 and array.tolist() == [1, 101, 3]


@pytest.mark.parametrize(("kind", "dtype", "letter", "initial", "wide", "expected"), [
    ("Char_t", "int8", "B", 120, 130, -126),
    ("UChar_t", "uint8", "b", 250, 260, 4),
    ("Short_t", "int16", "S", 32760, 32770, -32766),
    ("UShort_t", "uint16", "s", 65530, 65540, 4),
])
def test_cpp_reader_narrow_assignment_uses_the_template_pointee_width(
    kind: str, dtype: str, letter: str, initial: int, wide: int, expected: int,
) -> None:
    tree = ROOT.TTree("narrow", "")
    storage = np.array([initial], dtype=dtype)
    tree.Branch("x", storage, "x/" + letter)
    tree.Fill()
    cpp = f"""
    int f(TTree* tree) {{
        TTreeReader reader(tree); TTreeReaderValue<{kind}> value(reader, "x");
        reader.Next(); {kind}& alias = (*value = {wide});
        return alias;
    }}
    """
    assert run_source(cpp, "f.C", (tree,), root=ROOT) == expected
    assert int(storage[0]) == tree.GetLeaf("x").GetValue() == expected


@pytest.mark.parametrize(("kind", "dtype", "letter", "initial", "after", "added", "delta"), [
    ("Char_t", "int8", "B", -128, 127, -82, 300),
    ("UChar_t", "uint8", "b", 0, 255, 48, 300),
    ("Short_t", "int16", "S", -32768, 32767, -28302, 70000),
    ("UShort_t", "uint16", "s", 0, 65535, 4468, 70000),
])
def test_cpp_reader_prefix_and_postfix_narrow_after_wide_arithmetic(
    kind: str, dtype: str, letter: str, initial: int, after: int, added: int, delta: int,
) -> None:
    tree = ROOT.TTree("narrow", "")
    storage = np.array([initial], dtype=dtype)
    tree.Branch("x", storage, "x/" + letter)
    tree.Fill()
    # The native22 wide assignments are followed by these compound conversions.
    wide = {"Char_t": 130, "UChar_t": 260, "Short_t": 32770, "UShort_t": 65540}[kind]
    cpp = f"""
    std::vector<int> f(TTree* tree) {{
        TTreeReader reader(tree); TTreeReaderValue<{kind}> value(reader, "x"); reader.Next();
        int old = (*value)--; int current = *value;
        *value = {initial}; {kind}& prefix = --(*value);
        bool aliases = &prefix == &*value; int stepped = prefix;
        *value = {wide}; *value += {delta};
        return {{old, current, stepped, aliases, *value}};
    }}
    """
    assert list(run_source(cpp, "f.C", (tree,), root=ROOT)) == [initial, after, after, 1, added]
    assert int(storage[0]) == added


def _write_tree(path: Path) -> None:
    tree, _, _ = _tree()
    with xrdroot.create(str(path)) as output:
        tree.SetDirectory(output)
        tree.Write()


@pytest.mark.parametrize("branch_only", [False, True])
def test_direct_entry_read_reloads_the_same_held_file_buffer(
    tmp_path: Path, branch_only: bool,
) -> None:
    path = tmp_path / "events.root"
    _write_tree(path)
    with xrdroot.open_root(str(path)) as source:
        tree = wrap(source["events"])
        _, value, _, _ = _readers(tree)
        pointer = value.Get()
        pointer[0] = 85
        read = tree.GetBranch("x").GetEntry if branch_only else tree.GetEntry
        assert read(0) > 0 and pointer[0] == 10
        assert read(1) > 0 and pointer[0] == 20
        # No reader-cursor interleaving or branch-address manipulation is claimed here.


def test_direct_entry_read_calls_a_bound_value_setter_once() -> None:
    class Holder:
        def __init__(self) -> None:
            self.current = 10
            self.writes: list[int] = []

        @property
        def value(self) -> int:
            return self.current

        @value.setter
        def value(self, value: int) -> None:
            self.writes.append(int(value))
            self.current = int(value)

    holder = Holder()
    tree = ROOT.TTree("events", "")
    tree.Branch("x", holder, "x/I")
    tree.Fill()
    tree.SetBranchAddress("x", holder)
    reader = ROOT.TTreeReader(tree)
    value = ROOT.TTreeReaderValue["Int_t"](reader, "x")
    assert reader.Next() and value.Get()[0] == 10
    holder.writes.clear()
    assert tree.GetEntry(0) > 0 and holder.writes == [10]


def test_python_numeric_reader_bounds_and_unsupported_storage_remain_explicit() -> None:
    tree, _, _ = _tree()
    _, value, _, values = _readers(tree)
    pointer = value.Get()
    with pytest.raises(IndexError, match="reader buffer"):
        pointer[1] = 8
    with pytest.raises(IndexError, match="reader buffer"):
        values[3] = 8
    vector = ROOT.std.vector["float"]([1, 2])
    objects = ROOT.TTree("objects", "")
    objects.Branch("v", vector)
    objects.Fill()
    reader = ROOT.TTreeReader(objects)
    collection = ROOT.TTreeReaderValue["std::vector<float>"](reader, "v")
    assert reader.Next() and list(collection.Get()) == [1, 2]
    with pytest.raises(TypeError, match="fixed numeric leaf"):
        collection.value = 4
    array = ROOT.TTreeReaderArray["float"](reader, "v")
    with pytest.raises(TypeError, match="fixed numeric leaf"):
        array[0] = 4
    assert list(collection.Get()) == [1, 2]


def test_reader_raw_pointer_dereference_yields_the_reader_before_its_scalar() -> None:
    cpp = """
    int f(TTreeReaderValue<int>* pointer) {
        TTreeReaderValue<int>& reader = *pointer;
        return **pointer;
    }
    """
    python = translate(cpp, "f.C")
    assert "pointer.__deref__()" in python and "pointer[0].__deref__()" not in python


def test_macro_owned_reference_return_refusals_are_not_bypassed_by_reader_spelling() -> None:
    cpp = """
    struct TTreeReaderArray { int data; int& At(int i) { return data; } };
    void f() { TTreeReaderArray value; value.At(0) = 3; }
    """
    with pytest.raises(Refusal, match="macro's class"):
        translate(cpp, "f.C")



def test_reader_compound_reference_evaluates_its_side_effecting_index_once() -> None:
    tree, _, array = _tree()
    cpp = r"""
    int f(TTree* tree) {
        TTreeReader reader(tree); TTreeReaderArray<Int_t> values(reader, "a"); reader.Next();
        int i = 0; Int_t& selected = (values[i++] += 10);
        selected += 1;
        return i * 100 + values[0] * 10 + values[1];
    }
    """
    assert run_source(cpp, "f.C", (tree,), root=ROOT) == 222
    assert array.tolist() == [12, 2, 3]


def test_reader_assignment_evaluates_rhs_before_binding_its_destination() -> None:
    tree, _, array = _tree()
    cpp = r"""
    int f(TTree* tree) {
        TTreeReader reader(tree); TTreeReaderArray<Int_t> values(reader, "a"); reader.Next();
        int i = 0; Int_t& selected = (values[i] = ++i);
        selected += 10;
        return i * 100 + values[0] * 10 + values[1];
    }
    """
    assert run_source(cpp, "f.C", (tree,), root=ROOT) == 121
    assert array.tolist() == [1, 11, 3]


def test_reader_statement_compound_and_increment_evaluate_index_once() -> None:
    tree, _, array = _tree()
    cpp = r"""
    int f(TTree* tree) {
        TTreeReader reader(tree); TTreeReaderArray<Int_t> values(reader, "a"); reader.Next();
        int i = 0; values[i++] += 10; values[i++]++;
        return i;
    }
    """
    assert run_source(cpp, "f.C", (tree,), root=ROOT) == 2
    assert array.tolist() == [11, 3, 3]



def test_cpp_reader_reference_stays_bound_when_the_reader_changes_tree() -> None:
    old_tree, old_scalar, _ = _tree()
    new_tree, new_scalar, _ = _tree()
    cpp = r"""
    std::vector<int> f(TTree* old_tree, TTree* new_tree) {
        TTreeReader reader(old_tree); TTreeReaderValue<Int_t> value(reader, "x");
        reader.Next(); Int_t& original = *value;
        reader.SetTree(new_tree); reader.Next(); original = 77;
        return {original, *value, &original != &*value};
    }
    """
    assert list(run_source(cpp, "f.C", (old_tree, new_tree), root=ROOT)) == [77, 10, 1]
    assert old_scalar[0] == 77 and new_scalar[0] == 10


def test_cpp_reader_pointer_offsets_and_address_of_updated_lvalues_share_storage() -> None:
    tree, scalar, array = _tree()
    cpp = r"""
    int f(TTree* tree) {
        TTreeReader reader(tree); TTreeReaderValue<Int_t> value(reader, "x");
        TTreeReaderArray<Int_t> values(reader, "a"); reader.Next();
        Int_t* p = value.Get(); Int_t* same = p + 0;
        Int_t* base = &values[0]; Int_t* next = base + 1;
        *next += 100;
        Int_t* assigned = &(*value = 41); Int_t* incremented = &++(*value);
        return *same + *next + (assigned == incremented);
    }
    """
    assert run_source(cpp, "f.C", (tree,), root=ROOT) == 145
    assert scalar[0] == 42 and array.tolist() == [1, 102, 3]


@pytest.mark.parametrize(("kind", "dtype", "letter", "given", "expected"), [
    ("Bool_t", "bool", "O", 3, True),
    ("Float_t", "float32", "F", 16777217, 16777216.0),
    ("Double_t", "float64", "D", 1.25, 1.25),
])
def test_public_reader_pointer_writes_preserve_boolean_and_floating_storage_types(
    kind: str, dtype: str, letter: str, given: Any, expected: Any,
) -> None:
    tree = ROOT.TTree("typed", "")
    storage = np.zeros(1, dtype=dtype)
    tree.Branch("x", storage, "x/" + letter)
    tree.Fill()
    reader = ROOT.TTreeReader(tree)
    value = ROOT.TTreeReaderValue[kind](reader, "x")
    assert reader.Next()
    pointer = value.Get()
    assert len(pointer) == 1 and len(pointer[1:]) == 0 and pointer[0:] is pointer
    pointer[0] = given
    assert pointer[0] == storage[0] == value.__deref__() == expected
    with pytest.raises(IndexError, match="reader buffer"):
        pointer[1:][0] = given



def test_float_reader_addition_rounds_before_the_result_is_widened_to_double() -> None:
    tree = ROOT.TTree("float_values", "")
    first = np.array([16777216], dtype="float32")
    second = np.array([1], dtype="float32")
    tree.Branch("x", first, "x/F")
    tree.Branch("y", second, "y/F")
    tree.Fill()
    before = first.tobytes(), second.tobytes()
    cpp = r"""
    std::vector<double> f(TTree* tree) {
        TTreeReader reader(tree);
        TTreeReaderValue<Float_t> x(reader, "x"), y(reader, "y");
        reader.Next();
        double narrowed = *x + *y;
        double widened = static_cast<double>(*x) + static_cast<double>(*y);
        return {narrowed, widened};
    }
    """
    values = run_source(cpp, "f.C", (tree,), root=ROOT)
    expected = np.array([16777216.0, 16777217.0], dtype="float64")
    assert np.asarray(list(values), dtype="float64").tobytes() == expected.tobytes()
    assert (first.tobytes(), second.tobytes()) == before


@pytest.mark.parametrize(("index", "error", "message"), [
    (slice(0, 1), TypeError, "open slice without a step"),
    (slice(None, None, 2), TypeError, "open slice without a step"),
    (slice(-1, None), IndexError, "reader buffer"),
    (slice(2, None), IndexError, "reader buffer"),
])
def test_public_reader_pointer_offsets_reject_invalid_address_and_slice_forms(
    index: slice, error: type[Exception], message: str,
) -> None:
    tree, _, _ = _tree()
    _, value, _, _ = _readers(tree)
    pointer = value.Get()
    with pytest.raises(error, match=message):
        pointer[index]



def test_generic_reader_cell_protocol_mutates_the_shared_numeric_branch() -> None:
    tree, scalar, _ = _tree()
    _, value, other, _ = _readers(tree)
    assert deref(value) == value.value == 10
    assert store_through(value, 73) == 73
    assert value.value == other.Get()[0] == scalar[0] == tree.GetLeaf("x").GetValue() == 73


def test_cpp_pointer_to_reader_get_and_array_at_keep_the_template_pointee() -> None:
    tree, scalar, array = _tree()
    cpp = r"""
    std::vector<int> f(TTree* tree) {
        TTreeReader reader(tree);
        TTreeReaderValue<Int_t> value(reader, "x");
        TTreeReaderArray<Int_t> values(reader, "a");
        reader.Next();
        TTreeReaderValue<Int_t>* p = &value;
        TTreeReaderArray<Int_t>* a = &values;
        *p->Get() = 47;
        a->At(1) += 100;
        return {**p, a->At(1)};
    }
    """
    assert list(run_source(cpp, "f.C", (tree,), root=ROOT)) == [47, 102]
    assert scalar[0] == 47 and array.tolist() == [1, 102, 3]



def test_loading_a_bool_reader_keeps_the_original_branch_variable_boolean_for_streams(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xrdroot.cint.runtime import cout

    monkeypatch.setattr(cout, "alpha", False)
    cpp = r"""
    bool f() {
        Bool_t flag = true;
        TTree tree("bool_flags", "");
        tree.Branch("flag", &flag, "flag/O");
        tree.Fill();
        flag = false;
        TTreeReader reader(&tree);
        TTreeReaderValue<Bool_t> value(reader, "flag");
        reader.Next();
        bool loaded = *value;
        std::cout << std::boolalpha << loaded << " " << flag << " " << *value << std::endl;
        return loaded && flag;
    }
    """
    assert run_source(cpp, "f.C", root=ROOT) is True
    captured = capsys.readouterr()
    assert captured.out == "true true true\n" and captured.err == ""
