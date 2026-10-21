"""Optional named returns keep one native numeric local, with explicit decline cases."""

from __future__ import annotations

import types
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from xrdroot import pyroot as ROOT
from xrdroot.cint import translate
from xrdroot.cint.ctype import CType
from xrdroot.cint.execute import run_source
from xrdroot.cint.nodes import (
    Block,
    DeclStmt,
    ExprStmt,
    Function,
    Name,
    Param,
    Return,
    Unary,
    VarDecl,
)
from xrdroot.cint.nrvo import candidate
from xrdroot.cint.parse_decl import parse
from xrdroot.cint.program import Program, walk
from xrdroot.cint.tokens import tokenize


def parsed(source: str) -> tuple[Program, Function]:
    program = Program(parse(tokenize(source, "nrvo.C")))
    return program, program.functions["f"][0]


def chosen(source: str) -> bool:
    program, function = parsed(source)
    return candidate(function, program) is not None


@pytest.mark.parametrize("kind", [
    "std::vector<double>", "RVec<unsigned long>", "ROOT::RVec<float>",
    "ROOT::VecOps::RVec<int>", "RVec<RVec<size_t>>", "std::vector<RVec<double>>",
])
def test_an_exact_numeric_local_and_tail_return_share_the_same_binding(kind: str) -> None:
    source = f"{kind} f() {{ {kind} value(2); return value; }}"
    program, function = parsed(source)
    selected = candidate(function, program)
    assert selected is not None and function.body is not None
    statement = function.body.body[0]
    assert isinstance(statement, DeclStmt)
    assert selected.declaration is statement.decls[0]
    assert selected.tail is function.body.body[-1]
    python = translate(source)
    assert "return value\n" in python and "return value_copy(value)" not in python


@pytest.mark.parametrize("body", [
    "RVec<double> value; return value;",
    "RVec<double> value{1., 2.}; return value;",
    "RVec<double> other(2); RVec<double> value(other); return value;",
    "RVec<double> value(2); value[0] = 3.; return value;",
    "RVec<double> value(2); RVec<double>& r = value; r[0] = 3.; return value;",
    "RVec<double> value(2); RVec<double>* p = &value; keep(p); return value;",
    "RVec<double> value(2); int best = 3; best = 4; ++best; return value;",
])
def test_construction_element_mutation_and_escaped_identity_do_not_add_a_copy(body: str) -> None:
    # NRVO leaves constructor evaluation intact and permits the local's address to escape.
    assert chosen(f"RVec<double> f() {{ {body} }}")


@pytest.mark.parametrize("source", [
    "RVec<double> f();",
    "RVec<double> f() {}",
    "RVec<double> f() { RVec<double> value(2); }",
    "RVec<double> f() { return RVec<double>(2); }",
    "RVec<double> f() { RVec<double> value(2); return value[0]; }",
    "RVec<double> f() { RVec<double> value(2); return value; ; }",
    "RVec<double> f() { RVec<double> value(2); if (true) return value; return value; }",
    "RVec<double> f() { { RVec<double> value(2); return value; } }",
    "RVec<double> f() { RVec<double> value(2); { RVec<double> value(3); } return value; }",
    "RVec<double> value; RVec<double> f() { RVec<double> value(2); return value; }",
    "RVec<double> f(RVec<double> value) { return value; }",
    "RVec<double> f(RVec<double> value) { RVec<double> value(2); return value; }",
    "RVec<double> value; RVec<double> f() { return value; }",
    "RVec<double> f() { static RVec<double> value(2); return value; }",
    "RVec<double> f() { thread_local RVec<double> value(2); return value; }",
    "RVec<double> f() { extern RVec<double> value; return value; }",
    "RVec<double> f() { RVec<double> other(2); RVec<double> value = other; return value; }",
    "RVec<double> f() { RVec<double> value(2), other(2); value = other; return value; }",
    "RVec<double> f() { RVec<double> value(2), other(2); RVec<double>& r=value; "
    "r=other; return value; }",
    "RVec<double> f() { RVec<double> value(2), other(2); RVec<double>* p=&value; "
    "*p=other; return value; }",
    "RVec<double> f() { RVec<double> value(2); try { keep(value); } catch (...) {} "
    "return value; }",
    "RVec<double> f() { RVec<double> value(2); auto next=[](){return 1;}; return value; }",
    "RVec<double> f() { struct Other { int x; }; RVec<double> value(2); return value; }",
    "constexpr RVec<double> f() { RVec<double> value(2); return value; }",
    "consteval RVec<double> f() { RVec<double> value(2); return value; }",
    "template<class T> RVec<double> f() { RVec<double> value(2); return value; }",
])
def test_unmodelled_storage_control_flow_and_rebindings_keep_the_ordinary_path(
    source: str,
) -> None:
    assert not chosen(source)


@pytest.mark.parametrize(("returned", "declared"), [
    ("RVec<double>", "const RVec<double>"),
    ("RVec<double>", "volatile RVec<double>"),
    ("RVec<double>&", "RVec<double>"),
    ("RVec<double>*", "RVec<double>*"),
    ("RVec<double>", "std::vector<double>"),
    ("RVec<double>", "RVec<float>"),
    ("RVec<double>", "auto"),
    ("RVec<bool>", "RVec<bool>"),
    ("RVec<TString>", "RVec<TString>"),
    ("RVec<RVec<volatile double>>", "RVec<RVec<volatile double>>"),
    ("RVecF", "RVecF"),
    ("int", "int"),
])
def test_only_resolved_same_type_native_numeric_containers_are_eligible(
    returned: str, declared: str,
) -> None:
    assert not chosen(f"{returned} f() {{ {declared} value; return value; }}")


def test_alias_qualifiers_storage_and_constant_function_metadata_survive_parsing() -> None:
    source = """
    typedef volatile RVec<double> V;
    typedef V Again;
    constexpr RVec<double> f();
    RVec<double> f() { static int a; extern int b; thread_local int c; Again value; return value; }
    """
    program, function = parsed(source)
    assert function.constant_evaluation and function.body is not None
    declarations = {node.name: node for node in walk(function.body) if isinstance(node, VarDecl)}
    assert declarations["value"].ctype.volatile
    assert declarations["a"].storage == frozenset({"static"})
    assert declarations["b"].storage == frozenset({"extern"})
    assert declarations["c"].storage == frozenset({"thread_local"})
    assert candidate(function, program) is None


def test_a_macro_class_cannot_borrow_a_native_container_eligibility_name() -> None:
    assert not chosen("struct RVec {}; RVec f() { RVec value; return value; }")
    program, function = parsed("RVec<double> f() { RVec<double> value(2); return value; }")
    program.classes["RVec"] = object()  # type: ignore[assignment]
    assert candidate(function, program) is None


def test_noncanonical_ast_types_and_return_names_fail_closed() -> None:
    program, function = parsed("RVec<double> f() { RVec<double> value(2); return value; }")
    assert function.body is not None
    declaration = function.body.body[0]
    tail = function.body.body[-1]
    assert isinstance(declaration, DeclStmt) and isinstance(tail, Return)
    local = declaration.decls[0]
    original_type = local.ctype
    for invalid in [
        replace(original_type, args=[]), replace(original_type, args=["double"]),
        replace(original_type, args=[CType("double"), CType("Allocator")]),
        replace(original_type, dims=[2]), replace(original_type, enum="E"),
    ]:
        local.ctype = invalid
        assert candidate(function, program) is None


def test_noncanonical_return_names_methods_and_template_values_fail_closed() -> None:
    program, function = parsed("RVec<double> f() { RVec<double> value(2); return value; }")
    assert function.body is not None
    tail = function.body.body[-1]
    assert isinstance(tail, Return)
    original_name = tail.value
    assert isinstance(original_name, Name)
    for name in [replace(original_name, rooted=True), replace(original_name, parts=["a", "value"]),
                 replace(original_name, targs=[])]:
        tail.value = name
        assert candidate(function, program) is None
    tail.value = original_name
    assert candidate(replace(function, kind="operator"), program) is None
    assert candidate(function, program, method=True) is None
    function.values = [Param(function.where, "n", CType("int"))]
    assert candidate(function, program) is None


def test_static_binding_certification_declines_structured_or_unrecognised_initializers() -> None:
    program, function = parsed("RVec<double> f() { RVec<double> value(2); return value; }")
    assert function.body is not None
    declaration = function.body.body[0]
    assert isinstance(declaration, DeclStmt)
    local = declaration.decls[0]
    local.binding = ["part"]
    assert candidate(function, program) is None
    local.binding = None
    local.style = "unrecognised"
    assert candidate(function, program) is None
    local.style = "()"
    local.name = "other"
    assert candidate(function, program) is None


def test_a_nontoplevel_binding_or_increment_cannot_be_certified_by_its_spelling() -> None:
    program, function = parsed("RVec<double> f() { RVec<double> value(2); return value; }")
    assert function.body is not None
    declaration, tail = function.body.body
    assert isinstance(tail, Return) and isinstance(tail.value, Name)
    function.body.body[0] = Block(declaration.where, [declaration])
    assert candidate(function, program) is None
    function.body.body[0] = declaration
    increment = Unary(tail.where, "++", tail.value)
    function.body.body.insert(1, ExprStmt(tail.where, increment))
    assert candidate(function, program) is None


@pytest.mark.parametrize("kind", ["RVec<double>", "std::vector<double>"])
def test_a_returned_numeric_local_preserves_identity_full_initialized_storage_and_calls(
    kind: str,
) -> None:
    held: list[Any] = []
    published: list[Any] = []
    bits = np.array([0x8000000000000000, 0x7FF8000000000042, 1, 0x3FF0000000000000],
                    dtype=np.uint64)

    def keep(value: Any) -> None:
        value._data.view(np.uint64)[:] = np.resize(bits, value.capacity())
        held.append(value)

    root = types.SimpleNamespace(std=ROOT.std, RVec=ROOT.RVec, keep=keep, publish=published.append)
    source = f"""
    {kind} make() {{ {kind} value(4); value.reserve(9); keep(value); return value; }}
    void t() {{ publish(make()); publish(make()); }}
    """
    run_source(source, "t.C", root=root)
    assert len(held) == len(published) == 2
    assert all(result is local for result, local in zip(published, held, strict=True))
    assert held[0] is not held[1] and type(held[0]) is type(held[1])
    for value in published:
        assert value.size() == 4 and value.capacity() >= 9
        assert np.array_equal(value._data.view(np.uint64), np.resize(bits, value.capacity()))
    published[0][0] = 7.0
    assert published[1]._data.view(np.uint64)[0] == bits[0]


def test_a_nested_numeric_local_retains_its_live_leaves_and_separate_invocations() -> None:
    held: list[Any] = []
    root = types.SimpleNamespace(RVec=ROOT.RVec, keep=held.append)
    source = """
    RVec<RVec<unsigned long>> f() {
      RVec<RVec<unsigned long>> value(2);
      value[0].push_back(0); value[1].push_back(18446744073709551615UL);
      keep(value); return value;
    }
    """
    first = run_source(source, "f.C", root=root)
    second = run_source(source, "f.C", root=root)
    assert first is held[0] and second is held[1] and first is not second
    assert first[0] is held[0][0] and first[1] is held[0][1]
    assert list(first[0]) == [0] and list(first[1]) == [2**64 - 1]
    first[0].push_back(7)
    assert list(second[0]) == [0]


def test_noneligible_named_return_and_ordinary_assignment_still_make_value_copies() -> None:
    source = """
    RVec<double> f(RVec<double> value) { return value; }
    void t() { RVec<double> first(2); auto second = first; }
    """
    python = translate(source)
    assert "return value_copy(value)" in python
    assert "second = value_copy(first)" in python


@pytest.mark.parametrize("kind", ["RVec<double>", "std::vector<double>"])
@pytest.mark.parametrize("before", [True, False], ids=["guard-before", "guard-after"])
def test_a_destructor_local_keeps_ordinary_return_copy_and_lifecycle_in_both_orders(
    kind: str, before: bool,
) -> None:
    events: list[str] = []
    held: list[Any] = []
    bits = np.array([0x8000000000000000, 0x7FF8000000000042], dtype=np.uint64)

    def keep(value: Any) -> None:
        value.data().view(np.uint64)[:] = bits
        held.append(value)
        events.append("keep")

    def event(code: int) -> None:
        events.append({1: "construct", 2: "destroy"}[code])

    root = types.SimpleNamespace(std=ROOT.std, RVec=ROOT.RVec, keep=keep, event=event)
    vector = f"{kind} value(2);"
    declarations = "Guard guard; " + vector if before else vector + " Guard guard;"
    source = f"""
    struct Guard {{ Guard() {{ event(1); }} ~Guard() {{ event(2); }} }};
    {kind} f() {{ {declarations} keep(value); return value; }}
    """
    python = translate(source, "f.C")
    assert "return value_copy(value)" in python
    result = run_source(source, "f.C", root=root)
    assert events == ["construct", "keep", "destroy"] and len(held) == 1
    assert type(result) is type(held[0]) and result is not held[0]
    assert np.array_equal(result.data().view(np.uint64), bits)
    held[0][0] = 7.0
    assert np.array_equal(result.data().view(np.uint64), bits)


@pytest.mark.parametrize(("spelling", "maker"), [
    ("std::vector", ROOT.std.vector), ("RVec", ROOT.RVec),
])
@pytest.mark.parametrize("operation", ["construct", "push", "emplace", "setitem", "assign", "fill"])
def test_nrvo_preserves_nested_value_ownership_for_construction_and_mutation_boundaries(
    spelling: str, maker: Any, operation: str,
) -> None:
    leaf = maker["double"]
    nested = maker[leaf]
    words = np.array([0x8000000000000000, 0x7FF0000000000001,
                      0x7FF8000000000042, 1], dtype=np.uint64)
    original = leaf(words.view(np.float64))
    source = nested([original, original])
    locals_seen: list[Any] = []
    leaf_references: list[Any] = []
    root = types.SimpleNamespace(std=ROOT.std, RVec=ROOT.RVec,
                                 keep=locals_seen.append, keep_leaf=leaf_references.append)
    kind = f"{spelling}<{spelling}<double>>"
    bodies = {
        "construct": f"{kind} value(source);",
        "push": f"{kind} value; value.push_back(source[0]); value.push_back(source[0]);",
        "emplace": f"{kind} value; value.emplace_back(source[0]); value.emplace_back(source[0]);",
        "setitem": f"{kind} value(2); keep_leaf(value[0]); "
                   "value[0] = source[0]; value[1] = source[0];",
        "assign": f"{kind} value; value.assign({{source[0], source[0]}});",
        "fill": f"{kind} value; value.resize(2, source[0]);",
    }
    cpp = f"{kind} f(const {kind}& source) {{ {bodies[operation]} keep(value); return value; }}"
    python = translate(cpp, "f.C")
    assert "return value\n" in python and "return value_copy(value)" not in python
    first = run_source(cpp, "f.C", args=(source,), root=root)
    second = run_source(cpp, "f.C", args=(source,), root=root)
    _assert_nested_return_ownership(first, second, source, locals_seen, words)
    if operation == "setitem":
        assert leaf_references[0] is first[0] and leaf_references[1] is second[0]
    first[0][0] = 7.0
    for value in [first[1], second[0], second[1], source[0], source[1], original]:
        assert np.array_equal(value.data().view(np.uint64), words)


def _assert_nested_return_ownership(
    first: Any, second: Any, source: Any, observed: list[Any], words: Any,
) -> None:
    assert len(observed) == 2 and first is observed[0] and second is observed[1]
    assert type(first) is type(second) is type(source) and first is not second
    leaves = [first[0], first[1], second[0], second[1], source[0], source[1]]
    assert len({id(value) for value in leaves}) == 6
    for index, value in enumerate(leaves):
        assert type(value) is type(source[0])
        assert np.array_equal(value.data().view(np.uint64), words)
        assert all(not np.shares_memory(value.data(), other.data()) for other in leaves[index + 1:])


def test_a_fresh_two_leaf_reconstruction_local_returns_its_exact_independent_indices() -> None:
    observed: list[Any] = []
    root = types.SimpleNamespace(RVec=ROOT.RVec, keep=observed.append)
    cpp = """
    RVec<RVec<size_t>> f() {
      RVec<RVec<size_t>> value(2);
      value[0].reserve(2); value[1].reserve(2);
      value[0].emplace_back(0); value[0].emplace_back(2);
      value[1].emplace_back(1); value[1].emplace_back(3);
      keep(value); return value;
    }
    """
    first = run_source(cpp, "f.C", root=root)
    second = run_source(cpp, "f.C", root=root)
    assert first is observed[0] and second is observed[1] and first is not second
    assert [list(value) for value in first] == [[0, 2], [1, 3]]
    leaves = [first[0], first[1], second[0], second[1]]
    assert len({id(value) for value in leaves}) == 4
    first[0][0] = 9
    assert [list(value) for value in second] == [[0, 2], [1, 3]]
    assert list(first[1]) == [1, 3]
