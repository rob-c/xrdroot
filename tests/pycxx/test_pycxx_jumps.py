"""``goto``, its labels, and the increments Python has no statement for.

A jump back to a label is the loop a C++ program writes its retry as, and the
translator writes that loop. An increment of what an increment returned - the
variable itself, or a copy of it - is not a jump, but it is a place Python has
no ``++`` for either, and these are what it becomes.
"""

from __future__ import annotations

import pytest

from xrdroot.pycxx import Refusal, run_source, translate


def _run(source: str, name: str = "main") -> object:
    return run_source(source, "t.cxx", call=False)[name]()


def test_a_jump_back_to_a_label_is_the_loop_a_retry_is() -> None:
    python = translate(
        "int f(int n) {\n"
        "  int size = 4;\n"
        "again:\n"
        "  int m = n + size;\n"
        "  if (m > 10) { size *= 2; goto again; }\n"
        "  return m;\n"
        "}\n",
        "t.cxx",
    )
    assert "while True:" in python and "continue" in python and "break" in python
    assert "goto" not in python


def test_the_loop_a_label_heads_runs_the_body_again_until_it_is_done() -> None:
    source = (
        "int main() {\n"
        "  int size = 4;\n"
        "  int tries = 0;\n"
        "again:\n"
        "  int m = size;\n"
        "  ++tries;\n"
        "  if (m < 10) { size *= 2; goto again; }\n"
        "  return m + 10 * tries;\n"
        "}\n"
    )
    assert _run(source) == 46


def test_a_block_that_does_not_jump_back_leaves_the_loop_the_label_made() -> None:
    source = (
        "int f() {\n"
        "  int n = 0;\n"
        "again:\n"
        "  ++n;\n"
        "  return n;\n"
        "}\n"
    )
    assert _run(source, "f") == 1


def test_a_jump_forward_to_a_label_is_refused_by_name() -> None:
    with pytest.raises(Refusal) as raised:
        translate("int f() { goto done; done: return 1; }\n", "t.cxx")
    assert "goto forward to a label, or across a loop's edge" in str(raised.value)


def test_a_jump_back_across_a_break_is_refused_by_name() -> None:
    source = (
        "int f() {\n"
        "  int n = 0;\n"
        "  while (n < 3) {\n"
        "again:\n"
        "    ++n;\n"
        "    if (n < 2) goto again;\n"
        "    break;\n"
        "  }\n"
        "  return n;\n"
        "}\n"
    )
    with pytest.raises(Refusal) as raised:
        translate(source, "t.cxx")
    assert "which becomes a loop a break or a continue already in it would leave" in str(
        raised.value
    )


def test_a_jump_inside_a_loop_of_its_own_is_refused_by_name() -> None:
    source = (
        "int f() {\n"
        "again:\n"
        "  for (int i = 0; i < 3; ++i) { goto again; }\n"
        "  return 0;\n"
        "}\n"
    )
    with pytest.raises(Refusal) as raised:
        translate(source, "t.cxx")
    assert "goto forward to a label, or across a loop's edge" in str(raised.value)


def test_an_increment_of_what_an_increment_left_increments_the_variable_once_more() -> None:
    source = "int main() { int c = 3; auto d = (++c)++; return d * 100 + c; }\n"
    assert "c := c + 1" in translate(source, "t.cxx")
    assert _run(source) == 405


def test_the_value_of_a_postfix_increment_of_a_prefix_one_is_the_one_before_its_own() -> None:
    source = "int main() { int c = 3; auto d = (--c)--; return d * 100 + c; }\n"
    assert _run(source) == 201


def test_a_prefix_increment_of_a_prefix_one_is_the_last_value() -> None:
    source = "int main() { int c = 3; auto d = ++(++c); return d * 100 + c; }\n"
    assert _run(source) == 505


def test_both_increments_of_a_statement_are_taken_and_the_value_is_let_go_of() -> None:
    source = "int main() { int c = 3; (++c)--; ++(++c); return c; }\n"
    python = translate(source, "t.cxx")
    assert "-= 1" in python and "+= 1" in python
    assert _run(source) == 5


def test_the_step_taken_on_the_copy_a_postfix_increment_returns_is_dropped() -> None:
    source = "int main() { int c = 3; ++c++; return c; }\n"
    assert _run(source) == 4


def test_an_increment_of_a_value_rather_than_a_place_is_refused_by_name() -> None:
    source = (
        "struct Box { int f() const { return 1; } };\n"
        "int main() { Box b; ++(b.f()); return 0; }\n"
    )
    with pytest.raises(Refusal) as raised:
        translate(source, "t.cxx")
    assert "is not something this translator turns into Python" in str(raised.value)


def test_a_label_at_the_end_of_its_block_labels_nothing() -> None:
    source = "int main() { int c = 2; done:\n  return c; }\n"
    assert _run(source) == 2


def test_a_jump_to_a_computed_address_is_refused_by_name() -> None:
    with pytest.raises(Refusal) as raised:
        translate("int f(void *p) { goto *p; }\n", "t.cxx")
    assert "goto of a computed address" in str(raised.value)


def test_an_increment_of_an_increment_of_a_variable_held_in_a_cell() -> None:
    source = "int main() { int c = 3; int *p = &c; auto d = (++c)--; return d * 100 + c; }\n"
    assert _run(source) == 403


def test_the_steps_of_an_increment_of_an_increment_of_a_cell_are_both_taken() -> None:
    source = "int main() { int c = 3; int *p = &c; (++c)--; return c; }\n"
    assert _run(source) == 3


def test_a_step_each_way_of_a_copy_is_refused_by_name() -> None:
    source = "int main() { int c = 3; ++(c--); return c; }\n"
    with pytest.raises(Refusal) as raised:
        translate(source, "t.cxx")
    assert "which steps each way" in str(raised.value)


def test_the_increment_operators_a_class_defines_are_the_class_s_own() -> None:
    source = (
        "struct N { int v;\n"
        "  N &operator++() { ++v; return *this; }\n"
        "  N &operator--() { --v; return *this; } };\n"
        "int main() { N n; auto d = (++n)--; return d.v; }\n"
    )
    with pytest.raises(Refusal) as raised:
        translate(source, "t.cxx")
    assert "changing this in the middle of an expression" in str(raised.value)


def test_a_store_through_what_a_container_iterator_gives_is_refused_by_name() -> None:
    # ``auto it = v.begin()`` leaves the iterator's type to the dialect that names the
    # container's; spelled out, the iterators of a library's containers are the same shape.
    source = (
        "#include <vector>\n"
        "int main() { std::vector<int> v(3); std::vector<int>::iterator it = v.begin();\n"
        "  *it = 1; return 0; }\n"
    )
    with pytest.raises(Refusal) as raised:
        translate(source, "t.cxx")
    assert "assigning through an object that is a value, not a place" in str(raised.value)


def test_the_increment_of_the_copy_a_postfix_increment_returns_is_let_go_of() -> None:
    source = "int main() { int c = 3; auto d = ++c++; return d * 100 + c; }\n"
    assert _run(source) == 404


def test_a_step_each_way_of_a_copy_is_read_as_one_step_the_way_the_outer_one_goes() -> None:
    source = "int main() { int c = 3; auto d = ++(c--); return d * 100 + c; }\n"
    with pytest.raises(Refusal) as raised:
        translate(source, "t.cxx")
    assert "changing this in the middle of an expression" in str(raised.value)
