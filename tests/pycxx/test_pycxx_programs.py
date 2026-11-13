"""Plain C++ programs translate and run with the translator alone: no ROOT, no library past std."""

from __future__ import annotations

from pathlib import Path

import pytest

from xrdroot.pycxx import (
    CXX,
    MacroError,
    Unsupported,
    active,
    run_line,
    run_program,
    run_source,
    translate,
    translate_file,
)
from xrdroot.pycxx.runtime import CXX as HOST
from xrdroot.pycxx.runtime import deref, user_literal
from xrdroot.pycxx.runtime.namespace import EMPTY, HostNamespace

#: A program of the kind the translator is built for: the standard headers, a class template,
#: a free function overloaded, a map, integer division, and ``<cmath>``.
PROGRAM = r"""
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

template <typename T>
class Box {
public:
   explicit Box(T value) { fValue = value; }
   T get() const { return fValue; }
private:
   T fValue;
};

int half(int n) { return n / 2; }
double half(double x) { return x / 2; }

int main(int argc, char **argv) {
   std::vector<int> numbers{3, 1, 4, 1, 5};
   int total = 0;
   for (int n : numbers) total += n;
   std::map<std::string, int> counts;
   counts["three"] = 3;
   Box<double> box(2.0);
   std::cout << total << " " << half(7) << " " << half(7.0) << " " << counts["three"] << " "
             << std::sqrt(box.get() * 8) << " " << argc << std::endl;
   printf("%s\n", argv[0]);
   return total % 7;
}
"""


def _written(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_plain_program_translates_into_python_that_finds_std_through_its_own_host() -> None:
    python = translate(PROGRAM, "numbers.cxx")
    assert python.startswith("# Translated from numbers.cxx by xrdroot.pycxx.")
    assert "from xrdroot.pycxx.runtime import *" in python
    assert "CXX.std.map" in python and "ROOT" not in python
    assert "idiv(n, 2)" in python


def test_a_plain_program_runs_from_main_with_its_command_line(tmp_path, capsys) -> None:
    path = _written(tmp_path, "numbers.cxx", PROGRAM)
    assert run_program([path], argv=["numbers", "--flag"], use_cache=False) == 14 % 7
    assert capsys.readouterr().out == "14 3 3.5 3 4 2\nnumbers\n"


def test_a_cached_translation_runs_the_same_the_second_time(tmp_path, capsys) -> None:
    path = _written(tmp_path, "numbers.cxx", PROGRAM)
    assert run_program([path]) == run_program([path]) == 0
    assert capsys.readouterr().out.count("14 3 3.5 3 4 1\n") == 2


def test_a_file_translates_with_its_local_headers_read_in(tmp_path) -> None:
    _written(tmp_path, "twice.h", "int twice(int n) { return 2 * n; }\n")
    path = _written(tmp_path, "use.cxx", '#include "twice.h"\nint use() { return twice(4); }\n')
    python = translate_file(path)
    assert "def twice(n):" in python and "def use():" in python


def test_an_entry_function_named_like_the_file_runs_and_its_value_is_the_status(
    tmp_path, capsys
) -> None:
    path = _written(tmp_path, "seven.cxx", "int seven() { return 7; }\n")
    assert run_program([path], use_cache=False) == 7
    assert capsys.readouterr().out == ""
    other = _written(tmp_path, "text.cxx", 'const char *text() { return "x"; }\n')
    assert run_program([other], use_cache=False) == 0


def test_a_source_hands_back_its_names_or_its_entrys_value() -> None:
    names = run_source("int square(int n) { return n * n; }\n", "square.cxx", call=False)
    assert names["square"](5) == 25
    assert run_source("int square(int n) { return n * n; }\n", "square.cxx", (6,)) == 36
    assert run_source("void nothing() {}\n", "other.cxx") is None


def test_a_line_runs_at_once_and_hands_back_the_value_of_one_expression() -> None:
    run_source("int add(int a, int b) { return a + b; }\n", "add.cxx", call=False)
    assert run_line("add(2, 3)") == 5
    assert run_line("int unused = 4") is None
    assert run_line("7 / 2;") == 3


def test_a_program_failing_as_it_runs_fails_at_its_cpp_line() -> None:
    source = "int fail() {\n  std::vector<int> v;\n  return v.at(3);\n}\n"
    with pytest.raises(MacroError, match=r"fail.cxx:3"):
        run_source(source, "fail.cxx")


def test_the_host_finds_the_standard_library_and_nothing_a_program_did_not_declare() -> None:
    assert active() is CXX and CXX.host_object is HOST
    assert HOST.std.vector is HOST.std.vector
    assert HOST.namespace() is EMPTY
    with pytest.raises(AttributeError):
        HOST.Undeclared  # noqa: B018
    with HOST.bind(type("Names", (), {"answer": 42})):
        assert HOST.answer == 42


def test_a_host_may_look_in_namespaces_a_program_uses_unqualified() -> None:
    library = type("Library", (), {"inner": type("Inner", (), {"found": len})})
    host = HostNamespace(lambda: library, ("inner", "missing.deeper"))
    assert host.found is len and host.found is len


def test_the_runtime_reads_through_a_pointer_and_makes_the_standard_literals() -> None:
    assert deref([7, 8]) == 7
    assert user_literal("ms", 250).total_seconds() == 0.25
    with pytest.raises(Unsupported, match=r'operator""_furlong'):
        user_literal("_furlong", 1)
