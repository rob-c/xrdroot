"""A library's dialect of C++: its typedefs, classes, macros, host and hooks, said as data.

``Lab`` is a made-up library whose programs use ``Real`` for ``double``, name
classes ``LabSomething`` without declaring them, predefine ``LAB_VERSION``,
define ``METRES`` in ``lab/units.h`` and have ``Lab::Format``, which builds a
string. Its dialect says all that; the translator, which knows nothing of it,
then reads such a program and runs it against the library's namespace.
"""

from __future__ import annotations

import re
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from xrdroot.pycxx import (
    Dialect,
    Prelude,
    Refusal,
    active,
    run_program,
    run_source,
    translate,
    using,
)
from xrdroot.pycxx import runtime as plain
from xrdroot.pycxx.base import P
from xrdroot.pycxx.ctype import canonical
from xrdroot.pycxx.cursor import looks_like_type
from xrdroot.pycxx.runtime.namespace import HostNamespace


class LabHist:
    """The library's one class: a list of what is filled into it."""

    def __init__(self, name: str) -> None:
        self.name, self.entries = name, []

    def Fill(self, value: float) -> None:
        self.entries.append(value)

    def Entries(self) -> int:
        return len(self.entries)


#: The library's namespace, as its host finds names in it.
LIBRARY = types.SimpleNamespace(LabHist=LabHist, Format=lambda *a: "unused")

#: What the hooks saw, test by test.
SEEN: list[Any] = []


def _formatted(emitter: Any, func: Any, node: Any) -> tuple[str, int] | None:
    """``Lab::Format(fmt, ...)``: C's formatting, as the library's ``Format`` does it."""
    if func.parts != ["Lab", "Format"]:
        return None
    return f"cformat({emitter.arguments(node)})", P.POSTFIX


def _counted(emitter: Any, node: Any, decl: Any, turns: str) -> bool:
    SEEN.append(("counted", decl.name, turns))
    return False


def _ranged(emitter: Any, node: Any) -> bool:
    SEEN.append(("ranged", node.decl.name))
    return False


def _changed(emitter: Any, owner: Any, target: str, name: str, args: list[str]) -> str | None:
    return f"{target} = {args[0]}" if name == "Reset" else None


@contextmanager
def _session() -> Iterator[None]:
    SEEN.append("session")
    yield


HOST = HostNamespace(lambda: LIBRARY)

#: A macro of the library's header a translation refuses wherever it is used.
REFUSED = {"FURLONGS": "FURLONGS, a unit no lab measures in, is not translated"}

LAB = Dialect(
    name="lab",
    translator="lab-translator",
    typedefs={"Real": "double"},
    class_pattern=re.compile(r"^Lab[A-Z]\w*$"),
    type_namespaces=frozenset({"Lab"}),
    predefined="#define LAB_VERSION 3\n",
    preludes={"lab/units.h": Prelude("#define METRES(x) (x)\n", "METRES", REFUSED)},
    host="LAB",
    host_namespace="Lab",
    runtime="lab_runtime",
    runtime_names=frozenset(plain.__all__) | {"LAB"},
    host_object=HOST,
    calls={"Format": _formatted},
    call_classes=frozenset({"Lab"}),
    member_statement=_changed,
    counted_loop=_counted,
    range_loop=_ranged,
    spelling=lambda returns: returns.name if returns.name == "int" else None,
    shown=lambda value, spelled: f"({spelled}) {value}",
    status=lambda value, spelled: 0,
    session=_session,
)

PROGRAM = r"""
#include "lab/units.h"
#include "lab/units.h"

int filled() {
   LabHist h("h");
   Real step = METRES(0.5);
   for (int i = 0; i < LAB_VERSION; ++i) h.Fill(i * step);
   std::vector<int> more{4};
   for (int m : more) h.Fill(m);
   const char *text = Lab::Format("%d entries", 4);
   printf("%s\n", text);
   Real kept = 1;
   kept.Reset(2.5);
   return h.Entries();
}
"""


@pytest.fixture(autouse=True)
def _lab_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The library's runtime: plain C++'s, and its host as ``LAB``."""
    module = types.ModuleType("lab_runtime")
    module.__dict__.update({name: getattr(plain, name) for name in plain.__all__}, LAB=HOST)
    monkeypatch.setitem(sys.modules, "lab_runtime", module)
    SEEN.clear()
    yield


def test_the_dialect_says_what_its_names_are_and_plain_cpp_does_not() -> None:
    assert canonical("Real") == "Real" and not looks_like_type(["LabHist"], set())
    with using(LAB):
        assert active() is LAB
        assert canonical("Real") == "double" and canonical("size_t") == "unsigned long"
        assert looks_like_type(["LabHist"], set()) and looks_like_type(["Lab", "Axis"], set())


def test_a_program_in_the_dialect_reads_as_the_library_says(tmp_path) -> None:
    python = translate(PROGRAM, "filled.cxx", LAB)
    assert python.startswith("# Translated from filled.cxx by lab-translator.")
    assert "from lab_runtime import *" in python
    assert "h = LAB.LabHist('h')" in python
    assert "step = 0.5" in python and "range(0, 3)" in python
    assert "text = cformat('%d entries', 4)" in python
    assert "kept = 2.5" in python
    assert SEEN == [("counted", "i", "range(0, 3)"), ("ranged", "m")]


def test_a_program_in_the_dialect_runs_against_the_librarys_namespace(tmp_path, capsys) -> None:
    path = tmp_path / "filled.cxx"
    path.write_text(PROGRAM)
    assert run_program([path], dialect=LAB, use_cache=False) == 0
    assert capsys.readouterr().out == "4 entries\n(int) 4\n"
    assert "session" in SEEN


def test_a_run_may_look_names_up_in_another_namespace_than_the_hosts() -> None:
    other = types.SimpleNamespace(LabHist=lambda name: types.SimpleNamespace(name=name * 2))
    source = "const char *named() { LabHist h(\"ab\"); return h.name; }\n"
    assert run_source(source, "named.cxx", dialect=LAB, namespace=other) == "abab"


def test_plain_cpp_neither_knows_the_librarys_names_nor_prints_an_entrys_value(
    tmp_path, capsys
) -> None:
    path = tmp_path / "seven.cxx"
    path.write_text("int seven() { return 7; }\n")
    assert run_program([path], use_cache=False) == 7
    assert capsys.readouterr().out == ""
    assert "CXX.LabHist" in translate("void f() { LabHist h; }")


def test_cached_translations_are_kept_apart_by_dialect(tmp_path) -> None:
    from xrdroot.pycxx import cache

    assert cache.key("int f();", "f.cxx", LAB) != cache.key("int f();", "f.cxx")
    assert Path(cache.directory()).parent == tmp_path


def test_a_macro_the_dialect_refuses_is_defined_and_refused_where_it_is_used() -> None:
    source = '#include "lab/units.h"\n#ifdef FURLONGS\nint f() { return FURLONGS(2); }\n#endif\n'
    with pytest.raises(Refusal, match="FURLONGS, a unit no lab measures in"):
        translate(source, "t.cxx", dialect=LAB)
