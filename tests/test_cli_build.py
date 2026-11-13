"""``xrdroot build``: rootcling's dictionary, a library, a program and ACLiC's library."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyrootsupport import fresh
from xrdroot import pyroot as ROOT
from xrdroot.cli import main
from xrdroot.cli.build import launcher
from xrdroot.pycint.library import loading, read
from xrdroot.pycxx import cache
from xrdroot.pycxx.runtime.namespace import DECLARED


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(cache.ENVIRONMENT, str(tmp_path / "cache"))
    for variable in loading.ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(ROOT.gSystem, "_dynamic_path", "")
    saved, names = dict(DECLARED), set(vars(ROOT))
    loading.forget_all()
    try:
        yield from fresh(tmp_path)
    finally:
        loading.forget_all()
        DECLARED.clear()
        DECLARED.update(saved)
        for name in set(vars(ROOT)) - names:  # what gInterpreter.Declare made ROOT's
            if getattr(getattr(ROOT, name), "__module__", None) == "__cint__":
                delattr(ROOT, name)


def write(name: str, text: str) -> Path:
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_build_dictionary_writes_the_library_and_warns_of_unused_rules(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("A.h", "namespace ns { class A { public: int f() { return 3; } }; }")
    write("LinkDef.h", "#pragma link C++ class ns::A+;\n#pragma link C++ class Gone+;\n")
    assert main(["build", "dictionary", "--name", "A", "--linkdef", "LinkDef.h", "-I", ".",
                 "--out", "lib", "A.h"]) == 0  # fmt: skip
    assert capsys.readouterr().err == "Warning: Unused class rule: Gone\n"
    library = read(Path("lib/libA.xrdlib"))
    assert library is not None and library.rootmap
    write("sel.xml", '<lcgdict><class name="ns::A"/></lcgdict>')
    assert main(["build", "dictionary", "--name", "B", "--selection", "sel.xml",
                 "--no-rootmap", "--library", "libA", "A.h"]) == 0  # fmt: skip
    library = read(Path("libB.xrdlib"))
    assert library is not None and not library.rootmap and library.libraries == ["libA"]


def test_build_library_and_aclic(capsys: pytest.CaptureFixture[str]) -> None:
    write("s.cxx", "int seven() { return 7; }")
    assert main(["build", "library", "--name", "S", "s.cxx"]) == 0
    assert Path("libS.xrdlib").is_file()
    write("m.C", "int m() { return 1; }")
    assert main(["build", "aclic", "m.C+"]) == 0
    assert "creating shared library" in capsys.readouterr().err
    assert main(["build", "aclic", "m.C++g", "--name", "mine", "--out", "o"]) == 0
    assert Path("o/mine.xrdlib").is_file()


def test_a_build_that_fails_exits_1_with_the_reason(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["build", "dictionary", "--name", "A", "gone.h"]) == 1
    assert capsys.readouterr().err == (
        "fatal error: 'gone.h' file not found\nError: rootcling: compilation failure (gone.h)\n"
    )
    assert main(["build", "executable", "--name", "p", "gone.cxx"]) == 1
    assert capsys.readouterr().err == "Error: gone.cxx: no such file\n"
    write("bad.cxx", "int main() { return ; ; }}")
    assert main(["build", "executable", "--name", "p", "bad.cxx"]) == 2  # cannot be done
    (said,) = capsys.readouterr().err.splitlines()  # one line, as every refusal is
    assert said.startswith("xrdroot build: ") and "bad.cxx:1: " in said
    write("bad.C", "int bad() { return ; ; }}")
    assert main(["build", "aclic", "bad.C+kc"]) == 2
    assert capsys.readouterr().err.splitlines()[-1].startswith("xrdroot build: ")


def test_build_executable_writes_a_program_that_runs_as_xrdroot_run_from_anywhere(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("A.h", "int fromA() { return 40; }")
    assert main(["build", "library", "--name", "A", "--out", "libs", "A.h"]) == 0
    write("prog.cxx", '#include <cstdio>\nint fromA();\nint main(int argc, char **argv) {\n'
          '  printf("%s %d\\n", argv[1], argc);\n  return fromA() + argc; }\n')  # fmt: skip
    assert main(["build", "executable", "--name", "prog", "--out", "bin", "--library", "A",
                 "--library", "ROOT::Hist", "--library", "gtest", "-I", "libs",
                 "prog.cxx"]) == 0  # fmt: skip
    assert "library A is not built yet" in capsys.readouterr().err
    script = Path("bin/prog")
    assert os.access(script, os.X_OK)
    os.environ["LD_LIBRARY_PATH"] = str(Path("libs").resolve())
    try:
        done = subprocess.run([str(script.resolve()), "hello"], capture_output=True, text=True,
                              cwd="/", check=False)  # fmt: skip
    finally:
        del os.environ["LD_LIBRARY_PATH"]
    assert (done.returncode, done.stdout) == (42, "hello 2\n"), done.stderr


def test_a_program_s_libraries_are_found_beside_it_or_on_the_search_path() -> None:
    write("A.h", "int fromA() { return 1; }")
    assert main(["build", "library", "--name", "A", "--out", "bin", "A.h"]) == 0
    write("p.cxx", "int main() { return 0; }")
    assert main(["build", "executable", "--name", "p", "--out", "bin", "--library", "A",
                 "--library", "libTree", "p.cxx"]) == 0  # fmt: skip
    text = Path("bin/p").read_text()
    assert f"--library {Path('bin/libA.xrdlib').resolve()}" in text
    assert "libTree" not in text


def test_the_launcher_runs_the_sources_in_this_python_and_this_xrdroot() -> None:
    text = launcher([Path("a.cxx")], ["libA"])
    assert text.startswith("#!/bin/sh\n")
    assert sys.executable in text and f"--library libA {Path('a.cxx').resolve()}" in text
    assert text.endswith(' -- "$@"\n')


def test_xrdroot_run_loads_libraries_first_and_names_the_program(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("A.h", "int fromA() { return 3; }")
    assert main(["build", "library", "--name", "A", "A.h"]) == 0
    write("p.cxx", '#include <cstdio>\nint fromA();\nint main(int argc, char **argv) {\n'
          '  printf("%s\\n", argv[0]);\n  return fromA(); }\n')  # fmt: skip
    assert main(["run", "--library", "libA", "--argv0", "./p", "p.cxx"]) == 3
    assert capsys.readouterr().out == "./p\n"
    write("m.C", "int m(int n) { return n; }")
    assert main(["run", "m.C+(5)"]) == 5
    assert main(["run", "--python", "m.C+"]) == 0
    assert "def m(n):" in capsys.readouterr().out


def test_a_library_built_where_its_dictionary_is_adds_its_sources_to_it() -> None:
    write("S.h", "namespace geo { class S { public: int Area() const; }; }")
    write("S.cxx", '#include "S.h"\nint geo::S::Area() const { return 4; }')
    write("LinkDef.h", "#pragma link C++ class geo::S+;\n")
    assert main(["build", "dictionary", "--name", "S", "--linkdef", "LinkDef.h",
                 "--library", "libX", "S.h"]) == 0  # fmt: skip
    assert main(["build", "library", "--name", "S", "--library", "libY", "S.cxx"]) == 0
    library = read(Path("libS.xrdlib"))
    assert library is not None
    assert (library.kind, library.rootmap, library.libraries) == (
        "dictionary", True, ["libX", "libY"])  # fmt: skip
    assert [Path(each).name for each in library.resolved(library.sources)] == ["S.h", "S.cxx"]
    assert main(["build", "library", "--name", "S", "S.cxx"]) == 0  # again: nothing twice
    library = read(Path("libS.xrdlib"))
    assert library is not None and len(library.sources) == 2
