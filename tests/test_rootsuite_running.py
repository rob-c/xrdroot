"""Planning and running tests, with a stand-in for ``python -m xrdroot`` so nothing real runs."""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

import pytest
from tools.rootsuite import cases, plan, runner
from tools.rootsuite.model import Test

#: A Python that answers ``-m xrdroot run FILE... [-- ARGS]`` by obeying ``//`` lines in FILE.
FAKE = f"""#!{sys.executable}
import os, sys
args = sys.argv[1:]
if args[:1] == ["-c"]:
    sys.exit(0)
command, rest = args[2], args[3:]
if command != "run":
    print(command, *rest)
    sys.exit(0)
files = rest[:rest.index("--")] if "--" in rest else rest
code = 0
for name in files:
    for line in open(name.split("(")[0]).read().splitlines():
        key, _, value = line.partition(":")
        if key == "//out":
            print(value)
        elif key == "//err":
            print(value, file=sys.stderr)
        elif key == "//exit":
            code = int(value)
        elif key == "//write":
            open(value, "w").write("made")
        elif key == "//need" and not os.path.exists(value):
            print("missing " + value, file=sys.stderr)
            code = 1
        elif key == "//args":
            print(" ".join(rest[rest.index("--") + 1:]) if "--" in rest else "")
sys.exit(code)
"""


@pytest.fixture
def fake(tmp_path: Path) -> str:
    path = tmp_path / "fake-python"
    path.write_text(FAKE)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def spec(**fields) -> Test:
    base = {"id": "roottest/d/t", "ctest": "roottest-d-t", "kind": "roottest", "directory": "d",
            "runner": "macro", "files": ("d/t.C",)}
    return Test(**{**base, **fields})



def context(tmp_path: Path) -> plan.Context:
    return plan.Context(tmp_path / "root", tmp_path / "work", "py")


# --- plans ---------------------------------------------------------------------


def test_a_program_runs_all_its_sources_with_its_arguments_after_a_double_dash(tmp_path):
    made = plan.plan(spec(runner="program", files=("t/a.cxx", "t/b.cxx"), args=("-b",)),
                     context(tmp_path), tmp_path)
    work = tmp_path / "work"
    assert made.argv == ("py", "-m", "xrdroot", "run", f"{work}/t/a.cxx", f"{work}/t/b.cxx",
                         "--", "-b")
    assert made.cwd == work / "d"


def test_a_macro_with_macroarg_is_called_with_its_arguments(tmp_path):
    made = plan.plan(spec(args=("1, 2",)), context(tmp_path), tmp_path)
    assert made.argv[-1].endswith("work/d/t.C(1, 2)")


def test_a_test_checking_both_streams_merges_them_unless_it_has_an_errref(tmp_path):
    ctx = context(tmp_path)
    assert plan.plan(spec(), ctx, tmp_path).merge
    assert not plan.plan(spec(errref="d/t.err"), ctx, tmp_path).merge


def test_a_root_command_line_becomes_xrdroot_run_of_its_one_macro(tmp_path):
    made = plan.plan(spec(runner="root", args=("-b", "-q", "<build>/d/m.C+")), context(tmp_path),
                     tmp_path)
    assert made.argv[3:] == ("run", f"{tmp_path / 'work'}/d/m.C")


def test_root_expressions_and_several_macros_are_skips(tmp_path):
    ctx = context(tmp_path)
    assert "-e" in plan.plan(spec(runner="root", args=("-b", "-e", "1")), ctx, tmp_path)
    assert "2 macros" in plan.plan(spec(runner="root", args=("a.C", "b.C")), ctx, tmp_path)


def test_hadd_maps_onto_xrdroot_merge_and_other_commands_are_skips(tmp_path):
    ctx = context(tmp_path)
    made = plan.plan(spec(runner="command", command=("<build>/bin/hadd", "-f", "o.root")), ctx,
                     tmp_path)
    assert made.argv[2:4] == ("xrdroot", "merge")
    assert "genreflex" in plan.plan(spec(runner="command", command=("genreflex",)), ctx, tmp_path)


def test_build_steps_and_compiled_executables_are_skips_with_reasons(tmp_path):
    ctx = context(tmp_path)
    assert "build step" in plan.plan(spec(runner="setup", setup_of="dictionary"), ctx, tmp_path)
    assert "executable" in plan.plan(spec(runner="program", needs=("executable",)), ctx, tmp_path)


def test_a_pyunittest_runs_through_the_shim_with_a_junit_file(tmp_path):
    made = plan.plan(spec(runner="pyunittest", files=("d/t.py",)), context(tmp_path), tmp_path)
    assert made.argv[1] == str(plan.SHIM) and made.junit == tmp_path / "junit.xml"


def test_the_environment_keeps_the_tests_own_variables_but_not_ci_paths(tmp_path):
    made = plan.plan(spec(environment=("A=1", "PYTHONPATH=/ci")), context(tmp_path), tmp_path)
    assert made.env == {"ROOT_HIST": "0", "A": "1"}


# --- ordering and working copies ----------------------------------------------


def test_dependencies_and_fixtures_join_tests_into_one_ordered_unit():
    reader = spec(id="r", ctest="c-r", fixtures_required=("made",))
    writer = spec(id="w", ctest="c-w", fixtures_setup=("made",))
    after = spec(id="a", ctest="c-a", depends=("c-r",))
    alone = spec(id="x", ctest="c-x")
    groups = runner.units([after, reader, writer, alone])
    assert [[t.id for t in group] for group in groups] == [["w", "r", "a"], ["x"]]


def test_selection_is_by_kind_and_id_glob():
    tests = [spec(id="gtest/a/x", kind="gtest"), spec(id="roottest/b/y")]
    assert [t.id for t in runner.select(tests, ["gtest"])] == ["gtest/a/x"]
    assert [t.id for t in runner.select(tests, (), ["roottest/*"])] == ["roottest/b/y"]


def test_a_working_copy_holds_the_files_and_data_but_not_other_test_directories(tmp_path):
    root = tmp_path / "root"
    for name in ("d/t.C", "d/data/in.txt", "d/sub/CMakeLists.txt"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x")
    runner.prepare(root, tmp_path / "work", ["d"])
    work = tmp_path / "work" / "d"
    assert (work / "t.C").is_file() and (work / "data" / "in.txt").is_file()
    assert not (work / "sub").exists()


# --- whole runs ------------------------------------------------------------------


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    files = {
        "d/ok.C": "//out:value = 1\n", "d/ok.ref": "value = 1\n",
        "d/diff.C": "//out:value = 2\n",
        "d/write.C": "//write:made.root\n", "d/read.C": "//need:made.root\n//out:read\n",
        "d/main.cxx": "//args\n", "d/refuse.C": "//err:xrdroot run: d/refuse.C:1: no\n//exit:2\n",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def run_all(tmp_path: Path, source: Path, fake: str, tests: list[Test]) -> dict[str, dict]:
    settings = runner.Settings(root=source, out=tmp_path / "out", python=fake, jobs=2,
                               timeout=30, progress=lambda line: None)
    return {record["id"]: record for record in runner.Runner(settings).run(tests)}


def test_a_run_judges_each_test_and_keeps_a_log_of_each(tmp_path, source, fake):
    records = run_all(tmp_path, source, fake, [
        spec(id="ok", files=("d/ok.C",), outref="d/ok.ref", custom_diff=True),
        spec(id="diff", files=("d/diff.C",), outref="d/ok.ref", custom_diff=True),
        spec(id="refuse", files=("d/refuse.C",)),
        spec(id="args", runner="program", files=("d/main.cxx",), args=("-b",),
             passregex=("^-b$",)),
    ])
    assert {key: r["status"] for key, r in records.items()} == {
        "ok": "PASS", "diff": "DIFF", "refuse": "UNSUPPORTED", "args": "PASS"}
    assert (tmp_path / "out" / records["diff"]["log"] / "output.diff").is_file()
    assert records["ok"]["area"] and records["ok"]["runtime"] >= 0


def test_a_reader_runs_after_the_writer_its_fixture_names(tmp_path, source, fake):
    records = run_all(tmp_path, source, fake, [
        spec(id="read", ctest="c-read", files=("d/read.C",), fixtures_required=("f",)),
        spec(id="write", ctest="c-write", files=("d/write.C",), fixtures_setup=("f",)),
    ])
    assert records["read"]["status"] == "PASS"


def test_scope_labels_and_unexpressible_tests_are_recorded_without_running(tmp_path, source, fake):
    scope = tmp_path / "scope.toml"
    scope.write_text('[[label]]\nreason = "It is out by design."\nids = ["out/*"]\n')
    settings = runner.Settings(root=source, out=tmp_path / "out", python=fake, scope=scope,
                               progress=lambda line: None)
    records = runner.Runner(settings).run([spec(id="out/x"), spec(id="skip", skip="cannot.")])
    assert [r["status"] for r in records] == ["OUT-OF-SCOPE", "SKIP"]
    assert records[0]["reason"] == "It is out by design."


def test_the_shim_runs_a_unittest_file_with_root_as_xrdroot_pyroot(tmp_path):
    (tmp_path / "conftest.py").write_text("import ROOT._pythonization_internals\n")
    (tmp_path / "test_it.py").write_text(
        "import unittest\nimport ROOT\n\n\nclass T(unittest.TestCase):\n"
        "    def test_root_is_xrdroot(self):\n"
        "        self.assertEqual(ROOT.__name__, 'xrdroot.pyroot')\n")
    junit = tmp_path / "junit.xml"
    done = subprocess.run([sys.executable, str(plan.SHIM), "--junitxml", str(junit),
                           str(tmp_path / "test_it.py")], cwd=tmp_path, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "conftest.py not loaded" in done.stderr
    assert cases.junit(junit)[0]["passed"] == 1


def test_a_run_never_writes_into_the_root_sources(tmp_path, source, fake):
    before = sorted(p.name for p in source.rglob("*"))
    run_all(tmp_path, source, fake, [spec(id="write", files=("d/write.C",))])
    assert sorted(p.name for p in source.rglob("*")) == before
    assert (tmp_path / "out" / "work" / "d" / "made.root").is_file()
