"""Planning and running tests, with a stand-in for ``python -m xrdroot`` so nothing real runs."""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

import pytest
from tools.rootsuite import buildtree, cases, plan, runner
from tools.rootsuite.model import Test
from tools.rootsuite.process import Result

#: A Python that answers ``-m xrdroot``: ``--help`` lists ``run``, ``root``, ``build`` and
#: ``cp``; ``run``/``root`` obey ``//`` lines in the macros given; ``build`` writes its
#: product (a launcher for an executable) unless an argument says ``refuse`` or ``break``.
FAKE = f"""#!{sys.executable}
import os, sys
args = sys.argv[1:]
if args[:1] == ["-c"]:
    sys.exit(0)
command, rest = args[2], args[3:]
if command == "--help":
    print("usage: xrdroot [-h] COMMAND ...\\n\\npositional arguments:\\n  COMMAND")
    for name in ("run", "root", "build", "cp"):
        print("    " + name + "       what it does")
    sys.exit(0)
if command == "build":
    if any("refuse" in word for word in rest):
        print("xrdroot build: that is refused", file=sys.stderr)
        sys.exit(2)
    if any("break" in word for word in rest):
        print("error: it broke", file=sys.stderr)
        sys.exit(1)
    name = rest[rest.index("--name") + 1] if "--name" in rest else rest[1]
    with open("built.txt", "a") as log:
        log.write(" ".join(rest) + "\\n")
    if rest[0] == "executable":
        open(name, "w").write("#!/bin/sh\\necho ran $*\\n")
        os.chmod(name, 0o755)
    else:
        open("lib" + name + ".xrdlib", "w").write("made")
    sys.exit(0)
if command == "run":
    files = rest[:rest.index("--")] if "--" in rest else rest
elif command == "root":
    files = [w for i, w in enumerate(rest) if not w.startswith("-") and rest[i - 1] != "-e"]
    files += [w[3:] for w in rest if w.startswith(".L ")]
else:
    print(command, *rest)
    sys.exit(0)
code = 0
for name in files:
    for line in open(name.split("(")[0].rstrip("+")).read().splitlines():
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
            "runner": "root", "files": ("d/t.C",), "args": ("-q", "-b", "<build>/d/t.C")}
    return Test(**{**base, **fields})


def context(tmp_path: Path) -> plan.Context:
    return plan.Context(tmp_path / "root", tmp_path / "work", "py")


def planned(tmp_path: Path, **fields) -> plan.Invocation:
    made = plan.plan(spec(**fields), context(tmp_path), tmp_path)
    assert isinstance(made, plan.Invocation), made
    return made


def unrunnable(tmp_path: Path, **fields) -> plan.Unrunnable:
    made = plan.plan(spec(**fields), context(tmp_path), tmp_path)
    assert isinstance(made, plan.Unrunnable), made
    return made


# --- plans ---------------------------------------------------------------------


def test_a_program_runs_all_its_sources_with_its_arguments_after_a_double_dash(tmp_path):
    made = planned(tmp_path, runner="program", files=("t/a.cxx", "t/b.cxx"), args=("-b",))
    work = tmp_path / "work"
    assert made.argv == ("py", "-m", "xrdroot", "run", f"{work}/t/a.cxx", f"{work}/t/b.cxx",
                         "--", "-b")
    assert made.cwd == work / "d" and made.commands == {"run"}


def test_a_program_with_no_sources_is_a_skip(tmp_path):
    assert unrunnable(tmp_path, runner="program", files=()).status == "SKIP"


def test_a_root_command_line_is_xrdroot_root_with_the_same_words(tmp_path):
    made = planned(tmp_path, args=("-e", "#define A 1", "-q", "-b", "<build>/d/m.C+(1)",
                                    "<build>/d/n.C"))
    work = tmp_path / "work"
    assert made.argv[3:] == ("root", "-e", "#define A 1", "-q", "-b", f"{work}/d/m.C+(1)",
                             f"{work}/d/n.C")
    assert made.commands == {"root"}


def test_a_test_checking_both_streams_merges_them_unless_it_has_an_errref(tmp_path):
    assert planned(tmp_path).merge
    assert not planned(tmp_path, errref="d/t.err").merge


def test_roots_tools_map_onto_xrdroot_subcommands_with_the_same_words(tmp_path):
    ctx = context(tmp_path)
    for tool, sub in (("<build>/bin/hadd", "hadd"), ("rootls", "ls"), ("rootrm", "rm"),
                      ("rootmkdir", "mkdir"), ("rootmv", "mv"), ("rootprint", "print")):
        line = plan.command_line((tool, "-f", "<build>/o.root"), ctx)
        assert line == (("py", "-m", "xrdroot", sub, "-f", f"{ctx.work}/o.root"),
                        frozenset({sub}))


def test_root_config_is_unsupported_and_make_is_a_skip(tmp_path):
    ctx = context(tmp_path)
    refused = plan.command_line(("root-config", "--features"), ctx)
    assert isinstance(refused, plan.Unrunnable) and refused.status == "UNSUPPORTED"
    assert "root-config" in refused.reason and refused.reason.endswith(".")
    assert plan.command_line(("make", "cleantest"), ctx).status == "SKIP"
    assert plan.command_line((), ctx).status == "SKIP"


def test_dictionary_generators_run_as_xrdroot_build_dictionary(tmp_path):
    ctx = context(tmp_path)
    argv, commands = plan.command_line(
        ("cmake", "-E", "env", "LD_LIBRARY_PATH=<build>/lib", "<build>/bin/rootcling", "-v2",
         "-f", "D.cxx", "-s", "libD.so", "-rmf", "D.rootmap", "-I<build>/inc", "D.h",
         "DLinkDef.h"), ctx)
    assert commands == {"build"}
    assert argv[3:] == ("build", "dictionary", "--name", "D", "--linkdef", "DLinkDef.h", "-I",
                        f"{ctx.work}/inc", "D.h")
    argv, _ = plan.command_line(("genreflex", "a.h", "--select=s.xml", "-o", "<build>/d/a_rflx.cpp",
                                 "-I", "x", "--verbose"), ctx)
    assert argv[3:] == ("build", "dictionary", "--name", "a_rflx", "--selection", "s.xml", "-I",
                        "x", "--out", f"{ctx.work}/d", "--no-rootmap", "a.h")
    kept = ("cmake", "-E", "env", "A=1", "sh", "x")
    assert plan.command_line(kept, ctx)[0] == kept


def test_python_runs_a_script_through_xrdroot_and_anything_else_as_itself(tmp_path):
    ctx = context(tmp_path)
    assert plan.command_line(("python3", "s.py", "-v"), ctx)[0][3:] == ("run", "s.py", "--", "-v")
    assert plan.command_line(("python3", "-c", "1"), ctx)[0] == ("py", "-c", "1")
    assert plan.command_line(("python3", "<build>/bin/rootmv", "a"), ctx)[0][3:] == ("mv", "a")
    assert plan.command_line(("rootcp.py", "a"), ctx)[0][3:] == ("cp", "a")


def test_other_commands_run_as_written_when_this_machine_has_them(tmp_path):
    ctx = context(tmp_path)
    assert plan.command_line(("sh", "<build>/x.sh"), ctx)[0] == ("sh", f"{ctx.work}/x.sh")
    assert plan.command_line(("<build>/d/exe", "1"), ctx)[0] == (f"{ctx.work}/d/exe", "1")
    missing = plan.command_line(("no-such-program-here",), ctx)
    assert missing.status == "SKIP" and "does not have" in missing.reason


def test_a_command_test_that_cannot_be_run_says_why(tmp_path):
    assert unrunnable(tmp_path, runner="command", command=("root-config",)).status == \
        "UNSUPPORTED"


def test_a_setup_test_runs_its_last_build_after_the_others(tmp_path):
    builds = (("d", "dictionary", "--name", "E", "<build>/d/E.h"),
              ("d", "library", "--name", "E", "<build>/d/E.cxx"))
    made = planned(tmp_path, runner="setup", setup_of="dictionary", builds=builds)
    work = tmp_path / "work"
    assert made.argv == ("py", "-m", "xrdroot", "build", "library", "--name", "E",
                         f"{work}/d/E.cxx")
    assert [step.argv[3:5] for step in made.before] == [("build", "dictionary")]
    assert not made.before[0].shared and made.commands == {"build"}
    assert unrunnable(tmp_path, runner="setup", setup_of="x").status == "SKIP"


def test_a_built_program_runs_as_itself_after_its_shared_builds(tmp_path):
    made = planned(tmp_path, runner="gtest", launch="<build>/d/exe", args=("--x",),
                   builds=(("d", "executable", "--name", "exe", "<build>/d/a.cxx"),))
    assert made.argv == (f"{tmp_path / 'work'}/d/exe", "--x")
    assert made.before[0].shared and made.commands == {"build"}


def test_pre_and_post_commands_run_around_the_test_and_count_their_tools(tmp_path):
    made = planned(tmp_path, precmd=("rootls", "a.root"), postcmd=("sh", "-c", "true"),
                   copy=("e/f.root",))
    assert made.before[0].argv[3] == "ls" and made.after[0].argv[0] == "sh"
    assert made.commands == {"root", "ls"} and made.copy == (tmp_path / "work/e/f.root",)
    assert unrunnable(tmp_path, postcmd=("root-config",)).status == "UNSUPPORTED"


def test_a_skip_from_discovery_is_kept(tmp_path):
    assert unrunnable(tmp_path, skip="cannot.").reason == "cannot."


def test_a_pyunittest_runs_through_the_shim_with_a_junit_file(tmp_path):
    made = planned(tmp_path, runner="pyunittest", files=("d/t.py",))
    assert made.argv[1] == str(plan.SHIM) and made.junit == tmp_path / "junit.xml"


def test_a_script_runs_through_xrdroot_run(tmp_path):
    made = planned(tmp_path, runner="script", files=("d/s.py",), args=("1",))
    assert made.argv[3:] == ("run", f"{tmp_path / 'work'}/d/s.py", "--", "1")


def test_the_environment_keeps_the_tests_own_variables_but_not_ci_paths(tmp_path):
    made = planned(tmp_path, environment=("A=1", "PYTHONPATH=/ci", "B=<build>/x"))
    assert made.env == {"ROOT_HIST": "0", "A": "1", "B": f"{tmp_path / 'work'}/x"}


def test_the_build_trees_bin_comes_first_even_on_a_path_the_test_sets(tmp_path):
    run = runner.Runner(runner.Settings(root=tmp_path, out=tmp_path / "out"))
    env = run.environment({"PATH": "/only", "ROOTSYS": "/not"})
    assert env["PATH"] == f"{run.work / 'bin'}:/only" and env["ROOTSYS"] == str(run.work)


# --- the build tree ------------------------------------------------------------


def test_the_subcommands_are_read_from_xrdroots_help(fake):
    assert buildtree.subcommands(fake, {}) == {"run", "root", "build", "cp"}
    assert buildtree.subcommands("/no/such/python", {}) is None


def test_a_missing_subcommand_is_named_in_a_sentence(tmp_path):
    assert buildtree.missing(frozenset({"run"}), frozenset({"run"})) == ""
    assert buildtree.missing(frozenset({"hadd"}), None) == ""
    said = buildtree.missing(frozenset({"hadd", "build"}), frozenset({"run"}))
    assert said.startswith("This xrdroot has no `xrdroot build`") and said.endswith(".")


def test_launchers_start_xrdroots_stand_ins_or_refuse_as_xrdroot_refuses(tmp_path, fake):
    bin_dir = buildtree.launchers(tmp_path, fake, frozenset({"cp"}))
    done = subprocess.run([bin_dir / "rootcp", "a", "b"], capture_output=True, text=True)
    assert done.stdout.strip() == "cp a b"
    done = subprocess.run([bin_dir / "hadd"], capture_output=True, text=True)
    assert done.returncode == 2 and done.stderr.startswith("xrdroot hadd: This xrdroot has no")


def test_tutorials_are_copied_once_and_hsimple_made_as_the_build_makes_it(tmp_path, fake):
    assert buildtree.tutorials(tmp_path / "none", tmp_path / "w") is None
    (tmp_path / "root/tutorials").mkdir(parents=True)
    (tmp_path / "root/tutorials/hsimple.C").write_text("//write:hsimple.root\n")
    copied = buildtree.tutorials(tmp_path / "root", tmp_path / "w")
    assert buildtree.tutorials(tmp_path / "root", tmp_path / "w") == copied
    assert buildtree.hsimple(copied, fake, {}).exit_code == 0
    assert (copied / "hsimple.root").is_file() and buildtree.hsimple(copied, fake, {}) is None


# --- ordering, selection and working copies ------------------------------------


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


def test_selecting_a_test_selects_the_setups_of_its_fixtures_as_ctest_does():
    tests = [spec(id="s1", fixtures_setup=("f1",)), spec(id="s2", fixtures_setup=("f2",),
                                                        fixtures_required=("f1",)),
             spec(id="t", fixtures_required=("f2", "f1")), spec(id="other")]
    assert [t.id for t in runner.select(tests, (), ["t"])] == ["s1", "s2", "t"]


def test_a_working_copy_holds_the_files_and_data_but_not_other_test_directories(tmp_path):
    root = tmp_path / "root"
    for name in ("d/t.C", "d/data/in.txt", "d/sub/CMakeLists.txt"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x")
    runner.prepare(root, tmp_path / "work", ["d"])
    work = tmp_path / "work" / "d"
    assert (work / "t.C").is_file() and (work / "data" / "in.txt").is_file()
    assert not (work / "sub").exists()


def test_a_working_copy_another_xrdroot_made_is_copied_and_translated_again(tmp_path):
    """A .xrdlib is a source file as one xrdroot translated it, and another cannot run it.

    Handed over, it fails where the source would not - which reads as the test's failure and
    cost one phantom "missing name" cluster: twelve tests reported missing a member they had.
    """
    root = tmp_path / "root"
    (root / "d").mkdir(parents=True)
    (root / "d" / "t.C").write_text("x")
    work = tmp_path / "work"
    runner.prepare(root, work, ["d"])
    library = work / "d" / "t_C.xrdlib"
    library.write_text("translated by another xrdroot")
    stamp = work / ".rootsuite-stamp"
    assert stamp.exists() and not runner.stale(work)
    stamp.write_text("an older xrdroot\n")
    assert runner.stale(work)
    runner.prepare(root, work, ["d"])
    assert not library.exists(), "the other xrdroot's library is dropped"
    assert not runner.stale(work), "the copy is this xrdroot's now"


def test_the_directories_copied_are_those_below_root_the_tests_use():
    test = spec(files=("/abs/x.C", "e/y.C", "z.C"), copy=("f/g.root",),
                builds=(("h", "aclic", "x"),), working_dir="w")
    assert runner._directories([test]) == ["d", "e", "f", "h", "w"]


# --- whole runs ------------------------------------------------------------------


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    files = {
        "d/ok.C": "//out:value = 1\n", "d/ok.ref": "value = 1\n",
        "d/diff.C": "//out:value = 2\n",
        "d/write.C": "//write:made.root\n", "d/read.C": "//need:made.root\n//out:read\n",
        "d/main.cxx": "//args\n", "d/refuse.C": "//err:xrdroot run: d/refuse.C:1: no\n//exit:2\n",
        "d/bad.C": "//exit:1\n", "e/data.root": "data",
        "tutorials/hsimple.C": "//write:hsimple.root\n",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def macro(name: str, **fields) -> Test:
    return spec(id=name, ctest=f"c-{name}", files=(f"d/{name}.C",),
                args=("-q", "-b", f"<build>/d/{name}.C"), **fields)


def run_all(tmp_path: Path, source: Path, fake: str, tests: list[Test]) -> dict[str, dict]:
    settings = runner.Settings(root=source, out=tmp_path / "out", python=fake, jobs=2,
                               timeout=30, progress=lambda line: None)
    return {record["id"]: record for record in runner.Runner(settings).run(tests)}


def test_a_run_judges_each_test_and_keeps_a_log_of_each(tmp_path, source, fake):
    records = run_all(tmp_path, source, fake, [
        macro("ok", outref="d/ok.ref", custom_diff=True),
        macro("diff", outref="d/ok.ref", custom_diff=True),
        macro("refuse"),
        spec(id="args", runner="program", files=("d/main.cxx",), args=("-b",),
             passregex=("^-b$",)),
    ])
    assert {key: r["status"] for key, r in records.items()} == {
        "ok": "PASS", "diff": "DIFF", "refuse": "UNSUPPORTED", "args": "PASS"}
    assert (tmp_path / "out" / records["diff"]["log"] / "output.diff").is_file()
    assert records["ok"]["area"] and records["ok"]["runtime"] >= 0
    assert (tmp_path / "out/work/tutorials/hsimple.root").is_file()
    assert (tmp_path / "out/logs/build/hsimple/run.log").is_file()


def test_a_reader_runs_after_the_writer_its_fixture_names(tmp_path, source, fake):
    records = run_all(tmp_path, source, fake, [
        macro("read", fixtures_required=("f",)), macro("write", fixtures_setup=("f",)),
    ])
    assert records["read"]["status"] == "PASS"


def test_a_test_whose_fixture_was_not_set_up_is_not_run_and_keeps_the_root_cause(
        tmp_path, source, fake):
    records = run_all(tmp_path, source, fake, [
        macro("bad", fixtures_setup=("f",)),
        macro("refuse", fixtures_setup=("g",)),
        macro("read", fixtures_required=("f",), fixtures_setup=("h",)),
        macro("ok", fixtures_required=("g",)),
        macro("diff", fixtures_required=("h",)),
    ])
    assert records["read"]["status"] == "FAIL" and "runtime" not in records["read"]
    assert records["read"]["reason"].startswith(runner.BLOCKED + "exit 1")
    assert records["diff"]["reason"] == records["read"]["reason"]
    assert records["ok"]["status"] == "UNSUPPORTED"


def test_a_blocked_verdict_keeps_refusals_and_skips_and_names_a_silent_failure():
    assert runner.blocked({"id": "s", "status": "SKIP", "reason": "x."}).status == "SKIP"
    verdict = runner.blocked({"id": "s", "status": "TIMEOUT"})
    assert (verdict.status, verdict.reason) == ("FAIL", runner.BLOCKED + "s ended TIMEOUT")


def test_scope_labels_and_unexpressible_tests_are_recorded_without_running(tmp_path, source, fake):
    scope = tmp_path / "scope.toml"
    scope.write_text('[[label]]\nreason = "It is out by design."\nids = ["out/*"]\n')
    settings = runner.Settings(root=source, out=tmp_path / "out", python=fake, scope=scope,
                               progress=lambda line: None)
    records = runner.Runner(settings).run([spec(id="out/x"), spec(id="skip", skip="cannot.")])
    assert [r["status"] for r in records] == ["OUT-OF-SCOPE", "SKIP"]
    assert records[0]["reason"] == "It is out by design."


def test_a_test_needing_a_subcommand_this_xrdroot_lacks_is_unsupported(tmp_path, source, fake):
    records = run_all(tmp_path, source, fake, [
        spec(id="h", runner="command", command=("hadd", "o.root", "i.root"))])
    assert records["h"]["status"] == "UNSUPPORTED"
    assert "`xrdroot hadd`" in records["h"]["reason"]


def test_a_missing_python_package_is_a_skip(tmp_path, source, fake):
    settings = runner.Settings(root=source, out=tmp_path / "out", python=sys.executable,
                               progress=lambda line: None)
    records = runner.Runner(settings).run([macro("ok", python_deps=("no_such_package_x",))])
    assert records[0]["status"] == "SKIP" and "no_such_package_x" in records[0]["reason"]


def test_setup_tests_build_and_programs_run_what_they_built(tmp_path, source, fake):
    exe = ("d", "executable", "--name", "exe", "<build>/d/main.cxx")
    records = run_all(tmp_path, source, fake, [
        spec(id="lib", runner="setup", setup_of="dictionary", fixtures_setup=("lib",),
             builds=(("d", "dictionary", "--name", "E", "<build>/d/E.h"),
                     ("d", "library", "--name", "E", "<build>/d/E.cxx"))),
        spec(id="one", runner="gtest", launch="<build>/d/exe", args=("1",), builds=(exe,),
             passregex=("ran 1",)),
        spec(id="two", runner="gtest", launch="<build>/d/exe", args=("2",), builds=(exe,),
             passregex=("ran 2",), fixtures_required=("lib",)),
    ])
    assert [records[key]["status"] for key in ("lib", "one", "two")] == ["PASS"] * 3
    built = (tmp_path / "out/work/d/built.txt").read_text().splitlines()
    assert sorted(line.split()[0] for line in built) == ["dictionary", "executable", "library"]


def test_a_failed_or_refused_build_is_the_tests_verdict(tmp_path, source, fake):
    records = run_all(tmp_path, source, fake, [
        spec(id="no", runner="gtest", launch="<build>/d/x", builds=(("d", "library", "refuse"),)),
        spec(id="bad", runner="gtest", launch="<build>/d/x", builds=(("d", "library", "break"),)),
    ])
    assert (records["no"]["status"], records["no"]["reason"]) == ("UNSUPPORTED", "that is refused")
    assert records["bad"]["status"] == "FAIL"
    assert records["bad"]["reason"].startswith("building with xrdroot build library exited 1")


def test_a_timed_out_step_is_a_timeout():
    step = plan.Step(("x",), Path("."), "the pre-command")
    verdict = runner.step_verdict(step, Result(["x"], None, True, 5.0, "", ""))
    assert verdict.status == "TIMEOUT" and verdict.reason.startswith("the pre-command")


def test_a_shared_build_that_never_finished_reads_as_a_failure(tmp_path):
    run = runner.Runner(runner.Settings(root=tmp_path, out=tmp_path / "out"))
    step = plan.Step(("x",), tmp_path, "building", shared=True)
    run.building[(str(tmp_path), "x")] = done = __import__("threading").Event()
    done.set()
    assert run._step(step, {}, 1.0).stderr == "the shared build did not finish"


def test_pre_and_post_commands_and_copies_happen_around_the_test(tmp_path, source, fake):
    records = run_all(tmp_path, source, fake, [
        macro("read", precmd=("sh", "-c", "echo x > made.root"), copy=("e/data.root",)),
        macro("ok", postcmd=("sh", "-c", "exit 3")),
        macro("diff", precmd=("sh", "-c", "exit 4")),
    ])
    assert records["read"]["status"] == "PASS"
    assert (tmp_path / "out/work/d/data.root").read_text() == "data"
    assert (records["ok"]["status"], records["ok"]["reason"][:31]) == (
        "FAIL", "the post-command exit 3, expect")
    assert records["diff"]["reason"].startswith("the pre-command exited 4")
    log = tmp_path / "out" / records["diff"]["log"] / "run.log"
    assert "# the pre-command" in log.read_text()


def test_a_harness_error_is_one_failed_record_not_a_lost_run(tmp_path, source, fake,
                                                            monkeypatch):
    def broken(files, directory):
        raise OSError("disk full")

    monkeypatch.setattr(runner, "_copy_to_build", broken)
    records = run_all(tmp_path, source, fake, [macro("ok")])
    assert records["ok"]["status"] == "FAIL" and "harness error" in records["ok"]["reason"]


def test_the_shim_runs_a_unittest_file_with_root_as_a_module_of_pyroots_names(tmp_path):
    (tmp_path / "conftest.py").write_text("import ROOT._pythonization_internals\n")
    (tmp_path / "test_it.py").write_text(
        "import unittest\nimport ROOT\n\n\nclass T(unittest.TestCase):\n"
        "    def test_root_is_xrdroot(self):\n"
        "        self.assertEqual(ROOT.__name__, 'ROOT')\n")
    junit = tmp_path / "junit.xml"
    done = subprocess.run([sys.executable, str(plan.SHIM), "--junitxml", str(junit),
                           str(tmp_path / "test_it.py")], cwd=tmp_path, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "conftest.py not loaded" in done.stderr
    assert cases.junit(junit)[0]["passed"] == 1


def test_a_run_never_writes_into_the_root_sources(tmp_path, source, fake):
    before = sorted(p.name for p in source.rglob("*"))
    run_all(tmp_path, source, fake, [macro("write")])
    assert sorted(p.name for p in source.rglob("*")) == before
    assert (tmp_path / "out" / "work" / "d" / "made.root").is_file()


def test_a_second_run_reuses_the_working_copy_and_the_hsimple_already_made(tmp_path, source,
                                                                             fake):
    (source / "tutorials/hsimple.root").write_text("made by the build")
    (source / "d/t.py").write_text("")
    tests = [macro("ok", copy=("e",), python_deps=("os",)), macro("diff", python_deps=("os",)),
             spec(id="py", runner="pyunittest", files=("d/t.py",))]
    settings = runner.Settings(root=source, out=tmp_path / "out", python=fake, jobs=1,
                               xrdroot_src=tmp_path, progress=lambda line: None)
    for _ in range(2):
        records = {r["id"]: r for r in runner.Runner(settings).run(tests)}
    assert records["py"]["status"] == "PASS" and records["ok"]["status"] == "PASS"
    assert not (tmp_path / "out/logs/build/hsimple").exists()
    runner._copy(source / "d/ok.C", tmp_path / "out/work/d/ok.C")


def test_a_known_failure_may_be_capped_and_a_baseline_pass_may_not(tmp_path):
    """--timeout-known shortens a test the baseline already records as not PASS: it will not pass
    by being given its full time again. A baseline PASS keeps the run's timeout, so asking for the
    cap can never cost a PASS - and a test the baseline does not hold is treated as a PASS."""
    from tools.rootsuite import runner as running

    def settings(cap):
        return running.Settings(root=tmp_path, out=tmp_path / "out", timeout=300.0,
                                timeout_known=cap, progress=lambda line: None)

    made = running.Runner(settings(60.0))
    made._baseline = {"roottest/a": "FAIL", "roottest/b": "PASS", "roottest/c": "TIMEOUT"}
    def as_test(name):
        return running.Test(id=name, ctest=name, kind="roottest", directory="",
                            runner="root")
    assert made._timeout_of(as_test("roottest/a")) == 60.0   # a known FAIL is capped
    assert made._timeout_of(as_test("roottest/c")) == 60.0   # a known TIMEOUT too
    assert made._timeout_of(as_test("roottest/b")) == 300.0  # a baseline PASS is never shortened
    assert made._timeout_of(as_test("roottest/new")) == 300.0  # not in the baseline: assume PASS
    off = running.Runner(settings(None))
    off._baseline = {"roottest/a": "FAIL"}
    assert off._timeout_of(as_test("roottest/a")) == 300.0   # off by default
    own = running.Runner(settings(60.0))
    own._baseline = {"roottest/a": "FAIL"}
    shorter = running.Test(id="roottest/a", ctest="a", kind="roottest", directory="",
                           runner="root", timeout=5.0)
    assert own._timeout_of(shorter) == 5.0  # its own cap is shorter still
