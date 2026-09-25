"""The tutorial harness's own logic, on fixtures: no ROOT is needed or run here."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from tools.tutorials import catalog as cat
from tools.tutorials import classify, cli, compare, report, runner
from tools.tutorials.cmake import Interpreter, Token, commands, glob_regex, tokens
from tools.tutorials.environment import Oracle, cmake_variables, targets
from tools.tutorials.harness import Harness, Settings, oracle_key, selected

from xrdroot import Histogram, create

TOLERANCE = compare.Tolerance()


@pytest.fixture(autouse=True)
def no_probes(monkeypatch):
    """Asking the xrdroot under test what it has starts a Python each time: say it has nothing."""
    monkeypatch.setattr(Harness, "_probe", lambda self, code: False)


def interpreted(source: str, tmp_path: Path, **options) -> Interpreter:
    return Interpreter(tmp_path, **options).run(source)


def result(**fields) -> runner.RunResult:
    base = {"command": ["x"], "exit_code": 0, "timed_out": False, "runtime": 0.1}
    base.update(stdout="", stderr="")
    return runner.RunResult(**{**base, **fields})


# --- CMake -----------------------------------------------------------------


def test_tokens_keep_quoting_and_drop_comments():
    found = list(tokens('set(a "b c" d) # e\nif(NOT (x))'))
    assert Token("b c", True) in found
    assert Token("e") not in found
    assert [t.text for t in found].count("(") == 3


def test_commands_are_named_in_lower_case_with_their_arguments():
    (first, second) = commands('SET(x 1 2)\nmessage(STATUS "hi")')
    assert first.name == "set" and [t.text for t in first.args] == ["x", "1", "2"]
    assert second.args[1] == Token("hi", True)


def test_set_and_list_build_semicolon_lists(tmp_path):
    it = interpreted(
        "set(a x y)\nlist(APPEND a z)\nlist(REMOVE_ITEM a y)\nlist(LENGTH a n)\n"
        "list(FIND a z at)\nlist(FILTER a EXCLUDE REGEX ^x)\nset(b)\nunset(a)",
        tmp_path,
    )
    assert it.variables["n"] == "2" and it.variables["at"] == "1"
    assert "a" not in it.variables and "b" not in it.variables


def test_list_filter_keeps_what_the_regex_includes(tmp_path):
    it = interpreted("set(a MPI1 x MPI2)\nlist(FILTER a INCLUDE REGEX MPI)", tmp_path)
    assert it.values("a") == ["MPI1", "MPI2"]


def test_if_follows_the_build_options_with_not_and_or(tmp_path):
    source = (
        "if(NOT roofit)\nset(r off)\nelseif(MSVC OR (xml AND NOT fitsio))\nset(r mid)\n"
        "else()\nset(r on)\nendif()"
    )
    assert interpreted(source, tmp_path).variables["r"] == "off"
    assert interpreted(source, tmp_path, variables={"roofit": "ON"}).variables["r"] == "on"
    both = {"roofit": "ON", "xml": "ON"}
    assert interpreted(source, tmp_path, variables=both).variables["r"] == "mid"


def test_if_compares_strings_numbers_versions_and_lists(tmp_path):
    source = (
        "set(l a b)\nif(b IN_LIST l)\nset(in 1)\nendif()\n"
        'if("x" STREQUAL "x")\nset(eq 1)\nendif()\n'
        "if(3 GREATER_EQUAL 2)\nset(ge 1)\nendif()\n"
        "if(1.10 VERSION_GREATER 1.9)\nset(v 1)\nendif()\n"
        "if(TARGET Gui)\nset(t 1)\nendif()\nif(DEFINED l)\nset(d 1)\nendif()\n"
        "if(abc MATCHES ^a)\nset(m 1)\nendif()\nif(ON)\nset(on 1)\nendif()"
    )
    found = interpreted(source, tmp_path, targets={"Gui"}).variables
    assert all(found.get(name) == "1" for name in ("in", "eq", "ge", "v", "t", "d", "m", "on"))


def test_foreach_sets_variables_named_by_its_items(tmp_path):
    source = "set(xs p q)\nforeach(t ${xs})\nset(${t}-depends dep-${t})\nendforeach()\n"
    source += (
        "foreach(u IN LISTS xs)\nset(last ${u})\nendforeach()\nforeach(w IN ITEMS k)\nendforeach()"
    )
    found = interpreted(source, tmp_path).variables
    assert found["p-depends"] == "dep-p" and found["last"] == "q" and found["w"] == "k"


def test_string_replace_and_environment_expansion(tmp_path):
    it = interpreted(
        'string(REPLACE ".C" "" n a/b.C)\nset(e "$ENV{HOME_X}")', tmp_path, environ={"HOME_X": "h"}
    )
    assert it.variables["n"] == "a/b" and it.variables["e"] == "h"


def test_glob_stars_stay_within_a_directory():
    assert glob_regex("a/*.C") == r"a/[^/]*\.C"
    assert glob_regex("df10[2-7]*") == r"df10[2-7][^/]*"
    assert glob_regex("x[!ab]?") == r"x[^ab][^/]"


def test_file_glob_finds_files_and_recurse_finds_them_deeper(tmp_path):
    for name in ("a/x.C", "a/b/y.C", "z.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("")
    it = interpreted(
        "file(GLOB flat RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} a/*.C)\n"
        "file(GLOB_RECURSE deep RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.C)\nfile(WRITE f x)",
        tmp_path,
    )
    assert it.values("flat") == ["a/x.C"]
    assert it.values("deep") == ["a/b/y.C", "a/x.C"]


def test_root_add_test_is_recorded_with_its_sections(tmp_path):
    it = interpreted(
        'ROOT_ADD_TEST(tutorial-x COMMAND root -b -q /s/x.C PASSRC 255 FAILREGEX "Error in <" '
        "LABELS tutorial longtest DEPENDS tutorial-h ENVIRONMENT A=1 TIMEOUT 60 PYTHON_DEPS numpy "
        "WILLFAIL)",
        tmp_path,
    )
    (test,) = it.tests
    assert test.passrc == 255 and test.failregex == ("Error in <",)
    assert test.depends == ("tutorial-h",) and test.timeout == 60.0 and test.will_fail
    assert test.python_deps == ("numpy",) and test.labels == ("tutorial", "longtest")


def test_python_modules_and_processes_set_their_result_variables(tmp_path):
    it = interpreted(
        "ROOT_FIND_PYTHON_MODULE(torch)\nROOT_FIND_PYTHON_MODULE(numpy)\n"
        "execute_process(COMMAND x RESULT_VARIABLE rc)\n"
        'cmake_path(CONVERT "a;b" TO_NATIVE_PATH_LIST o)',
        tmp_path,
        module=lambda name: name == "numpy",
    )
    assert it.variables["ROOT_TORCH_FOUND"] == "FALSE"
    assert it.variables["ROOT_NUMPY_FOUND"] == "TRUE"
    assert it.variables["rc"] == "1" and it.variables["o"] == "a;b"


# --- the catalogue ---------------------------------------------------------

#: A tutorials CMakeLists shaped like ROOT's: vetoes, dependencies, a return code, a loop.
MINI_CMAKE = """
set(gui_veto gui/*.C)
if(NOT roofit)
  set(roofit_veto roofit/*.C)
endif()
set(odd_veto odd/*.C)
set(all_veto hsimple.C ${gui_veto} ${roofit_veto} ${odd_veto})
set(returncode_1 hist/rc.C)
set(hist-reader-depends tutorial-hist-writer)
file(GLOB_RECURSE tutorials RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.C)
file(GLOB tutorials_veto RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${all_veto})
list(REMOVE_ITEM tutorials ${tutorials_veto})
ROOT_ADD_TEST(tutorial-hsimple COMMAND ${ROOT_root_CMD} -b -n -q
    ${CMAKE_CURRENT_SOURCE_DIR}/hsimple.C
    PASSRC 255)
foreach(t ${tutorials})
  list(FIND returncode_1 ${t} index)
  if(index EQUAL -1)
    set(rc 0)
  else()
    set(rc 255)
  endif()
  string(REPLACE ".C" "" tname ${t})
  string(REPLACE "/" "-" tname ${tname})
  ROOT_ADD_TEST(tutorial-${tname} COMMAND root -b -q ${CMAKE_CURRENT_SOURCE_DIR}/${t}
    PASSRC ${rc} FAILREGEX "Error in <" DEPENDS tutorial-hsimple ${${tname}-depends})
endforeach()
"""

#: The files of that fixture tree.
MINI_FILES = (
    "hsimple.C", "hist/writer.C", "hist/reader.C", "hist/rc.C", "gui/g.C", "roofit/r.C",
    "odd/o.C", "hist/helper.py", "hist/.hidden/x.C",
)  # fmt: skip


@pytest.fixture
def tree(tmp_path):
    for name in MINI_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("{}\n")
    (tmp_path / "CMakeLists.txt").write_text(MINI_CMAKE)
    return tmp_path


def test_discovery_finds_every_tutorial_language_but_hidden_files(tree):
    found = cat.discover(tree)
    assert "hist/helper.py" in found and "hsimple.C" in found
    assert not any(".hidden" in path for path in found)


def test_the_catalogue_says_why_each_file_is_not_run(tree):
    found = cat.catalog(tree, {"ROOT_root_CMD": "root"})
    assert found["gui/g.C"].vetoed == "needs a display (GUI)"
    assert found["roofit/r.C"].vetoed == "needs ROOT built with roofit"
    assert found["odd/o.C"].vetoed == "needs ROOT built with odd"
    assert found["hist/helper.py"].vetoed == cat.NOT_A_TEST
    assert cat.catalog(tree, {"roofit": "ON"})["roofit/r.C"].vetoed is None


def test_the_catalogue_keeps_ci_return_codes_flags_and_dependencies(tree):
    found = cat.catalog(tree, {})
    assert found["hsimple.C"].passrc == 255 and found["hsimple.C"].flags == ("-b", "-n", "-q")
    assert found["hist/rc.C"].passrc == 255
    assert found["hist/reader.C"].depends == ("hsimple.C", "hist/writer.C")
    assert found["hist/reader.C"].area == "hist" and found["hsimple.C"].area == "(top)"
    assert found["hist/reader.C"].language == "cxx" and found["hist/helper.py"].language == "py"


def test_a_directory_without_cmakelists_is_all_tutorials(tree):
    (tree / "CMakeLists.txt").unlink()
    assert all(t.vetoed is None for t in cat.catalog(tree, {}).values())


def test_dependencies_come_first_and_waves_respect_them(tree):
    found = cat.catalog(tree, {})
    assert cat.closure(found, ["hist/reader.C"]) == ["hsimple.C", "hist/writer.C", "hist/reader.C"]
    waves = cat.levels(found, ["hist/reader.C", "hist/rc.C"])
    assert waves[0] == ["hsimple.C"] and "hist/reader.C" in waves[2]


def test_selection_is_by_directory_or_file_prefix():
    paths = ["hist/a.C", "histv/b.C", "io/c.C"]
    assert selected(paths, ["hist/"]) == ["hist/a.C"]
    assert selected(paths, ["io/c.C"]) == ["io/c.C"]
    assert selected(paths, []) == paths


def test_cmake_variables_derive_the_names_cmake_tests():
    variables = cmake_variables(frozenset({"imt", "roofit"}), Path("/x/tutorials"), "py")
    assert variables["TBB_FOUND"] == "ON" and variables["roofit"] == "ON"
    assert variables["Python3_EXECUTABLE"] == "py" and "GRAPHVIZ_FOUND" not in variables
    assert targets(frozenset({"cocoa"})) == {"Gui"} and targets(frozenset()) == set()


# --- comparison ------------------------------------------------------------


def test_normalising_drops_what_differs_between_runs():
    text = (
        "Processing /tmp/x/hsimple.C...\nobject at 0x7ffe12ab\nReal time 0:00:01, CP time 0.5\n"
        "made on Mon Jan  5 10:11:12 2026\n\n  a    b  \n/w/dir/file"
    )
    lines = compare.normalise(text, {"/w/dir": "<workdir>"})
    assert lines == ["object at 0x<addr>", "made on <date>", "a b", "<workdir>/file"]


def test_numbers_are_equal_within_the_relative_tolerance():
    assert compare.same_line("mean = 1.0000001", "mean = 1.0", TOLERANCE)
    assert not compare.same_line("mean = 1.1", "mean = 1.0", TOLERANCE)
    assert not compare.same_line("mean = 1", "median = 1", TOLERANCE)
    assert compare.same_line("x nan", "x nan", TOLERANCE)
    assert not compare.same_line("x nan", "x 1", TOLERANCE)


def test_the_first_differing_line_is_reported_with_both_sides():
    found = compare.first_difference(["a 1", "b 2"], ["a 1", "b 3"], TOLERANCE)
    assert found == "line 2: ROOT 'b 2', xrdroot 'b 3'"
    shorter = compare.first_difference(["a", "b"], ["a"], TOLERANCE)
    assert shorter is not None and shorter.startswith("line 2: only ROOT goes on")
    assert compare.first_difference(["a"], ["a"], TOLERANCE) is None


def test_streams_are_compared_each_by_its_own_paths():
    paths = ({"/r": "<workdir>"}, {"/x": "<workdir>"})
    assert compare.compare_streams("in /r/f", "in /x/f", paths, TOLERANCE) is None


def _histogram_file(path: Path, counts: list[float]) -> Path:
    h = Histogram.book("h", (len(counts), 0.0, float(len(counts))))
    h.fill(np.arange(len(counts)) + 0.5, weight=np.asarray(counts, dtype=float))
    with create(str(path)) as out:
        out["h"] = h
    return path


def test_root_files_are_compared_key_by_key(tmp_path):
    one = _histogram_file(tmp_path / "a.root", [1, 2, 3])
    two = _histogram_file(tmp_path / "b.root", [1, 2, 4])
    assert compare.compare_root_files(one, one, TOLERANCE) == []
    assert "contents differ" in compare.compare_root_files(one, two, TOLERANCE)[0]
    (tmp_path / "bad.root").write_bytes(b"nope")
    unread = compare.compare_root_files(one, tmp_path / "bad.root", TOLERANCE)
    assert unread[0].startswith("cannot read:")


def _png(path: Path, pixels) -> Path:
    from matplotlib import image

    image.imsave(str(path), np.asarray(pixels, dtype=float), cmap="gray", vmin=0, vmax=1)
    return path


def test_images_score_near_one_alike_and_lower_apart(tmp_path):
    white = _png(tmp_path / "w.png", np.ones((50, 70)))
    black = _png(tmp_path / "b.png", np.zeros((50, 70)))
    other = _png(tmp_path / "o.png", np.ones((40, 60)))
    assert compare.image_similarity(white, white).score == 1.0
    assert compare.image_similarity(white, black).score < 0.1
    assert "sizes" in compare.image_similarity(white, other).note
    (tmp_path / "junk.png").write_bytes(b"not a png")
    assert "cannot decode" in compare.image_similarity(white, tmp_path / "junk.png").note


def _files(tmp_path: Path, side: str, contents: dict) -> tuple[Path, dict]:
    root = tmp_path / side
    root.mkdir()
    for name, text in contents.items():
        (root / name).write_bytes(text)
    return root, {name: runner.checksum(root / name) for name in contents}


def test_outputs_are_compared_by_kind_and_missing_ones_said(tmp_path):
    left, ours = _files(
        tmp_path, "l", {"a.txt": b"v 1\n", "b.pdf": b"1", "c.bin": b"1", "d.txt": b"x"}
    )
    right, theirs = _files(tmp_path, "r", {"a.txt": b"v 2\n", "b.pdf": b"2", "c.bin": b"2"})
    found = {c.name: c for c in compare.compare_files(ours, theirs, (left, right), TOLERANCE)}
    assert found["a.txt"].differences and not found["b.pdf"].differences
    assert found["c.bin"].differences == ["c.bin: contents differ"]
    assert found["d.txt"].differences == ["d.txt: ROOT wrote it, xrdroot did not"]
    assert compare.extra_files(theirs, ours) == ["d.txt"]


def test_image_outputs_carry_their_score(tmp_path):
    (tmp_path / "l").mkdir()
    (tmp_path / "r").mkdir()
    _png(tmp_path / "l" / "c.png", np.ones((20, 20)))
    _png(tmp_path / "r" / "c.png", np.zeros((20, 20)))
    sums = ("1", "2")
    found = compare.compare_file(
        "c.png", tmp_path / "l/c.png", tmp_path / "r/c.png", sums, TOLERANCE
    )
    assert found.score is not None and found.differences[0].startswith("c.png: similarity")
    gone = compare.compare_file("c.png", tmp_path / "x.png", tmp_path / "y.png", sums, TOLERANCE)
    assert "too large" in gone.differences[0]


# --- classification --------------------------------------------------------


def _traceback(last: str) -> str:
    return f'Traceback (most recent call last):\n  File "x.py", line 3, in <module>\n{last}\n'


def test_a_missing_pyroot_name_is_unsupported_by_name():
    stderr = _traceback("AttributeError: ROOT has TLorentzVector; xrdroot.pyroot does not yet")
    assert classify.failure(result(exit_code=1, stderr=stderr)) == (
        "UNSUPPORTED",
        "missing ROOT.TLorentzVector",
    )


def test_a_missing_method_is_named_by_its_class():
    stderr = _traceback("AttributeError: 'TLorentzVector' object has no attribute 'Boost'")
    assert classify.failure(result(exit_code=1, stderr=stderr)).reason == (
        "missing ROOT.TLorentzVector.Boost"
    )
    module = _traceback("AttributeError: module 'xrdroot.pyroot' has no attribute 'TF1'")
    assert classify.failure(result(exit_code=1, stderr=module)).reason == "missing ROOT.TF1"


def test_refusals_by_name_are_unsupported_and_crashes_fail_masked():
    refused = _traceback("xrdroot.errors.UnsupportedFeatureError: TTree::Draw with option 'x'")
    assert classify.failure(result(exit_code=1, stderr=refused)).status == "UNSUPPORTED"
    crashed = _traceback("ValueError: bad value 3.5 at /some/where/file.py line 12")
    assert classify.failure(result(exit_code=1, stderr=crashed)) == (
        "FAIL",
        "ValueError: bad value <n> at <path> line <n>",
    )
    module = _traceback("ModuleNotFoundError: No module named 'xrdroot.pyroot'")
    assert classify.failure(result(exit_code=1, stderr=module)).reason.startswith("ModuleNotFound")


def test_timeouts_signals_and_cli_refusals_have_reasons():
    assert classify.failure(result(exit_code=None, timed_out=True)).reason == "timeout"
    assert classify.failure(result(exit_code=-11)).reason == "killed by signal 11"
    assert classify.failure(result(exit_code=None, stderr="cannot start")).reason.startswith("did")
    said = classify.failure(result(exit_code=2, stderr="xrdroot run: goto is not supported"))
    assert said == ("UNSUPPORTED", "goto is not supported")
    named = classify.failure(
        result(exit_code=2, stderr="xrdroot run: ROOT has TF3; xrdroot.pyroot does not yet")
    )
    assert named.reason == "missing ROOT.TF3"
    assert (
        classify.failure(result(exit_code=2, stderr="xrdroot run: no such file")).status == "FAIL"
    )
    assert classify.failure(result(exit_code=3, stderr="")).reason == "exit 3: no message"


def test_the_oracle_fails_by_its_own_ci_rules():
    assert classify.oracle_failure(result(exit_code=255), 255, ()) is None
    assert classify.oracle_failure(result(exit_code=0), 255, ()) == "ROOT exited 0, CI expects 255"
    printed = classify.oracle_failure(
        result(stderr="Error in <TFile::Open>: no"), 0, ["Error in <"]
    )
    assert printed is not None and printed.startswith("ROOT printed")
    assert classify.oracle_failure(result(timed_out=True), 0, ()) == "ROOT timed out"


# --- running ---------------------------------------------------------------


def test_a_run_is_stopped_at_its_timeout(tmp_path):
    done = runner.execute([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, {}, 0.5)
    assert done.timed_out and done.exit_code is None and done.runtime < 10
    missing = runner.execute([str(tmp_path / "nothing")], tmp_path, {}, 5)
    assert missing.exit_code is None and missing.stderr.startswith("cannot start")


def test_long_streams_keep_their_head_and_tail():
    clipped = runner._clipped("a" * runner.STREAM_LIMIT + "b" * 10)
    assert "elided" in clipped and clipped.endswith("b" * 10)


def test_a_sandbox_holds_neighbours_and_inputs_and_keeps_only_outputs(tmp_path):
    source = tmp_path / "src"
    (source / "data").mkdir(parents=True)
    (source / "t.py").write_text("x")
    (source / "data" / "in.txt").write_text("in")
    inputs = tmp_path / "dep"
    inputs.mkdir()
    (inputs / "hsimple.root").write_text("h")
    code = (
        "import os; assert os.path.exists('hsimple.root') and os.path.exists('data/in.txt');"
        "open('out.txt','w').write('o'); open('lib.so','w').write('');"
        "os.makedirs('sub'); open('sub/deep.txt','w').write('d'); print(os.getcwd())"
    )
    sandbox = runner.Sandbox(source, (inputs,), tmp_path / "keep", tmp_path / "scratch")
    done = runner.run_in_sandbox(sandbox, [sys.executable, "-c", code], {}, 60)
    assert done.exit_code == 0, done.stderr
    assert sorted(done.files) == ["out.txt", "sub/deep.txt"]
    assert (tmp_path / "keep" / "sub" / "deep.txt").read_text() == "d"
    assert done.stdout.strip() == done.workdir


def test_run_results_round_trip_through_json(tmp_path):
    first = result(files={"a": "1"}, note="n")
    first.save(tmp_path / "r.json")
    assert runner.RunResult.load(tmp_path / "r.json") == first


# --- the harness, end to end with stand-ins ----------------------------------


def _script(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\nimport sys, os\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


FAKE_ROOT = """
name = sys.argv[-1]
if 'hsimple' in name:
    open('hsimple.root', 'w').write('h')
    sys.exit(255)
if 'broken' in name:
    print('Error in <TFile::Open>: gone'); sys.exit(0)
assert os.path.exists('hsimple.root')
print('Processing', name, '...'); print('mean', 1.0)
open('out.txt', 'w').write('same')
"""

FAKE_XRDROOT = """
name = sys.argv[-1]
if 'reader' in name:
    raise AttributeError('ROOT has TLorentzVector; xrdroot.pyroot does not yet')
if 'rc' in name:
    print('mean', 2.0)
else:
    print('mean', 1.0000000001)
open('out.txt', 'w').write('same')
"""


@pytest.fixture
def harness(tree, tmp_path):
    (tree / "hist" / "broken.C").write_text("{}")
    (tree / "CMakeLists.txt").write_text(MINI_CMAKE)
    found = cat.catalog(tree, {"ROOT_root_CMD": "root"})
    fake_root = _script(tmp_path / "root", FAKE_ROOT)
    fake_xrd = _script(tmp_path / "xrd", FAKE_XRDROOT)
    oracle = Oracle(root=str(fake_root), version="6.0-test", python=sys.executable)
    settings = Settings(
        tutorials=tree, cache=tmp_path / "cache", jobs=2, timeout=60, only=("hist/",)
    )
    made = Harness(settings, found, oracle, log=lambda line: None)
    made.has_run = True
    made.xrdroot_command = lambda tutorial: [str(fake_xrd), Path(tutorial.path).name]
    return made


def test_the_harness_judges_every_selected_tutorial(harness):
    records = {record["path"]: record for record in harness.run()}
    assert records["hist/writer.C"]["status"] == "PASS"
    assert records["hist/reader.C"]["status"] == "UNSUPPORTED"
    assert records["hist/reader.C"]["reason"] == "missing ROOT.TLorentzVector"
    assert records["hist/rc.C"]["status"] == "ORACLE-FAIL"
    assert records["hist/broken.C"]["reason"].startswith("ROOT printed")
    assert records["hist/helper.py"]["status"] == "SKIP"
    assert "hsimple.C" not in records


def test_oracle_runs_are_cached_by_their_key(harness):
    harness.run()
    again = Harness(harness.settings, harness.tutorials, harness.oracle, log=lambda line: None)
    again.has_run, again.xrdroot_command = True, harness.xrdroot_command
    records = {record["path"]: record for record in again.run()}
    assert records["hist/writer.C"]["oracle"]["cached"] is True
    tutorial = harness.tutorials["hist/writer.C"]
    assert oracle_key(tutorial, harness.settings.tutorials, "a", []) != oracle_key(
        tutorial, harness.settings.tutorials, "b", []
    )


def test_output_differences_make_a_diff(harness):
    harness.tutorials["hist/rc.C"] = cat.Tutorial("hist/rc.C", test="t", depends=("hsimple.C",))
    records = {record["path"]: record for record in harness.run()}
    assert records["hist/rc.C"]["status"] == "DIFF"
    assert records["hist/rc.C"]["reason"] == "stdout"


def test_without_run_or_oracle_the_reasons_say_so(tree, tmp_path):
    found = cat.catalog(tree, {})
    settings = Settings(
        tutorials=tree, cache=tmp_path / "c", only=("hist/writer.C",), xrdroot_python="/no/python"
    )
    made = Harness(settings, found, None, log=lambda line: None)
    (record,) = made.run()
    assert (record["status"], record["reason"]) == ("FAIL", classify.NO_RUN)
    assert made.skip_reason(cat.Tutorial("n.C", labels=("needs_network",))) == "needs the network"


def test_the_oracle_lacking_python_packages_skips(tmp_path):
    oracle = Oracle(root="root", version="v", python=None)
    made = Harness(Settings(tutorials=tmp_path), {}, oracle, log=lambda line: None)
    assert made.skip_reason(cat.Tutorial("a.py")) == "no Python with PyROOT here"
    oracle.modules["torch"] = False
    rich = Oracle(root="root", version="v", python=sys.executable, modules={"torch": False})
    made.oracle = rich
    assert made.skip_reason(cat.Tutorial("a.py", python_deps=("torch",))) == (
        "needs Python package torch"
    )


# --- reports and the command line ------------------------------------------


def _records():
    return [
        {"path": "a/x.C", "area": "a", "status": "FAIL", "reason": "missing ROOT.TF1"},
        {"path": "a/y.C", "area": "a", "status": "UNSUPPORTED", "reason": "missing ROOT.TF1"},
        {"path": "b/z.C", "area": "b", "status": "FAIL", "reason": "timeout", "details": ["t"]},
        {"path": "b/p.C", "area": "b", "status": "PASS", "reason": "", "oracle": {"runtime": 1.0}},
    ]


def test_reasons_are_ranked_by_how_many_tutorials_they_block():
    rows = report.ranked(_records(), report.BLOCKING)
    assert [(row["reason"], row["count"]) for row in rows] == [
        ("missing ROOT.TF1", 2),
        ("timeout", 1),
    ]
    assert rows[0]["statuses"] == {"FAIL": 1, "UNSUPPORTED": 1}


def test_the_document_counts_overall_and_per_area():
    doc = report.document(_records(), {"root_version": "6"})
    assert doc["schema"] == report.SCHEMA and doc["counts"]["FAIL"] == 2
    assert doc["areas"]["b"]["PASS"] == 1 and len(doc["tutorials"]) == 4


def test_markdown_and_html_are_drawn_from_the_document(tmp_path):
    doc = report.document(_records(), {"root_version": "6", "xrdroot_commit": "abc"})
    text = report.markdown(doc)
    assert "| 1 | missing ROOT.TF1 | 2 | a/x.C, a/y.C |" in text
    page = report.page(doc)
    assert "<title>Tutorial conformance</title>" in page and "missing ROOT.TF1" in page
    written = report.write(doc, tmp_path / "out")
    assert [path.name for path in written] == ["results.json", "summary.md", "report.html"]
    assert report._examples(["a", "b", "c", "d"]) == "a, b, c (+1 more)"


def test_the_cli_lists_the_catalogue_and_redraws_reports(tree, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "find_oracle", lambda *args: None)
    assert cli.main(["list", "--tutorials", str(tree), "--only", "hist/"]) == 0
    assert "hist/reader.C\ttutorial-hist-reader" in capsys.readouterr().out
    doc = report.document(_records(), {})
    report.write(doc, tmp_path / "r")
    assert cli.main(["report", "--results", str(tmp_path / "r" / "results.json")]) == 0
    (tmp_path / "old.json").write_text(json.dumps({"schema": "other"}))
    assert cli.main(["report", "--results", str(tmp_path / "old.json")]) == 2


def test_the_cli_runs_uncompared_without_root(tree, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "find_oracle", lambda *args: None)
    argv = [
        "run",
        "--tutorials",
        str(tree),
        "--only",
        "hist/writer.C",
        "--out",
        str(tmp_path / "o"),
    ]
    assert cli.main([*argv, "--python", "/no/python"]) == 0
    doc = json.loads((tmp_path / "o" / "results.json").read_text())
    assert doc["counts"]["FAIL"] == 1 and doc["meta"]["xrdroot_has_run"] is False


def test_the_cli_can_assume_the_build_options_of_a_root_not_here(tree, capsys, monkeypatch):
    monkeypatch.setattr(cli, "find_oracle", lambda *args: None)
    assert cli.main(["list", "--tutorials", str(tree), "--features", "roofit,xml"]) == 0
    assert "roofit/r.C\ttutorial-roofit-r\trc=0" in capsys.readouterr().out


def test_imports_from_root_that_find_nothing_are_missing_names():
    named = _traceback("ImportError: cannot import name 'TCanvas' from 'xrdroot.pyroot' (/x.py)")
    assert classify.failure(result(exit_code=1, stderr=named)).reason == "missing ROOT.TCanvas"
    module = _traceback("ModuleNotFoundError: No module named 'ROOT.VecOps'")
    assert classify.failure(result(exit_code=1, stderr=module)).reason == "missing ROOT.VecOps"


def test_the_cpp_each_tutorial_hands_over_is_masked_from_its_reason():
    stderr = _traceback(
        "xrdroot.errors.UnsupportedFeatureError: gROOT.ProcessLine('.! x') runs C++"
    )
    assert classify.failure(result(exit_code=1, stderr=stderr)).reason == (
        "UnsupportedFeatureError: gROOT.ProcessLine(...) runs C++"
    )


def test_the_oracle_prefix_supplies_root_config_and_its_own_python(tmp_path, monkeypatch):
    from tools.tutorials import environment

    monkeypatch.setattr(environment, "DEFAULT_PREFIX", tmp_path / "absent")
    assert environment.root_config(None) == "root-config"
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "root-config").write_text("")
    (tmp_path / "bin" / "python").write_text("")
    assert environment.root_config(tmp_path) == str(tmp_path / "bin" / "root-config")
    assert environment.python_for("3.99", str(tmp_path / "bin")) == str(tmp_path / "bin" / "python")
    assert environment.python_for("", "") is None
    assert environment.find_oracle(str(tmp_path / "bin" / "nothing")) is None
