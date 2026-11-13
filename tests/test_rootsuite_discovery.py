"""Discovery of ROOT's ctest suite from its CMake, on small synthetic source trees."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.rootsuite import cmake
from tools.rootsuite.discover import DEFAULT_FEATURES, discover
from tools.rootsuite.model import Registry, Test, area_of, kind_of, make_id


def write(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def by_id(root: Path, features=DEFAULT_FEATURES) -> dict:
    return {test.id: test for test in discover(root, features)}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A ROOT-shaped tree: a module with a test directory, test/, and a roottest directory."""
    return write(tmp_path, {
        "CMakeLists.txt": "add_subdirectory(hist)\nadd_subdirectory(missing)\n"
                          "ROOT_ADD_TEST_SUBDIRECTORY(test)\n",
        "hist/CMakeLists.txt": "add_subdirectory(hist)\n",
        "hist/hist/CMakeLists.txt": (
            "function(HIST_ADD_GTEST name source)\n"
            "  ROOT_ADD_GTEST(${name} ${source} LIBRARIES Hist)\n"
            "endfunction()\n"
            "ROOT_ADD_TEST_SUBDIRECTORY(test)\n"),
        "hist/hist/test/CMakeLists.txt": (
            "ROOT_ADD_GTEST(testTH1 test_TH1.cxx LIBRARIES Hist)\n"
            "HIST_ADD_GTEST(test_axis test_axis.cxx)\n"
            "if(cuda)\n  ROOT_ADD_GTEST(testCuda cuda.cxx)\nendif()\n"
            "ROOT_ADD_PYUNITTEST(pyroot_th1 th1.py PYTHON_DEPS numpy)\n"
            "ROOT_ADD_PYUNITTEST(uhi_plot uhi_plot.py GENERIC)\n"
            "ROOT_EXECUTABLE(stressHist stressHist.cxx LIBRARIES Hist)\n"
            "ROOT_ADD_TEST(test-stresshist COMMAND stressHist -b FAILREGEX \"FAILED|Error in\")\n"),
        "hist/hist/test/test_TH1.cxx": "", "hist/hist/test/test_axis.cxx": "",
        "test/CMakeLists.txt": (
            "set(root_exe $<TARGET_FILE:root.exe>)\n"
            "ROOT_STANDARD_LIBRARY_PACKAGE(Event SOURCES Event.cxx LINKDEF EventLinkDef.h)\n"
            "ROOT_EXECUTABLE(eventexe MainEvent.cxx LIBRARIES Event RIO)\n"
            "ROOT_ADD_TEST(test-event COMMAND eventexe)\n"
            "ROOT_EXECUTABLE(stressVector stressVector.cxx LIBRARIES Physics)\n"
            "ROOT_ADD_TEST(test-stressvector COMMAND stressVector FAILREGEX \"FAILED|Error in\")\n"
            "ROOT_ADD_TEST(test-stressvector-interpreted COMMAND ${root_exe} -b -q "
            "${CMAKE_CURRENT_SOURCE_DIR}/stressVector.cxx DEPENDS test-stressvector)\n"),
        "roottest/CMakeLists.txt": "ROOTTEST_ADD_TESTDIRS()\n",
        "roottest/root/CMakeLists.txt": "ROOTTEST_ADD_TESTDIRS()\n",
        "roottest/root/io/CMakeLists.txt": (
            "ROOTTEST_GENERATE_DICTIONARY(libEvDict Ev.h LINKDEF EvLinkDef.h)\n"
            "ROOTTEST_ADD_TEST(write MACRO write.C OUTREF write.ref "
            "DEPENDS ${GENERATE_DICTIONARY_TEST})\n"
            "ROOTTEST_ADD_TEST(read MACRO read.C+ MACROARG \"1, 2\" DEPENDS write helper.C "
            "PASSRC 3 WILLFAIL)\n"
            "ROOTTEST_ADD_TEST(gpu MACRO gpu.C ENABLE_IF cuda)\n"
            "ROOTTEST_ADD_TEST(staged MACRO run.C ROOTEXE_OPTS -e \"#define STAGE 1\")\n"
            "ROOTTEST_ADD_TEST(ls COMMAND ${ROOT_hadd_CMD} -f out.root in.root OUTREF ls.ref)\n"
            "ROOTTEST_ADD_TEST(py MACRO script.py)\n"
            "ROOTTEST_ADD_AUTOMACROS()\n"),
        "roottest/root/io/write.ref": "", "roottest/root/io/write.ref64": "",
        "roottest/root/io/execFoo.C": "", "roottest/root/io/Foo.ref": "",
        "roottest/root/io/assertBar.cxx": "",
    })


def test_the_walk_follows_modules_to_their_test_directories_and_into_roottest(tree):
    found = by_id(tree)
    assert "gtest/hist/hist/testTH1" in found
    assert "test/stressvector" in found
    assert "roottest/root/io/write" in found


def test_a_gtest_gets_ctest_name_kind_and_sources_from_root_add_gtest(tree):
    test = by_id(tree)["gtest/hist/hist/testTH1"]
    assert test.ctest == "gtest-hist-hist-testTH1"
    assert test.kind == "gtest" and test.runner == "gtest"
    assert test.files == ("hist/hist/test/test_TH1.cxx",)


def test_a_function_defined_in_a_module_registers_gtests_from_its_test_directory(tree):
    assert by_id(tree)["gtest/hist/hist/test_axis"].ctest == "gtest-hist-hist-test-axis"


def test_a_feature_the_build_lacks_leaves_its_tests_unregistered(tree):
    assert "gtest/hist/hist/testCuda" not in by_id(tree)
    assert "gtest/hist/hist/testCuda" in by_id(tree, DEFAULT_FEATURES | {"cuda"})


def test_a_pyunittest_is_a_unittest_file_unless_generic_makes_it_a_script(tree):
    found = by_id(tree)
    th1, plot = found["pyunittest/hist/hist/pyroot_th1"], found["pyunittest/hist/hist/uhi_plot"]
    assert th1.ctest == "pyunittests-hist-hist-pyroot-th1"
    assert (th1.runner, th1.python_deps) == ("pyunittest", ("numpy",))
    assert plot.runner == "script"


def test_a_root_add_test_of_a_built_program_runs_its_sources_with_its_arguments(tree):
    test = by_id(tree)["test/hist/hist/stresshist"]
    assert (test.runner, test.files, test.args) == ("program", ("hist/hist/test/stressHist.cxx",),
                                                    ("-b",))
    assert test.failregex == ("FAILED|Error in",)


def test_a_program_linking_a_test_library_with_a_linkdef_needs_a_dictionary(tree):
    test = by_id(tree)["test/event"]
    assert "dictionary" in test.needs
    assert test.files == ("test/MainEvent.cxx", "test/Event.cxx")


def test_a_root_exe_command_is_a_root_runner_with_its_arguments(tree):
    test = by_id(tree)["test/stressvector-interpreted"]
    assert test.runner == "root" and test.args[:2] == ("-b", "-q")
    assert test.depends == ("test-stressvector",)


def test_a_roottest_macro_keeps_its_reference_preferring_the_64_bit_one(tree):
    test = by_id(tree)["roottest/root/io/write"]
    assert test.ctest == "roottest-root-io-write"
    assert test.outref == "roottest/root/io/write.ref64"
    assert test.check_out and test.check_err and test.custom_diff


def test_a_roottest_depending_on_a_generated_dictionary_needs_one(tree):
    found = by_id(tree)
    assert found["roottest/root/io/write"].needs == ("dictionary",)
    assert found["roottest/root/io/libEvDict-build"].setup_of == "dictionary"


def test_a_source_dependency_becomes_a_compile_step_and_names_are_prefixed(tree):
    found = by_id(tree)
    read = found["roottest/root/io/read"]
    assert read.depends == ("roottest-root-io-write", "roottest-root-io-helper-build")
    helper = found["roottest/root/io/helper-build"]
    assert helper.setup_of == "compiled-macro" and helper.failregex == ("Warning in",)
    assert helper.builds == (("roottest/root/io", "aclic", "<build>/roottest/root/io/helper.C+kc"),)
    assert "compile-macro-helper.C-fixture" in read.fixtures_required
    assert (read.files, read.args[-1], read.passrc, read.will_fail) == (
        ("roottest/root/io/read.C",), f"{tree}/roottest/root/io/read.C+(1, 2)", 3, True)


def test_enable_if_drops_a_roottest_the_features_do_not_enable(tree):
    assert "roottest/root/io/gpu" not in by_id(tree)


def test_a_roottest_macro_is_root_exe_with_the_macrotest_lines_then_its_own_options(tree):
    args = by_id(tree)["roottest/root/io/staged"].args
    assert args[:2] == ("-e", 'gSystem->SetBuildDir("<build>/roottest/root/io",true)')
    assert args[-5:] == ("-e", "#define STAGE 1", "-q", "-b", f"{tree}/roottest/root/io/run.C")


def test_a_command_test_checks_its_output_only_when_it_has_a_reference(tree):
    test = by_id(tree)["roottest/root/io/ls"]
    assert test.runner == "command" and test.command[0] == "hadd"
    assert test.check_out and test.outref == "roottest/root/io/ls.ref"


def test_automacros_register_run_exec_and_assert_macros_with_their_references(tree):
    found = by_id(tree)
    foo, bar = found["roottest/root/io/execFoo-auto"], found["roottest/root/io/assertBar-auto"]
    assert foo.outref == "roottest/root/io/Foo.ref"
    assert bar.check_err and not bar.check_out


def test_a_python_roottest_macro_is_a_script(tree):
    assert by_id(tree)["roottest/root/io/py"].runner == "script"


def test_without_module_cmakelists_every_test_directory_is_taken(tmp_path):
    write(tmp_path, {"math/foo/test/CMakeLists.txt": "ROOT_ADD_GTEST(testFoo foo.cxx)\n",
                     "roottest/x/test/CMakeLists.txt": "ROOT_ADD_GTEST(testRt rt.cxx)\n"})
    found = by_id(tmp_path, frozenset())
    assert "gtest/math/foo/testFoo" in found and "gtest/roottest/x/testRt" not in found


def test_the_tutorials_test_directory_is_the_tutorial_harnesss_and_names_repeat_uniquely(tmp_path):
    write(tmp_path, {"CMakeLists.txt": "ROOT_ADD_TEST_SUBDIRECTORY(tutorials)\n"
                                       "ROOT_ADD_TEST_SUBDIRECTORY(test)\n",
                     "tutorials/CMakeLists.txt": "ROOT_ADD_GTEST(never t.cxx)\n",
                     "test/CMakeLists.txt": "ROOT_ADD_TEST(test-a COMMAND true)\n"
                                            "ROOT_ADD_TEST(test-a COMMAND true)\n"})
    assert list(by_id(tmp_path)) == ["test/a", "test/a#2"]


def test_a_test_round_trips_through_json_with_its_builds():
    test = Test(id="t", ctest="c", kind="gtest", directory="d", runner="gtest",
                builds=(("d", "library", "--name", "x"),), args=("a",))
    assert Test.from_json({**test.to_json(), "unknown": 1}) == test
    assert "skip" not in test.to_json()


def test_kinds_ids_and_areas_follow_the_ctest_names():
    assert kind_of("gtest-a", "x") == "gtest" and kind_of("show-env", "test") == "test"
    assert kind_of("mathcore-x", "math/mathcore/test") == "other"
    assert make_id("roottest", "roottest/root/io", "x") == "roottest/root/io/x"
    assert area_of("roottest/root/io/x") == "root/io" and area_of("gtest/hist/h/t") == "hist"
    assert area_of("test/stress") == "test"


# --- the interpreter's additions ------------------------------------------


def interpreter(root: Path, source: str) -> cmake.Directory:
    directory = cmake.Directory(Registry(str(root)), "", {})
    return directory.run(source)


def test_functions_have_their_own_scope_and_parent_scope_reaches_the_caller(tmp_path):
    it = interpreter(tmp_path, (
        "function(f a)\n  set(inner ${a})\n  set(out ${a}-${ARGN} PARENT_SCOPE)\nendfunction()\n"
        "f(x y z)\n"))
    assert it.variables["out"] == "x-y;z"
    assert "inner" not in it.variables


def test_macros_set_variables_in_the_callers_scope_and_return_ends_a_function(tmp_path):
    it = interpreter(tmp_path, (
        "macro(m v)\n  set(seen ${v})\nendmacro()\n"
        "function(f)\n  set(a 1 PARENT_SCOPE)\n  return()\n  set(b 1 PARENT_SCOPE)\nendfunction()\n"
        "m(1)\nf()\n"))
    assert (it.variables["seen"], it.variables["a"]) == ("1", "1")
    assert "b" not in it.variables


def test_cmake_parse_arguments_splits_options_single_and_multi_values(tmp_path):
    it = interpreter(tmp_path, (
        'cmake_parse_arguments(ARG "FAST" "NAME" "FILES" FAST NAME n FILES a b extra)\n'))
    assert it.variables["ARG_FAST"] == "TRUE" and it.variables["ARG_NAME"] == "n"
    assert it.variables["ARG_FILES"] == "a;b;extra"


def test_string_list_and_filename_commands_compute_their_results(tmp_path):
    it = interpreter(tmp_path, (
        'string(REGEX REPLACE "[_]" "-" good a_b_c)\nstring(TOUPPER abc up)\n'
        "set(l c a b)\nlist(SORT l)\nlist(GET l 0 first)\n"
        "get_filename_component(stem dir/name.ext.C NAME_WE)\n"))
    assert it.variables["good"] == "a-b-c" and it.variables["up"] == "ABC"
    assert it.variables["l"] == "a;b;c" and it.variables["first"] == "a"
    assert it.variables["stem"] == "name"


def test_if_exists_looks_in_the_current_source_directory(tmp_path):
    (tmp_path / "here.txt").write_text("")
    it = interpreter(tmp_path, "if(EXISTS here.txt)\nset(a 1)\nendif()\n"
                               "if(EXISTS gone.txt)\nset(b 1)\nendif()\n")
    assert "a" in it.variables and "b" not in it.variables


def test_blocks_branch_loop_and_skip_what_decides_no_test(tmp_path):
    it = interpreter(tmp_path, (
        "endforeach()\nwhile(x)\n  set(never 1)\nendwhile()\nfunction()\nendfunction()\n"
        "if(NO)\n  set(a 1)\nelseif(YES)\n  set(a 2)\nelse()\n  set(a 3)\nendif()\n"
        "if(NO)\nelse()\n  set(b 1)\nendif()\n"
        "foreach(x 1 2)\n  set(last ${x})\nendforeach()\n"
        "message(STATUS hello)\n"))
    assert (it.variables["a"], it.variables["b"], it.variables["last"]) == ("2", "1", "2")
    assert "never" not in it.variables


def test_scopes_options_and_subdirectories_at_their_edges(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/CMakeLists.txt").write_text("set(seen 1 PARENT_SCOPE)\nreturn()\nset(x 1)\n")
    it = interpreter(tmp_path, (
        "set(top 1 PARENT_SCOPE)\noption(OPT \"doc\" ON)\noption(OPT \"doc\" OFF)\n"
        "function(f)\n  set(gone PARENT_SCOPE)\nendfunction()\nset(gone 1)\nf()\n"
        "add_subdirectory()\nadd_subdirectory(/elsewhere)\nadd_subdirectory(sub)\n"
        "add_subdirectory(sub)\nadd_subdirectory(missing)\n"))
    assert it.variables["OPT"] == "ON" and "gone" not in it.variables
    assert it.variables["seen"] == "1" and "top" not in it.variables
    assert it.subdirectory("") is None


def test_parse_arguments_from_argv_and_too_few_words_are_handled(tmp_path):
    it = interpreter(tmp_path, (
        "function(f first)\n"
        '  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "N" "")\n'
        "  set(got ${ARG_N} PARENT_SCOPE)\nendfunction()\nf(skipped N value)\n"
        "function(g)\n  cmake_parse_arguments(PARSE_ARGV x ARG \"\" \"N\" \"\")\n"
        "  set(other ${ARG_N} PARENT_SCOPE)\nendfunction()\ng(N v)\n"
        "cmake_parse_arguments(A B)\n"))
    assert it.variables["got"] == "value" and it.variables["other"] == "v"


def test_string_and_filename_commands_at_their_edges(tmp_path):
    it = interpreter(tmp_path, (
        'string(REGEX REPLACE "[" "-" bad a[b)\nstring(REGEX MATCH a m abc)\n'
        "string(TOUPPER only)\nstring(APPEND s x y)\nstring(APPEND)\n"
        "string(MAKE_C_IDENTIFIER a.b-c id)\nstring(MAKE_C_IDENTIFIER x)\n"
        "string(LENGTH abc n)\nset(l a)\nlist(APPEND l b)\n"
        "get_filename_component(e dir/n.x.C EXT)\nget_filename_component(q n.C UNKNOWN)\n"
        "include_directories(BEFORE inc)\n"))
    assert (it.variables["bad"], it.variables["s"], it.variables["id"]) == ("a[b", "xy", "a_b_c")
    assert it.variables["e"] == ".x.C" and "q" not in it.variables and "m" not in it.variables
    assert it.variables["l"] == "a;b" and it.includes() == ["inc"]


def test_paths_inside_the_tree_and_the_build_tree_are_kept_others_dropped(tmp_path):
    it = interpreter(tmp_path, "")
    assert it.inside("<build>/a/b") == "a/b" and it.inside("x/y") == "x/y"
    assert it.inside("/elsewhere") is None and it.inside("$<X:y>") is None


def test_a_quoted_list_is_split_as_roots_macros_split_argn_and_escapes_are_undone(tmp_path):
    assert cmake.parse_arguments(["F", "a;b", "c\\;d", "x;;"], [], [], ["F"]) == {
        "UNPARSED": [], "F": ["a", "b", "c\\;d", "x"]}
    it = interpreter(tmp_path, 'set(dir with\\ space/x.C)\nset(quoted "q\\ r")\n')
    assert it.variables["dir"] == "with space/x.C" and it.variables["quoted"] == "q r"


def test_root_ctest_adds_the_module_test_directories_before_roottest(tmp_path):
    write(tmp_path, {
        "CMakeLists.txt": "ROOT_ADD_TEST_SUBDIRECTORY(test)\ninclude(RootCTest)\ninclude(Other)\n"
                          "add_subdirectory(roottest)\n",
        "test/CMakeLists.txt": "ROOT_EXECUTABLE(prog prog.cxx)\n",
        "roottest/CMakeLists.txt": "ROOTTEST_ADD_TEST(uses COMMAND <build>/test/prog)\n"})
    test = by_id(tmp_path)["roottest/uses"]
    assert (test.runner, test.files) == ("program", ("test/prog.cxx",))


@pytest.fixture
def edges(tmp_path: Path) -> Path:
    return write(tmp_path, {
        "CMakeLists.txt": "ROOT_ADD_TEST_SUBDIRECTORY(test)\nadd_subdirectory(core)\n"
                          "ROOT_ADD_TEST_SUBDIRECTORY()\n",
        "core/CMakeLists.txt": "ROOT_LINKER_LIBRARY(Core core.cxx)\n"
                               "ROOT_EXECUTABLE(root.exe main.cxx)\n",
        "test/CMakeLists.txt": (
            "ROOT_EXECUTABLE()\nROOT_LINKER_LIBRARY()\nROOT_STANDARD_LIBRARY_PACKAGE()\n"
            "ROOT_ADD_GTEST()\nROOT_ADD_PYUNITTEST(only)\nROOT_ADD_PYUNITTESTS()\n"
            "ROOT_ADD_TEST()\nadd_executable()\nadd_library()\ntarget_link_libraries()\n"
            "target_include_directories()\ntarget_sources()\n"
            "add_executable(tool tool.cxx WIN32)\nROOT_ADD_TEST(test-tool COMMAND tool)\n"
            "ROOT_ADD_UNITTEST_DIR(Core)\nROOT_ADD_PYUNITTESTS(all_py)\n"
            "ROOT_ADD_TEST(test-empty COMMAND)\n"
            "ROOT_ADD_TEST(test-py COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/s.py -v)\n"
            "set_property(TEST test-tool PROPERTY)\n"),
        "test/u1.cxx": "", "test/a.py": "",
        "roottest/CMakeLists.txt": "ROOTTEST_ADD_TESTDIRS(EXCLUDED_DIRS skipped)\n",
        "roottest/skipped/CMakeLists.txt": "ROOTTEST_ADD_TEST(never MACRO x.C)\n",
        "roottest/r/CMakeLists.txt": (
            "ROOTTEST_ADD_TEST()\nROOTTEST_ADD_TEST(nothing COMMAND)\n"
            "ROOTTEST_ADD_TEST(off MACRO x.C DISABLE_IF cocoa)\n"
            "ROOTTEST_ADD_AUTOMACROS(EXCLUDE runSkip WILLFAIL execBad DEPENDS dep.C "
            "FIXTURES_REQUIRED fx)\n"
            "ROOTTEST_ADD_UNITTEST_DIR(Core WILLFAIL)\nROOTTEST_ADD_OLDTEST()\n"
            "ROOTTEST_TARGETNAME_FROM_FILE(named sub/f.C)\nROOTTEST_TARGETNAME_FROM_FILE(x)\n"
            "find_python_module(numpy)\nfind_python_module()\n"
            "ROOTTEST_ADD_TEST(${named} MACRO x.C)\n"),
        "roottest/r/runSkip.C": "", "roottest/r/execBad.C": "", "roottest/r/u.cxx": "",
        "roottest/r/assertOkay.C": "",
    })


def test_macros_called_without_arguments_register_nothing(edges):
    found = by_id(edges)
    assert "gtest/only" not in found and "test/empty" in found
    assert found["test/empty"].skip == "the test has no command"


def test_roots_own_programs_are_not_test_programs(edges):
    tool = by_id(edges)["test/tool"]
    assert tool.runner == "program" and tool.files == ("test/tool.cxx",)


def test_unittest_dirs_and_pyunittests_collect_their_directorys_files(edges):
    found = by_id(edges)
    assert found["gtest/testUnit"].files == ("test/u1.cxx",)
    assert found["pyunittest/all_py"].files == ("test/a.py",)
    assert found["test/py"].runner == "script" and found["test/py"].args == ("-v",)
    unit = found["roottest/r/unittests"]
    assert unit.runner == "gtest" and unit.will_fail and "roottest/r/u.cxx" in unit.files


def test_roottests_edges_automacros_oldtests_and_target_names(edges):
    found = by_id(edges)
    assert "roottest/r/off" not in found and "roottest/skipped/never" not in found
    assert "roottest/r/runSkip-auto" not in found
    bad = found["roottest/r/execBad-auto"]
    assert bad.will_fail and "fx" in bad.fixtures_required
    assert bad.depends == ("roottest-r-dep-build",)
    assert found["roottest/r/make"].skip.startswith("runs roottest's old Makefile")
    assert "roottest/r/f" in found
    assert found["roottest/r/assertOkay-auto"].depends == bad.depends
