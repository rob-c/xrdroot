"""Discovery of ROOT's ctest suite from its CMake, on small synthetic source trees."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.rootsuite import cmake
from tools.rootsuite.discover import DEFAULT_FEATURES, discover
from tools.rootsuite.model import Registry, area_of, kind_of, make_id


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
    assert found["roottest/root/io/helper-build"].setup_of == "compiled-macro"
    assert (read.files, read.args, read.passrc, read.will_fail) == (
        ("roottest/root/io/read.C",), ("1, 2",), 3, True)


def test_enable_if_drops_a_roottest_the_features_do_not_enable(tree):
    assert "roottest/root/io/gpu" not in by_id(tree)


def test_rootexe_options_are_a_skip_with_their_reason(tree):
    assert "passes root.exe options" in by_id(tree)["roottest/root/io/staged"].skip


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
    write(tmp_path, {"math/foo/test/CMakeLists.txt": "ROOT_ADD_GTEST(testFoo foo.cxx)\n"})
    assert "gtest/math/foo/testFoo" in by_id(tmp_path)


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
