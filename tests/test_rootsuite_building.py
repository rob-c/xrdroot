"""ROOT's build model as discovery sees it: the machine, dictionaries, libraries, executables."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest
from tools.rootsuite import builds, cli, dictgen, machine
from tools.rootsuite.discover import discover, variables
from tools.rootsuite.model import Dictionary, Registry, Target, Test


def write(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def by_id(root: Path, features=machine.DEFAULT_FEATURES) -> dict[str, Test]:
    return {test.id: test for test in discover(root, features)}


# --- the machine -----------------------------------------------------------------


def oracle(tmp_path: Path, script: str) -> Path:
    config = tmp_path / "oracle/bin/root-config"
    config.parent.mkdir(parents=True)
    config.write_text(script)
    config.chmod(config.stat().st_mode | stat.S_IEXEC)
    return tmp_path / "oracle"


def test_the_oracles_features_are_the_machines_less_what_no_mac_builds_plus_testing(tmp_path):
    prefix = oracle(tmp_path, "#!/bin/sh\necho 'cocoa imt x11 cuda'\n")
    assert machine.default_features(prefix) == {"cocoa", "imt", "testing", "roottest"}


def test_without_an_oracle_the_committed_macos_arm64_features_are_used(tmp_path, monkeypatch):
    monkeypatch.setattr(machine, "DEFAULT_PREFIX", tmp_path / "none")
    assert machine.oracle_config(None) is None
    assert machine.default_features(None) == machine.DEFAULT_FEATURES
    assert len(machine.MACOS_ARM64_FEATURES) == 43 and "cocoa" in machine.MACOS_ARM64_FEATURES
    assert not machine.MACOS_ARM64_FEATURES & machine.NOT_ON_MACOS


def test_an_oracle_that_fails_or_says_nothing_gives_no_features(tmp_path):
    assert machine.oracle_features(oracle(tmp_path, "#!/bin/sh\nexit 1\n") / "bin/root-config") \
        is None
    assert machine.oracle_features(tmp_path / "missing/root-config") is None


def test_the_cmake_variables_are_a_macs_on_arm64(tmp_path):
    found = variables(tmp_path, ["imt"])
    assert (found["APPLE"], found["CMAKE_SYSTEM_NAME"], found["CMAKE_SYSTEM_PROCESSOR"]) == (
        "1", "Darwin", "arm64")
    assert found["imt"] == "ON" and "LINUX" not in found and "x11" not in found


def test_a_test_roots_cmake_disables_on_apple_arm64_is_not_registered(tmp_path):
    write(tmp_path, {"math/x/test/CMakeLists.txt": (
        "if(NOT (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES arm64))\n"
        "  ROOT_ADD_GTEST(testLinux linux.cxx)\nendif()\nROOT_ADD_GTEST(testAll all.cxx)\n")})
    assert list(by_id(tmp_path)) == ["gtest/math/x/testAll"]


def test_the_command_line_asks_the_oracle_once_unless_features_are_given(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "default_features", lambda: calls.append(1) or frozenset({"a"}))
    args = cli.parser().parse_args(["list", "--root", "."])
    assert cli._features(args) == cli._features(args) == {"a"} and calls == [1]
    given = cli.parser().parse_args(["list", "--root", ".", "--features", "b,c"])
    assert cli._features(given) == {"b", "c"}


# --- build commands ----------------------------------------------------------------


def test_library_names_drop_lib_as_root_writes_them():
    assert builds.library_name("libEvent") == "Event"
    assert builds.library_name("Event") == "Event" and builds.library_name("lib") == "lib"


def test_a_library_is_its_dictionaries_then_its_sources_under_one_name():
    entry = Dictionary("G__E", "d", ["d/E.h"], linkdef="d/L.h", includes=["d", "inc"])
    target = Target(["d/E.cxx", "d/G__F.cxx"], directory="d", output="E",
                    dictionaries=[entry])
    registry = Registry("/r", dictionaries={"d/G__F.cxx": Dictionary("G__F", "d", ["d/F.h"],
                                                                    selection="d/s.xml")})
    assert builds.library(registry, target) == [
        ("d", "dictionary", "--name", "E", "--linkdef", "<build>/d/L.h", "-I", "<build>/d",
         "-I", "<build>/inc", "<build>/d/E.h"),
        ("d", "dictionary", "--name", "E", "--selection", "<build>/d/s.xml", "-I", "<build>/d",
         "<build>/d/F.h"),
        ("d", "library", "--name", "E", "-I", "<build>/d", "<build>/d/E.cxx")]


def test_an_executable_links_its_libraries_and_its_own_dictionaries():
    lib = Target(["d/L.cxx"], ["Other"], directory="d", output="L")
    other = Target(["d/O.cxx"], directory="d", output="O")
    exe = Target(["d/main.cxx", "d/G__X.cxx"], ["L", "Core"], directory="d", output="exe")
    registry = Registry("/r", libraries={"L": lib, "Other": other},
                        dictionaries={"d/G__X.cxx": Dictionary("G__X", "d", ["d/X.h"])})
    steps = builds.executable(registry, exe, "exe")
    assert [step[1] for step in steps] == ["library", "library", "dictionary", "executable"]
    assert steps[-1] == ("d", "executable", "--name", "exe", "--library", "L", "--library", "O",
                         "--library", "G__X", "-I", "<build>/d", "<build>/d/main.cxx")
    assert [step[1] for step in builds.executable(registry, exe, "exe", libraries=False)] == [
        "dictionary", "executable"]


def test_aclic_compiles_a_macro_with_root_tests_options():
    assert builds.aclic("d", "d/m.C") == ("d", "aclic", "<build>/d/m.C+kc")


# --- discovered builds -------------------------------------------------------------


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return write(tmp_path, {
        "CMakeLists.txt": "ROOT_ADD_TEST_SUBDIRECTORY(test)\n",
        "test/CMakeLists.txt": (
            "include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc SYSTEM /usr/include)\n"
            "ROOT_STANDARD_LIBRARY_PACKAGE(Event NO_INSTALL_HEADERS HEADERS "
            "${CMAKE_CURRENT_SOURCE_DIR}/Event.h SOURCES Event.cxx LINKDEF EventLinkDef.h)\n"
            "ROOT_EXECUTABLE(eventexe MainEvent.cxx LIBRARIES Event RIO)\n"
            "ROOT_ADD_TEST(test-event COMMAND eventexe)\n"
            "ROOT_ADD_GTEST(testDict Dict.cxx ElemDict.cxx INCLUDE_DIRS "
            "${CMAKE_CURRENT_SOURCE_DIR}/more /elsewhere REPEATS 2)\n"
            "ROOT_GENERATE_DICTIONARY(ElemDict ${CMAKE_CURRENT_SOURCE_DIR}/Elem.h "
            "LINKDEF ElemLinkDef.h)\n"
            "ROOT_ADD_GTEST(testMod Mod.cxx)\n"
            "ROOT_GENERATE_DICTIONARY(ModDict ${CMAKE_CURRENT_SOURCE_DIR}/Mod.h MODULE testMod)\n"
            "ROOT_GENERATE_DICTIONARY()\n"
            "ROOT_LINKER_LIBRARY(Hello Hello.cxx TEST LIBRARIES Graf)\n"
            "target_sources(Hello PRIVATE More.cxx)\n"
            "target_include_directories(Hello PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hi $<X>)\n"
            "target_link_libraries(Hello PUBLIC Event)\n"
            "target_link_libraries(NotATarget Event)\n"
            "add_library(Plain SHARED Plain.cxx)\n"
            "ROOT_EXECUTABLE(stress stress.cxx LIBRARIES Physics)\n"
            "ROOT_ADD_TEST(test-stress COMMAND stress -b)\n"
            "ROOT_ADD_PYUNITTEST(py py.py PRECMD sh pre.sh POSTCMD sh post.sh COPY_TO_BUILDDIR "
            "data.root)\n"
            "add_test(NAME test-build-hello COMMAND cmake --build <build> --target Hello/fast "
            "-- -k)\n"
            "add_test(NAME test-build-dict COMMAND cmake --build <build> --target ModDict)\n"
            "add_test(NAME test-build-none COMMAND cmake --build <build> --target Nothing)\n"
            "add_test(NAME test-build-what COMMAND cmake --build <build> --target What)\n"
            "add_test(NAME test-build-exe COMMAND cmake --build <build> --target eventexe)\n"
            "add_test(NAME test-plain COMMAND true WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}"
            "/sub)\n"
            "add_test(test-oldstyle true)\n"
            "set_property(TEST test-plain APPEND PROPERTY FIXTURES_REQUIRED a b)\n"
            "set_property(TEST test-plain APPEND PROPERTY FIXTURES_REQUIRED c)\n"
            "set_property(TEST test-stress PROPERTY WILL_FAIL ON)\n"
            "set_property(TEST test-stress)\n"
            "set_property(DIRECTORY PROPERTY X)\n"
            "set_tests_properties(test-event PROPERTIES PASS_REGULAR_EXPRESSION ok TIMEOUT 30 "
            "DISABLED OFF RESOURCE_LOCK x)\n"
            "set_tests_properties(test-build-none PROPERTIES DISABLED True)\n"
            "set_tests_properties(test-nothing)\n"),
        "test/sub/x": "",
    })


def test_a_program_linking_a_dictionary_library_is_built_then_run_as_itself(tree):
    test = by_id(tree)["test/event"]
    assert [step[1] for step in test.builds] == ["dictionary", "library", "executable"]
    assert test.builds[0] == ("test", "dictionary", "--name", "Event", "--linkdef",
                              "<build>/test/EventLinkDef.h", "-I", "<build>/test", "-I",
                              "<build>/test/inc", "<build>/test/Event.h")
    assert test.launch == "<build>/test/eventexe" and test.needs == ("dictionary",)
    assert (test.passregex, test.timeout) == (("ok",), 30.0)


def test_a_gtest_with_a_dictionary_generated_after_it_is_built_with_it(tree):
    found = by_id(tree)
    test = found["gtest/testDict"]
    assert [step[1] for step in test.builds] == ["dictionary", "executable"]
    assert "--library" in test.builds[1] and "<build>/test/more" in test.builds[1]
    assert test.args == ("--gtest_repeat=2", "--gtest_break_on_failure")
    assert found["gtest/testMod"].builds[0][3] == "ModDict"


def test_a_program_needing_nothing_built_runs_through_xrdroot_run(tree):
    test = by_id(tree)["test/stress"]
    assert (test.builds, test.launch, test.files, test.will_fail) == (
        (), "", ("test/stress.cxx",), True)


def test_cmake_build_tests_build_their_targets_and_a_disabled_test_is_gone(tree):
    found = by_id(tree)
    hello = found["test/build-hello"]
    assert hello.runner == "setup" and hello.args == ("Hello",)
    assert [step[1] for step in hello.builds] == ["library"]
    assert "<build>/test/More.cxx" in hello.builds[0] and "<build>/test/hi" in hello.builds[0]
    assert found["test/build-dict"].builds[0][1:4] == ("dictionary", "--name", "ModDict")
    assert found["test/build-exe"].builds[-1][1] == "executable"
    assert "test/build-none" not in found
    assert found["test/build-what"].skip.startswith("builds What, which no CMake here defines")


def test_cmakes_own_tests_and_their_properties_are_kept(tree):
    plain = by_id(tree)["test/plain"]
    assert plain.runner == "command" and plain.working_dir == "test/sub"
    assert plain.fixtures_required == ("a", "b", "c")
    assert "test/oldstyle" not in by_id(tree)


def test_a_pyunittest_keeps_its_pre_and_post_commands_and_copies(tree):
    test = by_id(tree)["pyunittest/py"]
    assert (test.precmd, test.postcmd, test.copy) == (("sh", "pre.sh"), ("sh", "post.sh"),
                                                      ("test/data.root",))


def test_a_command_naming_no_program_built_here_runs_as_written(tmp_path):
    write(tmp_path, {"roottest/CMakeLists.txt": "ROOTTEST_ADD_TESTDIRS()\n",
                     "roottest/a/CMakeLists.txt": (
                         "ROOTTEST_ADD_TEST(script EXEC ${CMAKE_CURRENT_SOURCE_DIR}/run.sh "
                         "OPTS -v)\n")})
    test = by_id(tmp_path)["roottest/a/script"]
    assert (test.runner, test.command[-1], test.program) == ("command", "-v", "")


@pytest.fixture
def roottest(tmp_path: Path) -> Path:
    return write(tmp_path, {
        "roottest/CMakeLists.txt": "include_directories(${CMAKE_CURRENT_SOURCE_DIR})\n"
                                   "ROOTTEST_ADD_TESTDIRS()\n",
        "roottest/io/CMakeLists.txt": (
            "ROOTTEST_GENERATE_DICTIONARY(libEvDict Ev.h LINKDEF EvLinkDef.h SOURCES Ev.cxx "
            "FIXTURES_SETUP ev)\n"
            "ROOTTEST_GENERATE_DICTIONARY(Bare Bare.h)\n"
            "ROOTTEST_GENERATE_REFLEX_DICTIONARY(Refl R.h SELECTION sel.xml)\n"
            "ROOTTEST_GENERATE_REFLEX_DICTIONARY(Named N.h SELECTION sel.xml LIBNAME libNamed "
            "FIXTURES_SETUP more)\n"
            "ROOTTEST_GENERATE_DICTIONARY()\nROOTTEST_GENERATE_REFLEX_DICTIONARY()\n"
            "ROOTTEST_GENERATE_EXECUTABLE()\nROOTTEST_COMPILE_MACRO()\n"
            "ROOTTEST_GENERATE_EXECUTABLE(exe main.cxx LIBRARIES libEvDictlibgen Core "
            "FIXTURES_SETUP exe)\n"
            "ROOTTEST_ADD_TEST(runexe EXEC ./exe OPTS -v FIXTURES_REQUIRED exe ev)\n"
            "ROOTTEST_ADD_TEST(after MACRO after.C DEPENDS ${GENERATE_EXECUTABLE_TEST})\n"
            "ROOTTEST_ADD_TEST(tut MACRO ${ROOT_TUTORIALS_DIR}/hsimple.C PASSRC 255)\n"
            "ROOTTEST_ADD_TEST(cmd MACRO .x WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR} "
            "COPY_TO_BUILDDIR d.root PRECMD sh p.sh POSTCMD sh q.sh)\n"
            "ROOTTEST_ADD_TEST(inline MACRO \"run.C(3)\")\n"
            "ROOTTEST_ADD_TEST(py MACRO s.py OPTS -x)\n"
            "ROOTTEST_COMPILE_MACRO(lib.C FIXTURES_SETUP lib)\n"),
    })


def test_roottest_dictionaries_build_their_library_named_as_root_names_it(roottest):
    found = by_id(roottest)
    ev = found["roottest/io/libEvDict-build"]
    assert ev.setup_of == "dictionary" and ev.fixtures_setup == ("ev",)
    assert [step[1:4] for step in ev.builds] == [("dictionary", "--name", "EvDict"),
                                                ("library", "--name", "EvDict")]
    assert "<build>/roottest" in ev.builds[0]
    assert [step[1] for step in found["roottest/io/Bare-build"].builds] == ["dictionary"]


def test_reflex_dictionaries_name_their_library_and_set_up_their_own_fixture(roottest):
    found = by_id(roottest)
    refl = found["roottest/io/Refl-libgen-build"]
    assert refl.builds[0][1:6] == ("dictionary", "--name", "Refl_dictrflx", "--selection",
                                   "<build>/roottest/io/sel.xml")
    assert refl.fixtures_setup == ("roottest-io-Refl-libgen-build-fixture",)
    named = found["roottest/io/Named-libgen-build"]
    assert named.builds[0][3] == "Named" and "more" in named.fixtures_setup


def test_a_roottest_executable_is_built_by_its_setup_and_run_by_its_test(roottest):
    found = by_id(roottest)
    build = found["roottest/io/exe-build"]
    assert build.builds == (("roottest/io", "executable", "--name", "exe", "--library",
                             "EvDict", "-I", "<build>/roottest/io", "-I", "<build>/roottest",
                             "<build>/roottest/io/main.cxx"),)
    run = found["roottest/io/runexe"]
    assert (run.launch, run.args, run.builds) == ("<build>/roottest/io/exe", ("-v",), ())
    assert run.needs == ("dictionary", "executable")
    assert "exe" in found["roottest/io/after"].fixtures_required


def test_a_compiled_macro_step_fails_on_warnings(roottest):
    lib = by_id(roottest)["roottest/io/lib-build"]
    assert lib.failregex == ("Warning in",) and lib.fixtures_setup == ("lib",)


def test_roottest_macros_name_their_files_and_pass_commands_through(roottest):
    found = by_id(roottest)
    tut = found["roottest/io/tut"]
    assert tut.files == ("tutorials/hsimple.C",) and tut.args[-1] == "<build>/tutorials/hsimple.C"
    cmd = found["roottest/io/cmd"]
    assert (cmd.args[-1], cmd.files, cmd.working_dir) == (".x", (), "")
    assert (cmd.copy, cmd.precmd, cmd.postcmd) == (("roottest/io/d.root",), ("sh", "p.sh"),
                                                    ("sh", "q.sh"))
    assert found["roottest/io/inline"].args[-1] == f"{roottest}/roottest/io/run.C(3)"
    assert found["roottest/io/py"].args == ("-x",)


# --- dictionary generators run by tests --------------------------------------------


def test_genreflex_names_its_dictionary_after_its_library_output_or_header():
    words = dictgen.dictionary_words
    assert words("genreflex", ["a.h", "-l", "/d/libMy.dylib", "--rootmap", "m"])[:3] == [
        "dictionary", "--name", "My"]
    assert words("genreflex", ["a.h", "--rootmap-lib=libR.so", "-s", "s.xml"])[2] == "R"
    assert words("genreflex", ["dir/a.h"]) == ["dictionary", "--name", "a_rflx", "--no-rootmap",
                                              "dir/a.h"]
    assert words("genreflex", ["--verbose"])[2] == "dict_rflx"


def test_rootcling_takes_its_output_first_and_its_linkdef_last():
    words = dictgen.dictionary_words
    assert words("rootcling", ["out.cxx", "x.h", "LinkDef.h"]) == [
        "dictionary", "--name", "out", "--linkdef", "LinkDef.h", "--no-rootmap", "x.h"]
    assert words("rootcint", ["-f", "o.cxx", "-rml", "libL.so", "-rmf", "L.rootmap", "x.h"]) == [
        "dictionary", "--name", "L", "x.h"]
    assert words("rootcling", [])[2] == "dict"


def test_a_linkdef_alone_is_the_header_its_dictionary_reads():
    entry = Dictionary("D", "d", [], linkdef="d/LinkDef.h")
    assert builds.dictionary(entry, "D")[-1] == "<build>/d/LinkDef.h"
    assert dictgen.dictionary_words("rootcling", ["-f", "D.cxx", "LinkDef.h"])[-1] == "LinkDef.h"


def test_roottests_ctest_module_defines_its_workarounds_and_python_flags(tmp_path):
    write(tmp_path, {
        "roottest/cmake/modules/RoottestCTest.cmake": (
            "include(CTest)\nadd_definitions(-DWorkAround -DONE=1 -D_WIN32 bad)\n"
            "set(WorkAround TRUE)\nset(PYROOT_EXTRAFLAGS --fixcling)\n"),
        "roottest/CMakeLists.txt": "include(RoottestCTest)\ninclude(NotRoottests)\n"
                                   "ROOTTEST_ADD_TESTDIRS()\n",
        "roottest/a/CMakeLists.txt": "ROOTTEST_ADD_TEST(m MACRO m.C)\n"
                                     "ROOTTEST_ADD_TEST(p MACRO p.py OPTS 1)\n"
                                     "if(NOT WorkAround)\n"
                                     "  ROOTTEST_ADD_TEST(gone MACRO g.C)\nendif()\n"})
    found = by_id(tmp_path)
    args = found["roottest/a/m"].args
    assert args[:4] == ("-e", "#define WorkAround", "-e", "#define ONE=1")
    assert args[4:6] == ("-e", 'gSystem->SetBuildDir("<build>/roottest/a",true)')
    assert found["roottest/a/p"].args == ("--fixcling", "1")
    assert "roottest/a/gone" not in found


def test_foreach_range_counts_as_cmake_counts(tmp_path):
    body = "  ROOTTEST_ADD_TEST({0}-${{i}} MACRO {0}.C)\nendforeach()\n"
    loops = [("RANGE 2", "z"), ("RANGE 3 7 2", "o"), ("RANGE 1 2 0", "s"), ("RANGE x", "n"),
             ("IN ITEMS p q", "i")]
    write(tmp_path, {"roottest/CMakeLists.txt": "ROOTTEST_ADD_TESTDIRS()\n",
                     "roottest/a/CMakeLists.txt": "".join(
                         f"foreach(i {how})\n" + body.format(name) for how, name in loops)})
    assert [test.id.split("/")[-1] for test in discover(tmp_path)] == [
        "z-0", "z-1", "z-2", "o-3", "o-5", "o-7", "s-1", "s-2", "i-p", "i-q"]
