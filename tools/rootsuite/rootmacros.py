"""ROOT's own test macros - ``ROOT_ADD_GTEST``, ``ROOT_ADD_PYUNITTEST``, ``ROOT_ADD_TEST`` -
as what each registers, rather than as the CMake in ``RootMacros.cmake``.

The real macros build executables and wrap every command in
``RootTestDriver.cmake``; none of that applies to xrdroot. What does is the
ctest name each makes (``gtest-hist-hist-testTH1``), the sources the program
is built from, the arguments it is run with and how it is judged - so each
handler here computes exactly those, the way the macro of the same name does
(the line numbers are ROOT 6.40's ``cmake/modules/RootMacros.cmake``).
"""

from __future__ import annotations

import re
from pathlib import Path

from .cmake import HANDLERS, Directory, parse_arguments
from .model import Target, Test, kind_of, make_id

__all__ = ["path_name", "sources", "in_test_tree", "register_command", "ROOT_ADD_TEST_KEYWORDS"]

#: ``ROOT_ADD_TEST``'s keywords (RootMacros.cmake:1642): options, one-value, many-value.
ROOT_ADD_TEST_KEYWORDS = (
    ["DEBUG", "WILLFAIL", "CHECKOUT", "CHECKERR", "RUN_SERIAL"],
    ["TIMEOUT", "BUILD", "INPUT", "OUTPUT", "ERROR", "SOURCE_DIR", "BINARY_DIR", "WORKING_DIR",
     "PROJECT", "PASSRC", "RESOURCE_LOCK"],
    ["COMMAND", "COPY_TO_BUILDDIR", "DIFFCMD", "OUTCNV", "OUTCNVCMD", "PRECMD", "POSTCMD",
     "ENVIRONMENT", "DEPENDS", "PASSREGEX", "OUTREF", "ERRREF", "FAILREGEX", "LABELS",
     "PYTHON_DEPS", "FIXTURES_SETUP", "FIXTURES_CLEANUP", "FIXTURES_REQUIRED", "PROPERTIES"],
)  # fmt: skip

#: ``ROOT_ADD_GTEST``'s keywords (RootMacros.cmake:1943).
_GTEST_KEYWORDS = (
    ["WILLFAIL"],
    ["TIMEOUT", "REPEATS", "FAILREGEX"],
    ["COPY_TO_BUILDDIR", "LIBRARIES", "LABELS", "FIXTURES_SETUP", "FIXTURES_CLEANUP",
     "FIXTURES_REQUIRED", "INCLUDE_DIRS", "ENVIRONMENT"],
)  # fmt: skip

#: ``ROOT_ADD_PYUNITTEST``'s keywords (RootMacros.cmake:2047).
_PYUNITTEST_KEYWORDS = (
    ["WILLFAIL", "GENERIC"],
    [],
    ["COPY_TO_BUILDDIR", "ENVIRONMENT", "PYTHON_DEPS", "FIXTURES_SETUP", "FIXTURES_CLEANUP",
     "FIXTURES_REQUIRED", "PRECMD", "POSTCMD"],
)  # fmt: skip


def path_name(self: Directory, name: str) -> str:
    """``ROOT_PATH_TO_STRING(... PATH_SEPARATOR_REPLACEMENT "-")`` then ``-test-`` dropped.

    ``testTH1`` in ``hist/hist/test`` is ``-hist-hist-testTH1``: the directory
    and name made a C identifier, its underscores dashes, ``-test-`` removed.
    """
    mangled = re.sub(r"[^A-Za-z0-9_]", "_", f"/{self.relative}/{name}").replace("_", "-")
    return mangled.replace("-test-", "-")


def in_test_tree(relative: str) -> bool:
    """Whether a directory holds tests - where a library built is the test's, not ROOT's."""
    parts = relative.split("/")
    return "test" in parts or parts[0] == "roottest"


def sources(self: Directory, given: list[str]) -> list[str]:
    """``ROOT_GET_SOURCES``: globs expanded, each made relative to ROOT's source tree.

    A source that does not exist - a ``G__`` dictionary rootcling writes - is
    kept by name, as the macro keeps it.
    """
    found: list[str] = []
    for entry in given:
        matches = sorted(self.source.glob(entry)) if any(c in entry for c in "*?[") else []
        paths = [str(match) for match in matches] or [entry]
        found.extend(self.relative_path(path) for path in paths)
    return found


def _generated(files: list[str]) -> bool:
    return any(Path(name).name.startswith("G__") for name in files)


def _target(self: Directory, name: str, files: list[str], libraries: list[str]) -> Target:
    self.targets.add(name)
    return Target(files, list(libraries), _generated(files))


# --- targets ---------------------------------------------------------------


def _executable(self: Directory, args: list[str]) -> None:
    """``ROOT_EXECUTABLE(name sources... LIBRARIES ...)``: a program a test may run."""
    if not args:
        return
    parsed = parse_arguments(args[1:], ["CMAKENOEXPORT", "NOINSTALL", "TEST"], [],
                             ["LIBRARIES", "BUILTINS", "ADDITIONAL_COMPILE_FLAGS"])
    _program(self, args[0], sources(self, parsed["UNPARSED"]), parsed.get("LIBRARIES", []))


def _program(self: Directory, name: str, files: list[str], libraries: list[str]) -> None:
    """A program built in a test directory; ROOT's own (``root.exe``, ``hadd``) are xrdroot's."""
    target = _target(self, name, files, libraries)
    if in_test_tree(self.relative):
        self.registry.programs[name] = target


def _add_executable(self: Directory, args: list[str]) -> None:
    files = [arg for arg in args[1:] if arg not in ("EXCLUDE_FROM_ALL", "WIN32", "MACOSX_BUNDLE")]
    if args:
        _program(self, args[0], sources(self, files), [])


def _library(self: Directory, name: str, files: list[str], libraries: list[str]) -> None:
    target = _target(self, name, files, libraries)
    if in_test_tree(self.relative):
        self.registry.libraries[name] = target


def _package(self: Directory, args: list[str]) -> None:
    """``ROOT_STANDARD_LIBRARY_PACKAGE``: always a dictionary, from its headers and LinkDef."""
    if not args:
        return
    parsed = parse_arguments(
        args[1:], ["NO_INSTALL_HEADERS", "NO_SOURCES", "OBJECT_LIBRARY", "NO_CXXMODULE"],
        [], ["SOURCES", "HEADERS", "LINKDEF", "DEPENDENCIES", "LIBRARIES", "DICTIONARY_OPTIONS"])
    _library(self, args[0], sources(self, parsed.get("SOURCES", [])), parsed.get("LIBRARIES", []))
    if args[0] in self.registry.libraries:
        self.registry.libraries[args[0]].dictionary = True


def _linker_library(self: Directory, args: list[str]) -> None:
    if not args:
        return
    parsed = parse_arguments(args[1:], ["TYPE", "SHARED", "STATIC"], [],
                             ["LIBRARIES", "DEPENDENCIES", "BUILTINS"])
    _library(self, args[0], sources(self, parsed["UNPARSED"]), parsed.get("LIBRARIES", []))


def _add_library(self: Directory, args: list[str]) -> None:
    kinds = {"STATIC", "SHARED", "MODULE", "OBJECT", "INTERFACE", "EXCLUDE_FROM_ALL", "IMPORTED"}
    if args:
        _library(self, args[0], sources(self, [a for a in args[1:] if a not in kinds]), [])


def _link(self: Directory, args: list[str]) -> None:
    """``target_link_libraries``: a test program picking up a library its directory built."""
    target = self.registry.programs.get(args[0]) if args else None
    if target is not None:
        scopes = {"PRIVATE", "PUBLIC", "INTERFACE"}
        target.libraries.extend(arg for arg in args[1:] if arg not in scopes)


def _test_subdirectory(self: Directory, args: list[str]) -> None:
    for subdir in args[:1]:
        self.registry.test_subdirs.append(self.relative_path(subdir))


# --- tests -----------------------------------------------------------------


def _gtest(self: Directory, args: list[str]) -> None:
    """``ROOT_ADD_GTEST(suite sources... [LIBRARIES ...] ...)``: one gtest program."""
    if not args:
        return
    parsed = parse_arguments(args[1:], *_GTEST_KEYWORDS)
    suite = args[0]
    self.registry.programs[suite] = _target(
        self, suite, sources(self, parsed["UNPARSED"]), parsed.get("LIBRARIES", []))
    ctest = "gtest" + path_name(self, suite)
    extra = ["--gtest_repeat=" + parsed["REPEATS"][0]] if parsed.get("REPEATS") else []
    self.registry.add(Test(
        id=make_id("gtest", self.relative, suite), ctest=ctest,
        kind=kind_of(ctest, self.relative), directory=self.relative, runner="gtest",
        command=(suite, *extra), args=tuple(extra), program=suite,
        will_fail="WILLFAIL" in parsed, failregex=tuple(parsed.get("FAILREGEX", [])),
        check_out=False, check_err=False, **_common(parsed)))


def _common(parsed: dict[str, list[str]]) -> dict[str, object]:
    """The keywords every ROOT test macro passes through to ``ROOT_ADD_TEST`` alike."""
    timeout = parsed.get("TIMEOUT") or [""]
    return {
        "labels": tuple(parsed.get("LABELS", [])),
        "environment": tuple(parsed.get("ENVIRONMENT", [])),
        "fixtures_setup": tuple(parsed.get("FIXTURES_SETUP", [])),
        "fixtures_required": tuple(parsed.get("FIXTURES_REQUIRED", [])),
        "timeout": float(timeout[0]) if timeout[0].replace(".", "", 1).isdigit() else None,
    }


def _unittest_dir(self: Directory, args: list[str]) -> None:
    """``ROOT_ADD_UNITTEST_DIR(libraries...)``: every ``.cxx`` beside it as one gtest."""
    files = sorted(path.name for path in self.source.glob("*.cxx"))
    suite = re.sub(r"[^A-Za-z0-9]", "", f"/{self.relative}/") + "Unit"
    _gtest(self, [suite, *files, "LIBRARIES", *args])


def _pyunittest(self: Directory, args: list[str]) -> None:
    """``ROOT_ADD_PYUNITTEST(name file.py ...)``: a unittest file, or a script if ``GENERIC``."""
    if len(args) < 2:
        return
    parsed = parse_arguments(args[2:], *_PYUNITTEST_KEYWORDS)
    ctest = "pyunittests" + path_name(self, args[0].replace("_", "-"))
    pre = parsed.get("PRECMD", []) + parsed.get("POSTCMD", [])
    self.registry.add(Test(
        id=make_id("pyunittest", self.relative, args[0]), ctest=ctest,
        kind=kind_of(ctest, self.relative), directory=self.relative,
        runner="script" if "GENERIC" in parsed else "pyunittest",
        command=("python", args[1]), files=(self.relative_path(args[1]),),
        will_fail="WILLFAIL" in parsed, python_deps=tuple(parsed.get("PYTHON_DEPS", [])),
        check_out=False, check_err=False,
        skip=f"runs {' '.join(pre)} around the test, which the harness does not" if pre else "",
        **_common(parsed)))


def _pyunittests(self: Directory, args: list[str]) -> None:
    """``ROOT_ADD_PYUNITTESTS(name)``: every ``.py`` beside it, discovered by unittest."""
    if not args:
        return
    files = tuple(self.relative_path(p.name) for p in sorted(self.source.glob("*.py")))
    ctest = "pyunittests-" + args[0].replace("_", "-")
    self.registry.add(Test(
        id=make_id("pyunittest", self.relative, args[0]), ctest=ctest, kind="pyunittest",
        directory=self.relative, runner="pyunittest", files=files,
        check_out=False, check_err=False))


def _add_test(self: Directory, args: list[str]) -> None:
    """``ROOT_ADD_TEST(name COMMAND ...)``: a command, judged by its exit and its regexes."""
    if not args:
        return
    parsed = parse_arguments(args[1:], *ROOT_ADD_TEST_KEYWORDS)
    name = args[0]
    kind = kind_of(name, self.relative)
    short = name[len(kind) + 1:] if name.startswith(kind + "-") else name
    register_command(self, parsed, make_id(kind, self.relative, short), name)


def register_command(self: Directory, parsed: dict[str, list[str]], test_id: str,
                     ctest: str) -> None:
    """A test whose ``COMMAND`` is a program built here, ``root``, Python, or anything else."""
    command = parsed.get("COMMAND", [])
    passrc = parsed.get("PASSRC") or [""]
    self.registry.add(Test(
        id=test_id, ctest=ctest, kind=kind_of(ctest, self.relative), directory=self.relative,
        command=tuple(command), passrc=int(passrc[0]) if passrc[0].lstrip("-").isdigit() else None,
        passregex=tuple(parsed.get("PASSREGEX", [])), failregex=tuple(parsed.get("FAILREGEX", [])),
        will_fail="WILLFAIL" in parsed, depends=tuple(parsed.get("DEPENDS", [])),
        python_deps=tuple(parsed.get("PYTHON_DEPS", [])),
        working_dir=_working_dir(self, parsed), check_out=False, check_err=False,
        **_how(self, command), **_common(parsed)))


def _working_dir(self: Directory, parsed: dict[str, list[str]]) -> str:
    given = (parsed.get("WORKING_DIR") or [""])[0]
    return self.relative_path(given) if given and not given.startswith("<build>") else ""


#: The names ``root`` goes by on a command line, after ``$<TARGET_FILE:...>`` is undone.
ROOT_NAMES = ("root", "root.exe")


def _how(self: Directory, command: list[str]) -> dict[str, object]:
    """Which runner a command needs: a program built here, ``root``, a Python script, or other."""
    if not command:
        return {"runner": "command", "skip": "the test has no command"}
    first = Path(command[0]).name
    if command[0] in self.registry.programs or first in self.registry.programs:
        return {"runner": "program", "program": first, "args": tuple(command[1:])}
    if first in ROOT_NAMES or first.startswith("root${"):
        return {"runner": "root", "args": tuple(command[1:])}
    if "python" in first.lower() and len(command) > 1 and command[1].endswith(".py"):
        return {"runner": "script", "files": (self.relative_path(command[1]),),
                "args": tuple(command[2:])}
    return {"runner": "command", "args": tuple(command[1:])}


HANDLERS.update(
    {
        "root_executable": _executable,
        "add_executable": _add_executable,
        "root_standard_library_package": _package,
        "root_linker_library": _linker_library,
        "add_library": _add_library,
        "target_link_libraries": _link,
        "root_add_test_subdirectory": _test_subdirectory,
        "root_add_gtest": _gtest,
        "root_add_unittest_dir": _unittest_dir,
        "root_add_pyunittest": _pyunittest,
        "root_add_pyunittests": _pyunittests,
        "root_add_test": _add_test,
    }
)
