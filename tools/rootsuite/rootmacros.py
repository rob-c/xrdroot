"""ROOT's own test macros - ``ROOT_ADD_GTEST``, ``ROOT_ADD_PYUNITTEST``, ``ROOT_ADD_TEST`` -
as what each registers, rather than as the CMake in ``RootMacros.cmake``.

The real macros build executables and wrap every command in
``RootTestDriver.cmake``. What matters here is the ctest name each makes
(``gtest-hist-hist-testTH1``), the targets built - sources, the
dictionaries rootcling generates into them, the libraries they link, which
:mod:`~tools.rootsuite.builds` turns into ``xrdroot build`` commands - the
arguments a test is run with and how it is judged; so each handler here
computes exactly those, the way the macro of the same name does (the line
numbers are ROOT 6.40's ``cmake/modules/RootMacros.cmake``). CMake's own
``add_test``, ``set_property(TEST ...)`` and ``set_tests_properties`` are
here too: a property set after the fact - a fixture, ``DISABLED`` - is part
of the test ctest runs.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .cmake import HANDLERS, Directory, parse_arguments
from .model import Dictionary, Target, Test, kind_of, make_id

__all__ = ["path_name", "sources", "in_test_tree", "register_command", "ROOT_ADD_TEST_KEYWORDS",
           "library", "generate_dictionary", "build_targets", "set_test_property", "working_dir"]

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


def _generated(self: Directory, files: list[str]) -> bool:
    """Whether a target's sources include a dictionary rootcling writes."""
    return any(Path(name).name.startswith("G__") or name in self.registry.dictionaries
               for name in files)


def _target(self: Directory, name: str, files: list[str], libraries: list[str],
            output: str = "") -> Target:
    """A target as built here: ``output`` is its file's name, a library's less ``lib``."""
    self.targets.add(name)
    return Target(files, list(libraries), _generated(self, files), directory=self.relative,
                  output=output or name, includes=self.includes())


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


def generate_dictionary(self: Directory, args: list[str]) -> Dictionary:
    """``ROOT_GENERATE_DICTIONARY(dict headers... [MODULE m] [LINKDEF l] ...)``.

    rootcling writes ``dict.cxx`` beside the build; a target listing that
    source compiles it in, and a ``MODULE`` already a target gets it linked.
    """
    parsed = parse_arguments(args[1:], ["STAGE1", "MULTIDICT", "NOINSTALL", "NO_CXXMODULE"],
                             ["MODULE", "LINKDEF"], _DICTIONARY_LISTS)
    linkdef = (parsed.get("LINKDEF") or [""])[0]
    entry = Dictionary(args[0], self.relative, sources(self, parsed["UNPARSED"]),
                       linkdef=self.relative_path(linkdef) if linkdef else "",
                       includes=self.includes())
    self.registry.dictionaries[self.relative_path(f"{args[0]}.cxx")] = entry
    module = parsed.get("MODULE", [""])[0]
    owner = self.registry.programs.get(module) or self.registry.libraries.get(module)
    if owner is not None and module != args[0]:
        owner.dictionaries.append(entry)
        owner.dictionary = True
    return entry


#: ``ROOT_GENERATE_DICTIONARY``'s many-value keywords.
_DICTIONARY_LISTS = ["NODEPHEADERS", "OPTIONS", "DEPENDENCIES", "EXTRA_DEPENDENCIES", "BUILTINS"]


def _generate_dictionary(self: Directory, args: list[str]) -> None:
    if args:
        generate_dictionary(self, args)


def _add_executable(self: Directory, args: list[str]) -> None:
    files = [arg for arg in args[1:] if arg not in ("EXCLUDE_FROM_ALL", "WIN32", "MACOSX_BUNDLE")]
    if args:
        _program(self, args[0], sources(self, files), [])


def library(self: Directory, name: str, files: list[str], libraries: list[str],
            output: str = "") -> Target:
    """A library built here, kept when it is a test's; ``output`` its file name less ``lib``."""
    target = _target(self, name, files, libraries, output)
    if in_test_tree(self.relative):
        self.registry.libraries[name] = target
    return target


def _package(self: Directory, args: list[str]) -> None:
    """``ROOT_STANDARD_LIBRARY_PACKAGE``: always a dictionary, from its headers and LinkDef."""
    if not args:
        return
    parsed = parse_arguments(
        args[1:], ["NO_INSTALL_HEADERS", "NO_SOURCES", "OBJECT_LIBRARY", "NO_CXXMODULE"],
        [], ["SOURCES", "HEADERS", "LINKDEF", "DEPENDENCIES", "LIBRARIES", "DICTIONARY_OPTIONS"])
    target = library(self, args[0], sources(self, parsed.get("SOURCES", [])),
                     parsed.get("LIBRARIES", []))
    linkdef = parsed.get("LINKDEF", [""])[0]
    target.dictionaries.append(Dictionary(
        f"G__{args[0]}", self.relative, sources(self, parsed.get("HEADERS", [])),
        linkdef=self.relative_path(linkdef) if linkdef else "", includes=self.includes()))
    target.dictionary = True


def _linker_library(self: Directory, args: list[str]) -> None:
    """``ROOT_LINKER_LIBRARY``/``ROOTTEST_LINKER_LIBRARY(name sources... [TEST] ...)``."""
    if not args:
        return
    parsed = parse_arguments(args[1:], ["DLLEXPORT", "CMAKENOEXPORT", "TEST", "NOINSTALL"],
                             ["TYPE"], ["LIBRARIES", "DEPENDENCIES", "BUILTINS"])
    library(self, args[0], sources(self, parsed["UNPARSED"]), parsed.get("LIBRARIES", []))


def _add_library(self: Directory, args: list[str]) -> None:
    kinds = {"STATIC", "SHARED", "MODULE", "OBJECT", "INTERFACE", "EXCLUDE_FROM_ALL", "IMPORTED"}
    if args:
        library(self, args[0], sources(self, [a for a in args[1:] if a not in kinds]), [])


def _built(self: Directory, name: str) -> Target | None:
    return self.registry.programs.get(name) or self.registry.libraries.get(name)


#: The words of ``target_*`` commands that are not the values they add.
_SCOPES = {"PRIVATE", "PUBLIC", "INTERFACE", "SYSTEM", "BEFORE", "AFTER"}


def _link(self: Directory, args: list[str]) -> None:
    """``target_link_libraries``: a test target picking up a library a test directory built."""
    target = _built(self, args[0]) if args else None
    if target is not None:
        target.libraries.extend(arg for arg in args[1:] if arg not in _SCOPES)


def _target_includes(self: Directory, args: list[str]) -> None:
    """``target_include_directories``: where a test target's headers are found."""
    target = _built(self, args[0]) if args else None
    if target is not None:
        found = (self.inside(arg) for arg in args[1:] if arg not in _SCOPES)
        target.includes.extend(path for path in found if path is not None)


def _target_sources(self: Directory, args: list[str]) -> None:
    """``target_sources``: more sources for a test target."""
    target = _built(self, args[0]) if args else None
    if target is not None:
        target.files.extend(sources(self, [arg for arg in args[1:] if arg not in _SCOPES]))


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
    target = self.registry.programs[suite] = _target(
        self, suite, sources(self, parsed["UNPARSED"]), parsed.get("LIBRARIES", []))
    found = (self.inside(entry) for entry in parsed.get("INCLUDE_DIRS", []))
    target.includes.extend(entry for entry in found if entry is not None)
    ctest = "gtest" + path_name(self, suite)
    extra = ["--gtest_repeat=" + parsed["REPEATS"][0], "--gtest_break_on_failure"] \
        if parsed.get("REPEATS") else []
    self.registry.add(Test(
        id=make_id("gtest", self.relative, suite), ctest=ctest,
        kind=kind_of(ctest, self.relative), directory=self.relative, runner="gtest",
        command=(suite, *extra), args=tuple(extra), program=suite,
        will_fail="WILLFAIL" in parsed, failregex=tuple(parsed.get("FAILREGEX", [])),
        check_out=False, check_err=False, **_common(self, parsed)))


def _common(self: Directory, parsed: dict[str, list[str]]) -> dict[str, object]:
    """The keywords every ROOT test macro passes through to ``ROOT_ADD_TEST`` alike."""
    timeout = parsed.get("TIMEOUT") or [""]
    return {
        "labels": tuple(parsed.get("LABELS", [])),
        "environment": tuple(parsed.get("ENVIRONMENT", [])),
        "fixtures_setup": tuple(parsed.get("FIXTURES_SETUP", [])),
        "fixtures_required": tuple(parsed.get("FIXTURES_REQUIRED", [])),
        "timeout": float(timeout[0]) if timeout[0].replace(".", "", 1).isdigit() else None,
        "copy": tuple(self.relative_path(name) for name in parsed.get("COPY_TO_BUILDDIR", [])
                      if name),
        "precmd": tuple(parsed.get("PRECMD", [])),
        "postcmd": tuple(parsed.get("POSTCMD", [])),
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
    self.registry.add(Test(
        id=make_id("pyunittest", self.relative, args[0]), ctest=ctest,
        kind=kind_of(ctest, self.relative), directory=self.relative,
        runner="script" if "GENERIC" in parsed else "pyunittest",
        command=("python", args[1]), files=(self.relative_path(args[1]),),
        will_fail="WILLFAIL" in parsed, python_deps=tuple(parsed.get("PYTHON_DEPS", [])),
        check_out=False, check_err=False, **_common(self, parsed)))


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
        working_dir=working_dir(self, parsed), check_out=False, check_err=False,
        **_how(self, command), **_common(self, parsed)))


def working_dir(self: Directory, parsed: dict[str, list[str]]) -> str:
    """``WORKING_DIR``, below ROOT's tree; the build tree mirrors it, so either names one."""
    given = (parsed.get("WORKING_DIR") or [""])[0]
    found = self.inside(given) if given else None
    return found if found is not None and found != self.relative else ""


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


# --- CMake's own add_test and test properties ---------------------------------


def build_targets(command: list[str] | tuple[str, ...]) -> list[str] | None:
    """The targets a ``cmake --build ... --target T... [-- ...]`` command builds, else ``None``."""
    if list(command[:2]) != ["cmake", "--build"] or "--target" not in command:
        return None
    rest = list(command[command.index("--target") + 1:])
    rest = rest[:rest.index("--")] if "--" in rest else rest
    return [name.removesuffix("/fast") for name in rest if name]


def _cmake_add_test(self: Directory, args: list[str]) -> None:
    """``add_test(NAME n COMMAND ... [WORKING_DIRECTORY d])``: no driver, so only its exit counts.

    A ``cmake --build --target T`` command is a build step: T built by xrdroot.
    """
    if args[:1] != ["NAME"] or len(args) < 2:
        return
    parsed = parse_arguments(args[2:], [], ["WORKING_DIRECTORY"], ["COMMAND", "CONFIGURATIONS"])
    name, command = args[1], parsed.get("COMMAND", [])
    kind = kind_of(name, self.relative)
    short = name[len(kind) + 1:] if name.startswith(kind + "-") else name
    test_id = make_id(kind, self.relative, short.removeprefix(f"{_dashed(self.relative)}-"))
    targets = build_targets(command)
    if targets is not None:
        self.registry.add(Test(id=test_id, ctest=name, kind=kind, directory=self.relative,
                               runner="setup", setup_of="build", args=tuple(targets),
                               command=tuple(command), check_out=False, check_err=False))
        return
    register_command(self, {"COMMAND": command, "WORKING_DIR": parsed.get("WORKING_DIRECTORY", [])},
                     test_id, name)


def _dashed(relative: str) -> str:
    """A directory as roottest spells it in test names: ``root/io`` as ``root-io``."""
    return relative.removeprefix("roottest/").replace("/", "-")


#: Test properties kept as a list, by the :class:`Test` field holding them.
_LIST_PROPERTIES = {
    "FIXTURES_SETUP": "fixtures_setup", "FIXTURES_REQUIRED": "fixtures_required",
    "ENVIRONMENT": "environment", "DEPENDS": "depends", "LABELS": "labels",
    "PASS_REGULAR_EXPRESSION": "passregex", "FAIL_REGULAR_EXPRESSION": "failregex",
}  # fmt: skip


def _flag(values: list[str]) -> bool:
    return bool(values) and values[0].upper() in ("1", "ON", "TRUE", "YES")


def _timeout(test: Test, values: list[str]) -> Test:
    number = values[0] if values else ""
    return replace(test, timeout=float(number)) if number.replace(".", "", 1).isdigit() else test


#: Test properties of one value, and how each changes a test; ``None`` drops it.
_SCALAR_PROPERTIES: dict[str, Callable[[Test, list[str]], Test | None]] = {
    "DISABLED": lambda test, values: None if _flag(values) else test,
    "WILL_FAIL": lambda test, values: replace(test, will_fail=_flag(values)),
    "TIMEOUT": _timeout,
}


def _property(test: Test, name: str, values: list[str], append: bool) -> Test | None:
    """``test`` with one property set (or appended to); ``None`` when it is DISABLED."""
    values = [part for value in values for part in value.split(";") if part]
    if name in _LIST_PROPERTIES:
        key = _LIST_PROPERTIES[name]
        before: tuple[str, ...] = getattr(test, key) if append else ()
        return replace(test, **{key: (*before, *values)})
    change = _SCALAR_PROPERTIES.get(name)
    return change(test, values) if change is not None else test


def set_test_property(self: Directory, names: list[str], name: str, values: list[str],
                      append: bool = False) -> None:
    """A property of registered tests, as ctest will see it; a DISABLED test is dropped."""
    kept: list[Test] = []
    for test in self.registry.tests:
        changed = _property(test, name, values, append) if test.ctest in names else test
        if changed is not None:
            kept.append(changed)
    self.registry.tests[:] = kept


def _set_property(self: Directory, args: list[str]) -> None:
    """``set_property(TEST names... [APPEND] PROPERTY name values...)``."""
    if args[:1] != ["TEST"] or "PROPERTY" not in args:
        return
    at = args.index("PROPERTY")
    names = [arg for arg in args[1:at] if arg not in ("APPEND", "APPEND_STRING")]
    if at + 1 < len(args):
        set_test_property(self, names, args[at + 1], args[at + 2:], "APPEND" in args[1:at])


def _set_tests_properties(self: Directory, args: list[str]) -> None:
    """``set_tests_properties(names... PROPERTIES name value [name value]...)``."""
    if "PROPERTIES" not in args:
        return
    at = args.index("PROPERTIES")
    pairs = args[at + 1:]
    for name, value in zip(pairs[::2], pairs[1::2], strict=False):
        set_test_property(self, args[:at], name, [value])


HANDLERS.update(
    {
        "add_test": _cmake_add_test,
        "set_property": _set_property,
        "set_tests_properties": _set_tests_properties,
        "root_generate_dictionary": _generate_dictionary,
        "target_include_directories": _target_includes,
        "target_sources": _target_sources,
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
