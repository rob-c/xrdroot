"""Discovery: ROOT's CMake run as its configure step would, to list every test it registers.

The walk is the build's own. ROOT's top-level ``CMakeLists.txt`` adds each
module, and each module names its test directory with
``ROOT_ADD_TEST_SUBDIRECTORY``; ``RootCTest.cmake`` then adds those, sorted,
from the top-level scope, and finally ``roottest`` (whose own
``ROOTTEST_ADD_TESTDIRS`` recurse). The machine is a Mac on ARM64
(:mod:`~tools.rootsuite.machine`): the build options are the oracle's, else
:data:`~tools.rootsuite.machine.DEFAULT_FEATURES`, or ``--features``;
directories a sparse checkout lacks (``interpreter/``, module sources) are
skipped.

Afterwards each test is completed from what was registered anywhere: the
sources of the program it runs and of the test-local libraries that links,
and whether it - or a setup step it depends on - needs a dictionary
generated or an executable compiled.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from . import builds, roottestmacros  # noqa: F401  (roottestmacros registers roottest's macros)
from .cmake import HANDLERS, Directory
from .machine import DEFAULT_FEATURES, PLATFORM
from .model import Registry, Target, Test

__all__ = ["DEFAULT_FEATURES", "variables", "discover", "complete", "fixture_providers"]


def variables(root: Path, features: Iterable[str]) -> dict[str, str]:
    """What ROOT's configure step has set by the time tests are added, on a Mac with ARM64."""
    enabled = sorted(features)
    found = dict.fromkeys(enabled, "ON")
    found.update(PLATFORM)
    found.update(
        CMAKE_SOURCE_DIR=str(root), ROOT_SOURCE_DIR=str(root), CMAKE_BINARY_DIR="<build>",
        ROOT_BINARY_DIR="<build>", ROOTSYS="<build>", ROOT_BINDIR="<build>/bin",
        ROOT_TUTORIALS_DIR="<build>/tutorials", ROOT_INCLUDE_DIRS="<build>/include",
        ROOT_LIBRARY_DIR="<build>/lib",
        CMAKE_SIZEOF_VOID_P="8", **{"64BIT": "1"},
        CMAKE_PROJECT_NAME="ROOT", PROJECT_NAME="ROOT", CMAKE_COMMAND="cmake",
        CMAKE_BUILD_TYPE="Release", CMAKE_CXX_STANDARD="20",
        Python3_EXECUTABLE="python3", Python3_VERSION="3.12", Python3_VERSION_MAJOR="3",
        Python3_VERSION_MINOR="12", ROOT_root_CMD="root.exe", ROOT_hadd_CMD="hadd",
        ROOT_genreflex_CMD="genreflex", ROOT_rootcling_CMD="rootcling",
        ROOT_rootcint_CMD="rootcint", ROOT_CONFIG_EXECUTABLE="root-config",
        TOOLS_PREFIX="<build>/bin", PY_TOOLS_PREFIX="<build>/bin", ref_suffix=".ref",
        _root_enabled_options=";".join(enabled), _root_all_options=";".join(enabled),
        ROOT_TEST_DRIVER=str(root / "cmake/modules/RootTestDriver.cmake"),
    )
    return found


def discover(root: Path, features: Iterable[str] = DEFAULT_FEATURES) -> list[Test]:
    """Every test a CI configure of ``root`` registers, completed, in registration order."""
    root = Path(root).resolve()
    chosen = frozenset(features)
    registry = Registry(str(root), chosen, targets=_built_targets(chosen))
    top = Directory(registry, "", variables(root, chosen))
    if (root / "CMakeLists.txt").is_file():
        top.run_file(root / "CMakeLists.txt")
    if not registry.test_subdirs:  # no module CMakeLists to name them: take every */test
        registry.test_subdirs = _test_directories(root)
    _test_subdirectories(top)
    top.subdirectory("roottest")
    return complete(registry)


def _test_subdirectories(top: Directory) -> None:
    """``RootCTest.cmake``'s walk: each module's test directory, sorted, from the top scope."""
    for subdir in sorted(set(top.registry.test_subdirs)):
        if subdir != "tutorials":
            top.subdirectory(subdir)


#: roottest's own CMake modules that set what its tests are run with: ``PYROOT_EXTRAFLAGS``
#: and the ``ClingWorkAround`` definitions every macro is run with.
_ROOTTEST_MODULES = ("RoottestCTest",)


def _include(self: Directory, args: list[str]) -> None:
    """``include(RootCTest)`` adds the module test directories there - before roottest;
    ``include(RoottestCTest)`` is read where it is included."""
    if args[:1] == ["RootCTest"]:
        _test_subdirectories(self)
    elif args[:1] and args[0] in _ROOTTEST_MODULES:
        modules = Path(self.registry.root) / "roottest/cmake/modules"
        for module in modules.glob(f"{args[0]}.cmake"):
            self.run_file(module)


HANDLERS["include"] = _include


def _built_targets(features: frozenset[str]) -> set[str]:
    """Targets ``if(TARGET ...)`` finds whose CMakeLists a sparse checkout may lack.

    ROOT's test support library is there in any testing build; the GUI
    libraries (``Gui``, ``TreeViewer``) in any build with graphics.
    """
    found = {"ROOT::TestSupport", "Core"}
    if features & {"x11", "cocoa", "opengl"}:
        found |= {"Gui", "TreeViewer", "Ged", "RGL"}
    return found


def _test_directories(root: Path) -> list[str]:
    """Every ``test`` directory with a CMakeLists, outside roottest, tutorials and interpreter."""
    found = []
    for listing in root.glob("**/test/CMakeLists.txt"):
        relative = listing.parent.relative_to(root).as_posix()
        if relative.split("/")[0] not in ("roottest", "tutorials", "interpreter"):
            found.append(relative)
    return sorted(found)


def complete(registry: Registry) -> list[Test]:
    """Each test with its program's sources filled in and what it needs said.

    A dictionary generated after the target listing its source is only known
    now, so each target's need of one is settled first.
    """
    for target in (*registry.programs.values(), *registry.libraries.values()):
        target.dictionary = target.dictionary or bool(builds.split(registry, target)[0])
    by_name = {test.ctest: test for test in registry.tests}
    providers = fixture_providers(registry.tests)
    seen: set[str] = set()
    done = []
    for test in registry.tests:
        test = _with_builds(registry, _with_program(registry, test))
        needs = _upstream_needs(test, by_name, providers)
        done.append(replace(test, needs=tuple(sorted({*test.needs, *needs})),
                            id=_unique(test.id, seen)))
    return done


def _upstream_needs(test: Test, by_name: dict[str, Test],
                    providers: dict[str, list[Test]]) -> set[str]:
    """A dictionary or an executable a setup step it depends on builds."""
    upstream = [by_name[d] for d in test.depends if d in by_name]
    upstream += [p for f in test.fixtures_required for p in providers.get(f, [])]
    return {p.setup_of for p in upstream if p.setup_of in ("dictionary", "executable")}


def fixture_providers(tests: Iterable[Test]) -> dict[str, list[Test]]:
    """Each fixture name and the tests that set it up."""
    providers: dict[str, list[Test]] = {}
    for test in tests:
        for fixture in test.fixtures_setup:
            providers.setdefault(fixture, []).append(test)
    return providers


def _unique(test_id: str, seen: set[str]) -> str:
    """Ids are unique: a name registered twice in one directory gets ``#2``, ``#3``..."""
    candidate, count = test_id, 1
    while candidate in seen:
        count += 1
        candidate = f"{test_id}#{count}"
    seen.add(candidate)
    return candidate


def _with_program(registry: Registry, test: Test) -> Test:
    """The program a test runs: how it is built, or its sources for ``xrdroot run``.

    A program that needs a dictionary or links a library of the tests' own is
    built - its libraries, its dictionaries, then it - and run as the
    executable that writes; roottest's compiled executables are built by
    setup tests of their own, so they are only run. A command naming no
    program built here runs as written.
    """
    if not test.program or test.runner == "setup":
        return test
    target = registry.programs.get(test.program)
    if target is None:
        return replace(test, runner="command", program="")
    test = _with_sources(registry, test, target)
    launch = f"<build>/{target.directory}/{target.output}"
    if target.compiled:
        return replace(test, launch=launch)
    if builds.closure(registry, target) or target.dictionary:
        steps = builds.executable(registry, target, target.output)
        return replace(test, builds=tuple(steps), launch=launch)
    return test


def _with_sources(registry: Registry, test: Test, target: Target) -> Test:
    """A test's program's sources, with its test libraries', and what building them needs."""
    libraries = [registry.libraries[name] for name in builds.closure(registry, target)]
    files = [*target.files, *(f for library in libraries for f in library.files)]
    return replace(test, files=tuple(dict.fromkeys(files)),
                   needs=tuple(sorted({*test.needs, *_needs(target, libraries)})))


def _needs(target: Target, libraries: list[Target]) -> list[str]:
    """A dictionary when the program or a library it links has one; a compiler for roottest's."""
    needs = ["dictionary"] if any(t.dictionary for t in [target, *libraries]) else []
    return [*needs, "executable"] if target.compiled else needs


def _with_builds(registry: Registry, test: Test) -> Test:
    """A setup test's builds, once every target it may build is known.

    A roottest executable links libraries their own setup tests build; a
    ``cmake --build --target`` test builds its targets with what they link.
    """
    if test.runner != "setup" or test.builds:
        return test
    if test.program in registry.programs:
        target = registry.programs[test.program]
        steps = builds.executable(registry, target, target.output, libraries=False)
        return replace(test, builds=tuple(steps))
    steps = [step for name in test.args for step in _target_builds(registry, name)]
    if not steps:
        return replace(test, skip=f"builds {' '.join(test.args) or 'nothing'}, which no CMake "
                                  "here defines as a target the harness can build")
    return replace(test, builds=tuple(dict.fromkeys(steps)))


def _target_builds(registry: Registry, name: str) -> list[builds.Build]:
    if name in registry.libraries:
        return builds.library(registry, registry.libraries[name])
    if name in registry.programs:
        target = registry.programs[name]
        return builds.executable(registry, target, target.output)
    found = [entry for entry in registry.dictionaries.values() if entry.name == name]
    return [builds.dictionary(entry, builds.library_name(name)) for entry in found[:1]]
