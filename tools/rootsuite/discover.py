"""Discovery: ROOT's CMake run as its configure step would, to list every test it registers.

The walk is the build's own. ROOT's top-level ``CMakeLists.txt`` adds each
module, and each module names its test directory with
``ROOT_ADD_TEST_SUBDIRECTORY``; ``RootCTest.cmake`` then adds those, sorted,
from the top-level scope, and finally ``roottest`` (whose own
``ROOTTEST_ADD_TESTDIRS`` recurse). Build options come from
:data:`DEFAULT_FEATURES` - a CI build's - or ``--features``; directories a
sparse checkout lacks (``interpreter/``, module sources) are skipped.

Afterwards each test is completed from what was registered anywhere: the
sources of the program it runs and of the test-local libraries that links,
and whether it - or a setup step it depends on - needs a dictionary
generated or an executable compiled.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from . import roottestmacros  # noqa: F401  (registers roottest's macros)
from .cmake import Directory
from .model import Registry, Target, Test

__all__ = ["DEFAULT_FEATURES", "variables", "discover", "complete", "fixture_providers"]

#: The build options of a ROOT 6.40 CI build on Linux: what ``if(<option>)`` finds ON.
DEFAULT_FEATURES = frozenset(
    """asimage builtin_clang builtin_cling builtin_llvm builtin_nlohmannjson builtin_openui5
    builtin_vdt clad curl dataframe davix fftw3 fitsio gdml geom geombuilder gviz http imt mathmore
    minuit2 opengl pyroot roofit root7 roottest rpath runtime_cxxmodules shared spectrum sqlite
    ssl testing tmva tmva-cpu tmva-pymva tpython unuran uring vdt webgui x11 xml xrootd""".split()
)


def variables(root: Path, features: Iterable[str]) -> dict[str, str]:
    """What ROOT's configure step has set by the time tests are added, on a 64-bit Linux."""
    enabled = sorted(features)
    found = dict.fromkeys(enabled, "ON")
    found.update(
        CMAKE_SOURCE_DIR=str(root), ROOT_SOURCE_DIR=str(root), CMAKE_BINARY_DIR="<build>",
        ROOT_BINARY_DIR="<build>", ROOTSYS="<build>", ROOT_BINDIR="<build>/bin",
        CMAKE_SYSTEM_NAME="Linux", UNIX="1", CMAKE_SIZEOF_VOID_P="8", **{"64BIT": "1"},
        CMAKE_PROJECT_NAME="ROOT", PROJECT_NAME="ROOT", CMAKE_COMMAND="cmake",
        CMAKE_CXX_COMPILER_ID="GNU", CMAKE_BUILD_TYPE="Release", CMAKE_CXX_STANDARD="20",
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
    for subdir in sorted(set(registry.test_subdirs)):
        if subdir != "tutorials":
            top.subdirectory(subdir)
    top.subdirectory("roottest")
    return complete(registry)


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
    """Each test with its program's sources filled in and what it needs said."""
    by_name = {test.ctest: test for test in registry.tests}
    providers = fixture_providers(registry.tests)
    done: list[Test] = []
    seen: set[str] = set()
    for test in registry.tests:
        test = _with_program(registry, test)
        upstream = [by_name[d] for d in test.depends if d in by_name]
        upstream += [p for f in test.fixtures_required for p in providers.get(f, [])]
        needs = {p.setup_of for p in upstream if p.setup_of in ("dictionary", "executable")}
        test = replace(test, needs=tuple(sorted({*test.needs, *needs})), id=_unique(test.id, seen))
        done.append(test)
    return done


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
    """The sources of the program a test runs, and of the test-local libraries it links."""
    if not test.program:
        return test
    target = registry.programs.get(test.program)
    if target is None:
        return replace(test, skip=test.skip or f"runs {test.program}, which no CMake here builds")
    libraries = [registry.libraries[name] for name in _closure(registry, target)]
    files = [*target.files, *(f for library in libraries for f in library.files)]
    return replace(test, files=tuple(dict.fromkeys(files)),
                   needs=tuple(sorted({*test.needs, *_needs(target, libraries)})))


def _needs(target: Target, libraries: list[Target]) -> list[str]:
    """A dictionary when the program or a library it links has one; a compiler for roottest's."""
    needs = ["dictionary"] if any(t.dictionary for t in [target, *libraries]) else []
    return [*needs, "executable"] if target.compiled else needs


def _closure(registry: Registry, target: Target) -> list[str]:
    """The test-local libraries a target links, and those they link, each once."""
    found: list[str] = []
    pending = list(target.libraries)
    while pending:
        name = pending.pop(0)
        if name in registry.libraries and name not in found:
            found.append(name)
            pending.extend(registry.libraries[name].libraries)
    return found
