"""Which tutorials there are, and what ROOT's own build says about each.

Every ``.C``, ``.cxx``, ``.cpp`` and ``.py`` file under the tutorials
directory is a candidate. ``tutorials/CMakeLists.txt``, run by
:mod:`tools.tutorials.cmake` with the installed ROOT's build options, turns
the ones ROOT's CI runs into tests - with the exit code that counts as a
pass, the output that counts as a failure, labels (``needs_network``,
``multithreaded``, ``longtest``), the Python packages each needs, and the
tests that must run first because they write its input (``hsimple.root``
for nearly every C++ tutorial). The rest are vetoed, and the name of the
``*_veto`` list that dropped each is kept as the reason it is not run.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .cmake import Interpreter, TestSpec

__all__ = ["LANGUAGES", "Tutorial", "discover", "catalog", "veto_reason", "closure", "levels"]

#: The file extensions a tutorial is written in, and which interpreter runs it.
LANGUAGES = {".C": "cxx", ".cxx": "cxx", ".cpp": "cxx", ".py": "py"}

#: Why a ``*_veto`` list keeps a tutorial out of ROOT's CI, by the list's name.
VETO_REASONS = {
    "gui_veto": "needs a display (GUI)",
    "extra_veto": "not a standalone tutorial (a helper, or disabled upstream)",
    "pyveto": "not a standalone tutorial, or needs a GUI (ROOT's Python veto)",
    "xrootd_veto": "needs xrootd and the network",
    "davix_veto": "needs davix and the network",
    "sycl_veto": "needs SYCL (a GPU)",
    "imt_veto": "needs implicit multithreading",
    "tbb_veto": "needs TBB",
    "mpi_veto": "needs MPI",
    "root7_veto": "needs ROOT 7 components (root7)",
    "tmva_veto": "needs TMVA or the Python ML packages it drives",
    "dataframe_veto": "needs RDataFrame or one of its data sources",
    "classic_veto": "not run in a classic build",
    "histfactory_veto": "not a standalone tutorial (HistFactory helper)",
    "gviz_veto": "needs graphviz",
    "vecgeom_veto": "needs VecGeom",
    "pythia_veto": "needs Pythia8",
}

#: Why a file with a tutorial's extension is not run when no veto names it.
NOT_A_TEST = "not a test in ROOT's CMakeLists (helper, or needs a build option)"

#: A test's command names the tutorial by its path, then perhaps ACLiC's ``+``.
_SOURCE = re.compile(r"(?P<path>.*\.(?:C|cxx|cpp|py))(?P<suffix>[^/]*)$")


@dataclass(frozen=True)
class Tutorial:
    """One tutorial file, and how ROOT's build runs and judges it."""

    path: str
    test: str | None = None
    flags: tuple[str, ...] = ("-b", "-q")
    suffix: str = ""
    passrc: int = 0
    failregex: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    depends: tuple[str, ...] = ()
    python_deps: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    vetoed: str | None = None

    @property
    def language(self) -> str:
        return LANGUAGES[PurePosixPath(self.path).suffix]

    @property
    def area(self) -> str:
        """The directory it is in: ``hist``, ``analysis/dataframe``, ``(top)``."""
        parent = PurePosixPath(self.path).parent.as_posix()
        return "(top)" if parent == "." else parent


def discover(directory: Path) -> list[str]:
    """Every tutorial-language file under the directory, relative, in order."""
    root = Path(directory)
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.suffix in LANGUAGES and path.is_file() and not _hidden(path.relative_to(root))
    )


def _hidden(relative: Path) -> bool:
    return any(part.startswith(".") for part in relative.parts)


def _located(test: TestSpec, source: str) -> re.Match[str] | None:
    """The tutorial a test runs, found among its command's arguments."""
    for arg in test.command:
        if arg.startswith(source):
            return _SOURCE.match(arg[len(source) :])
    return None


def _flags(test: TestSpec) -> tuple[str, ...]:
    """The interpreter's options: every dashed argument, since no program name has a dash."""
    return tuple(arg for arg in test.command if arg.startswith("-"))


def _tested(tests: Iterable[TestSpec], source: str) -> dict[str, tuple[TestSpec, str]]:
    """Each tested tutorial's path, with its test and any ACLiC suffix; first one wins."""
    found: dict[str, tuple[TestSpec, str]] = {}
    for test in tests:
        match = _located(test, source)
        if match is not None:
            found.setdefault(match.group("path"), (test, match.group("suffix")))
    return found


def _veto_lists(interpreter: Interpreter) -> dict[str, str]:
    """Each vetoed file, and the first ``*_veto`` list naming it."""
    found: dict[str, str] = {}
    for name in list(interpreter.variables):
        if name.endswith("veto") and name not in ("all_veto", "tutorials_veto"):
            for path in interpreter.glob(interpreter.values(name)):
                found.setdefault(path, name)
    return found


def veto_reason(listed: str | None) -> str:
    """The reason a veto list gives, in words; one not in the table, by its option."""
    if listed is None:
        return NOT_A_TEST
    if listed in VETO_REASONS:
        return VETO_REASONS[listed]
    return f"needs ROOT built with {listed[: -len('_veto')]}"


def _from_test(path: str, test: TestSpec, suffix: str, names: Mapping[str, str]) -> Tutorial:
    return Tutorial(
        path=path,
        test=test.name,
        flags=_flags(test) if LANGUAGES[PurePosixPath(path).suffix] == "cxx" else (),
        suffix=suffix,
        passrc=test.passrc,
        failregex=test.failregex,
        labels=test.labels,
        depends=tuple(
            dict.fromkeys(names[d] for d in test.depends if d in names and names[d] != path)
        ),
        python_deps=tuple(dict.fromkeys(test.python_deps)),
        environment=test.environment,
    )


def catalog(
    directory: Path,
    variables: Mapping[str, str],
    targets: Iterable[str] = (),
    module: Callable[[str], bool] = lambda name: False,
) -> dict[str, Tutorial]:
    """Every tutorial under the directory, by path, as its CMakeLists describes it."""
    root = Path(directory)
    paths = discover(root)
    cmakelists = root / "CMakeLists.txt"
    if not cmakelists.is_file():
        return {path: Tutorial(path) for path in paths}
    interpreter = Interpreter(root, variables, targets, module).run_file(cmakelists)
    tested = _tested(interpreter.tests, str(root) + "/")
    names = {test.name: path for path, (test, _) in tested.items()}
    vetoes = _veto_lists(interpreter)
    found: dict[str, Tutorial] = {}
    for path in paths:
        if path in tested:
            test, suffix = tested[path]
            found[path] = _from_test(path, test, suffix, names)
        else:
            found[path] = Tutorial(path, vetoed=veto_reason(vetoes.get(path)))
    return found


def closure(tutorials: Mapping[str, Tutorial], chosen: Iterable[str]) -> list[str]:
    """The chosen tutorials and everything they depend on, dependencies first."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(path: str) -> None:
        if path in seen or path not in tutorials:
            return
        seen.add(path)
        for dependency in tutorials[path].depends:
            visit(dependency)
        order.append(path)

    for path in chosen:
        visit(path)
    return order


def levels(tutorials: Mapping[str, Tutorial], paths: Iterable[str]) -> list[list[str]]:
    """The paths in waves, each wave's dependencies all in earlier ones."""
    depth: dict[str, int] = {}
    for path in closure(tutorials, paths):
        below = [depth[d] for d in tutorials[path].depends if d in depth]
        depth[path] = 1 + max(below, default=-1)
    waves: list[list[str]] = [[] for _ in range(1 + max(depth.values(), default=-1))]
    for path, level in depth.items():
        waves[level].append(path)
    return waves
