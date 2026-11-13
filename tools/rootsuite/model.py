"""One test as ROOT's CMake registers it, and the registry discovery fills.

ROOT's ctest suite is not a list in a file: each test is what a CMake macro
(``ROOT_ADD_GTEST``, ``ROOTTEST_ADD_TEST``...) made of its arguments in the
directory it was called from. A :class:`Test` keeps what those macros
decided - the ctest name, how the test is run, how its output is judged -
in a form that needs no CMake to read, so the runner and the reports work
from ``list``'s output alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import PurePosixPath
from typing import Any

__all__ = ["KINDS", "CI_COUNTS", "Test", "Target", "Dictionary", "Registry", "kind_of",
           "area_of", "make_id"]

#: The kinds of test, keyed by the prefix of their ctest name, in the order reports list them.
KINDS = {"gtest-": "gtest", "roottest-": "roottest", "pyunittests-": "pyunittest",
         "test-": "test", "tutorial-": "tutorial"}  # fmt: skip

#: How many tests of each kind one ROOT 6.40 CI job ran: what discovery is held against.
CI_COUNTS = {"tutorial": 918, "gtest": 389, "roottest": 2210, "pyunittest": 109, "test": 64}

#: How a test is run: a gtest program, a ``main`` program, a Python unittest file, a Python
#: script, a ROOT command line (roottest's macros among them), another command, a build step.
RUNNERS = ("gtest", "program", "pyunittest", "script", "root", "command", "setup")


def kind_of(ctest: str, directory: str) -> str:
    """The kind a ctest name belongs to; ``test/``'s are stress programs by any name."""
    for prefix, kind in KINDS.items():
        if ctest.startswith(prefix):
            return kind
    return "test" if directory.split("/")[0] == "test" else "other"


def make_id(kind: str, directory: str, name: str) -> str:
    """A stable id: the kind, the directory without its ``test`` steps, the CMake name.

    ``gtest/hist/hist/testTH1`` for ``hist/hist/test``'s ``testTH1``;
    roottest's ids drop the leading ``roottest`` the kind already says.
    """
    parts = [part for part in directory.split("/") if part and part != "test"]
    if kind == "roottest" and parts[:1] == ["roottest"]:
        parts = parts[1:]
    return "/".join([kind, *parts, name])


def area_of(test_id: str) -> str:
    """The top-level area of an id: its first directory, or first two for roottest's deep tree.

    A test directly in ``test/`` (``test/stresshistogram``) is in area ``test``.
    """
    parts = PurePosixPath(test_id).parts
    depth = 3 if parts[0] == "roottest" else 2
    if len(parts) <= 2:
        return parts[0]
    return "/".join(parts[1:min(depth, len(parts) - 1)])


@dataclass(frozen=True)
class Test:
    """One registered ctest test and everything needed to run and judge it.

    Paths (``directory``, ``files``, ``outref``...) are relative to ROOT's
    source tree; ``<build>`` in an argument stands for the build tree, which
    the runner replaces by the test's working copy.

    ``builds`` are the ``xrdroot build`` commands that make what it runs -
    each a directory, then the command's arguments - and ``launch`` the
    executable one of them writes, run instead of ``xrdroot run``. ``copy``
    is ``COPY_TO_BUILDDIR``; ``precmd`` and ``postcmd`` the driver's ``PRE``
    and ``POST`` commands.
    """

    __test__ = False  # a ROOT test described, not one of pytest's to collect

    id: str
    ctest: str
    kind: str
    directory: str
    runner: str
    command: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    args: tuple[str, ...] = ()
    outref: str = ""
    errref: str = ""
    outcnv: str = ""
    outcnvcmd: tuple[str, ...] = ()
    input: str = ""
    passrc: int | None = None
    passregex: tuple[str, ...] = ()
    failregex: tuple[str, ...] = ()
    will_fail: bool = False
    check_out: bool = True
    check_err: bool = True
    custom_diff: bool = False
    depends: tuple[str, ...] = ()
    fixtures_setup: tuple[str, ...] = ()
    fixtures_required: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    python_deps: tuple[str, ...] = ()
    timeout: float | None = None
    working_dir: str = ""
    program: str = ""
    needs: tuple[str, ...] = ()
    skip: str = ""
    setup_of: str = ""
    builds: tuple[tuple[str, ...], ...] = ()
    launch: str = ""
    copy: tuple[str, ...] = ()
    precmd: tuple[str, ...] = ()
    postcmd: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in ((), "", None)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Test:
        known = {item.name: item for item in fields(cls)}
        values = {key: _tuples(value) for key, value in data.items() if key in known}
        return cls(**values)


def _tuples(value: Any) -> Any:
    """JSON's lists as the tuples a :class:`Test` holds, nested ones included."""
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


@dataclass
class Dictionary:
    """A dictionary rootcling or genreflex would generate: from which headers, selected how.

    ``name`` is the generated source's stem (``G__Event``, ``base_cling``);
    the paths are below ROOT's source tree.
    """

    name: str
    directory: str
    headers: list[str] = field(default_factory=list)
    linkdef: str = ""
    selection: str = ""
    includes: list[str] = field(default_factory=list)


@dataclass
class Target:
    """An executable or library a test directory builds: its sources and what it links.

    ``dictionary`` is whether building it runs rootcling or genreflex - a
    ``G__`` source, or a library package with a LinkDef; ``compiled`` that it
    is one of roottest's executables, built by a setup step and run as a
    compiled program. ``output`` is what ROOT names the built file, less
    ``lib`` and its suffix: what ``xrdroot build --name`` is given.
    ``dictionaries`` are those generated into it.
    """

    files: list[str] = field(default_factory=list)
    libraries: list[str] = field(default_factory=list)
    dictionary: bool = False
    compiled: bool = False
    directory: str = ""
    output: str = ""
    dictionaries: list[Dictionary] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)


@dataclass
class Registry:
    """What discovery has seen so far: tests, and the build targets tests refer to.

    Targets are global in CMake - a function or library defined in one
    directory is visible in every later one - so this is shared by every
    directory's interpreter. Only targets built in test directories are kept
    as ``programs`` and ``libraries``: ROOT's own libraries are what xrdroot is.
    """

    root: str
    features: frozenset[str] = frozenset()
    tests: list[Test] = field(default_factory=list)
    test_subdirs: list[str] = field(default_factory=list)
    programs: dict[str, Target] = field(default_factory=dict)
    libraries: dict[str, Target] = field(default_factory=dict)
    dictionaries: dict[str, Dictionary] = field(default_factory=dict)
    functions: dict[str, Any] = field(default_factory=dict)
    targets: set[str] = field(default_factory=set)
    visited: set[str] = field(default_factory=set)

    def add(self, test: Test) -> None:
        self.tests.append(test)
