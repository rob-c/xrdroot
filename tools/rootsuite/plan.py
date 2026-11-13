"""How xrdroot runs one test: the commands, where, with what - or why it cannot.

Each of ROOT's programs has an xrdroot stand-in taking the same command line:

* ``root.exe [options] [-e line]... [file.C[+][(args)]]...`` is ``xrdroot
  root`` with the same words - roottest's ``MACRO`` tests included, with the
  ``-e`` lines ``ROOTTEST_SETUP_MACROTEST`` puts first;
* ``hadd``, ``rootls``, ``rootrm``, ``rootmkdir``, ``rootmv``, ``rootcp`` and
  ``rootprint`` are ``xrdroot hadd``, ``ls``, ``rm``, ``mkdir``, ``mv``,
  ``cp`` and ``print``;
* a build step is ``xrdroot build`` (:mod:`~tools.rootsuite.builds`), and an
  executable it builds is run as itself, ``./exe args``;
* a program's sources with no dictionary or library of the tests' own run as
  ``xrdroot run sources... -- args``, a gtest's the same way, and a Python
  script as ``xrdroot run script.py``; a unittest file runs under pytest
  through :mod:`pyshim`, ``ROOT`` meaning ``xrdroot.pyroot``.

Anything else - ``cmake -E``, ``tar``, ``diff``, a shell script - runs as
written when this machine has it, with the working copy's ``bin`` (launchers
for the stand-ins, as ROOT's build tree has its programs) first on ``PATH``
and ``ROOTSYS`` the working copy, as ``RootTestDriver.cmake`` sets them.
What cannot be run is :class:`Unrunnable`: SKIP when the harness cannot
express it, UNSUPPORTED when it is ROOT's own tool xrdroot has no stand-in for.

Paths into ROOT's source tree and ``<build>`` are both rewritten to the
test's working copy, so nothing a test writes can land in the sources.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .model import Test

__all__ = ["Invocation", "Step", "Context", "Unrunnable", "plan", "command_line", "SHIM",
           "TOOLS", "NO_STAND_IN"]

#: The pytest launcher a unittest file is run through, in the xrdroot under test's Python.
SHIM = Path(__file__).with_name("pyshim.py")

#: ROOT's programs and the xrdroot subcommand taking the same command line.
TOOLS = {"root": "root", "root.exe": "root", "hadd": "hadd", "rootls": "ls", "rootrm": "rm",
         "rootmkdir": "mkdir", "rootmv": "mv", "rootcp": "cp", "rootprint": "print"}

#: ROOT's programs xrdroot has no stand-in for, and what each does.
NO_STAND_IN = {
    "rootcling": "generates a dictionary from rootcling's own command line",
    "rootcint": "generates a dictionary from rootcint's own command line",
    "genreflex": "generates a dictionary from genreflex's own command line",
    "root-config": "reports how ROOT was configured",
}  # fmt: skip

#: Programs the harness does not run though the machine has them, and why.
_NOT_RUN = {"make": "runs roottest's old Makefile, which compiles C++ with the system compiler "
                    "against ROOT's libraries"}


@dataclass(frozen=True)
class Unrunnable:
    """A test that is not run: SKIP (the harness cannot express it) or UNSUPPORTED."""

    status: str
    reason: str


@dataclass(frozen=True)
class Step:
    """A command run around the test: a build, a ``PRECMD`` or a ``POSTCMD``.

    ``shared`` builds are the build tree's - a library several tests link -
    so one run of each serves them all.
    """

    argv: tuple[str, ...]
    cwd: Path
    what: str
    shared: bool = False


@dataclass(frozen=True)
class Invocation:
    """One test's commands: its own argv and directory, and the steps around them."""

    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    merge: bool = False
    stdin: Path | None = None
    junit: Path | None = None
    before: tuple[Step, ...] = ()
    after: tuple[Step, ...] = ()
    commands: frozenset[str] = frozenset()
    copy: tuple[Path, ...] = ()


@dataclass(frozen=True)
class Context:
    """What every plan shares: ROOT's sources, the working copy of them, and the xrdroot to run."""

    root: Path
    work: Path
    python: str
    xrdroot_src: Path | None = None


def _rewrite(text: str, context: Context) -> str:
    return text.replace("<build>", str(context.work)).replace(str(context.root), str(context.work))


def _xrdroot(context: Context, *args: str) -> tuple[str, ...]:
    return (context.python, "-m", "xrdroot", *args)


def _with_args(files: list[str], args: tuple[str, ...]) -> list[str]:
    return [*files, "--", *args] if args else files


#: A command line made runnable: its argv and the xrdroot subcommands it uses.
Line = tuple[tuple[str, ...], frozenset[str]]


def command_line(words: Sequence[str], context: Context) -> Line | Unrunnable:
    """A command from ROOT's CMake as this machine runs it: ROOT's tools as xrdroot's."""
    if not words:
        return Unrunnable("SKIP", "the test has no command")
    words = _unwrapped(words)
    name = _tool(Path(words[0]).name)
    rest = tuple(_rewrite(word, context) for word in words[1:])
    if name in TOOLS:
        return _xrdroot(context, TOOLS[name], *rest), frozenset({TOOLS[name]})
    refused = _refused(name)
    if refused is not None:
        return refused
    if name.startswith("python"):
        return _python(rest, context)
    return _program_line(words[0], name, rest, context)


def _unwrapped(words: Sequence[str]) -> Sequence[str]:
    """``python <build>/bin/rootmv ...``: ROOT's Python tools, run by Python, are the tools."""
    python = Path(words[0]).name.startswith("python")
    return words[1:] if python and len(words) > 1 and _tool(Path(words[1]).name) in TOOLS \
        else words


def _tool(name: str) -> str:
    """A program's name, ``rootcp.py`` spelled as the ROOT tool ``rootcp`` it is."""
    return name.removesuffix(".py") if name.removesuffix(".py") in TOOLS else name


def _refused(name: str) -> Unrunnable | None:
    if name in NO_STAND_IN:
        return Unrunnable("UNSUPPORTED", f"It runs ROOT's {name}, which {NO_STAND_IN[name]}; "
                                         "xrdroot has no stand-in for it.")
    return Unrunnable("SKIP", _NOT_RUN[name]) if name in _NOT_RUN else None


def _python(rest: tuple[str, ...], context: Context) -> Line:
    """A Python script as a PyROOT script; any other Python command line as it is."""
    if rest and rest[0].endswith(".py"):
        return _xrdroot(context, "run", *_with_args([rest[0]], rest[1:])), frozenset({"run"})
    return (context.python, *rest), frozenset()


def _program_line(given: str, name: str, rest: tuple[str, ...],
                  context: Context) -> Line | Unrunnable:
    """Any other program, by path or from ``PATH``, if this machine has it."""
    first = _rewrite(given, context)
    if "/" not in given and shutil.which(first) is None:
        return Unrunnable("SKIP", f"runs {name}, which this machine does not have")
    return (first, *rest), frozenset()


def plan(test: Test, context: Context, logs: Path) -> Invocation | Unrunnable:
    """The invocation running ``test``, or why it is not run."""
    if test.skip:
        return Unrunnable("SKIP", test.skip)
    main = _ARGV[test.runner](test, context, logs)
    if isinstance(main, Unrunnable):
        return main
    around = _around(test, context)
    if isinstance(around, Unrunnable):
        return around
    return _invocation(test, context, logs, main, around)


def _invocation(test: Test, context: Context, logs: Path, main: Line,
                around: tuple[tuple[Step, ...], tuple[Step, ...], frozenset[str]]) -> Invocation:
    """The test's own command, its streams, and every step and subcommand around it."""
    before, after, used = around
    argv, commands = main
    building = frozenset({"build"}) if _shared(test) else frozenset()
    return Invocation(
        argv=argv, cwd=_cwd(test, context), env=_environment(test, context),
        merge=test.check_out and test.check_err and not test.errref,
        stdin=context.work / test.input if test.input else None,
        junit=logs / "junit.xml" if test.runner == "pyunittest" else None,
        before=(*_builds(test, context), *before), after=after,
        commands=commands | used | building,
        copy=tuple(context.work / name for name in test.copy))


def _cwd(test: Test, context: Context) -> Path:
    """Where a test runs: its ``WORKING_DIR``, else its directory; a build, where it builds."""
    if test.runner == "setup" and test.builds:
        return context.work / test.builds[-1][0]
    return context.work / (test.working_dir or test.directory)


def _shared(test: Test) -> bool:
    """Whether a test's builds are the build tree's, run before it rather than as it."""
    return bool(test.builds) and test.runner != "setup"


def _step(build: tuple[str, ...], context: Context, shared: bool) -> Step:
    argv = _xrdroot(context, "build", *(_rewrite(word, context) for word in build[1:]))
    return Step(argv, context.work / build[0], f"building with xrdroot build {build[1]}", shared)


def _builds(test: Test, context: Context) -> list[Step]:
    """The builds that come first: all of a program's; all but the last of a setup test's."""
    steps = test.builds if _shared(test) else test.builds[:-1]
    return [_step(build, context, _shared(test)) for build in steps]


def _around(test: Test, context: Context) -> tuple[tuple[Step, ...], tuple[Step, ...],
                                                   frozenset[str]] | Unrunnable:
    """The driver's ``PRECMD`` and ``POSTCMD``, run in the test's directory."""
    cwd = _cwd(test, context)
    steps: list[tuple[Step, ...]] = []
    used: frozenset[str] = frozenset()
    for words, what in ((test.precmd, "the pre-command"), (test.postcmd, "the post-command")):
        line = command_line(words, context) if words else None
        if isinstance(line, Unrunnable):
            return line
        steps.append((Step(line[0], cwd, what),) if line else ())
        used |= line[1] if line else frozenset()
    return steps[0], steps[1], used


def _environment(test: Test, context: Context) -> dict[str, str]:
    """The test's ``ENVIRONMENT``, its paths in the working copy, less the library paths of
    CI's build tree; and ROOT's ``ROOT_HIST=0``."""
    env = {"ROOT_HIST": "0"}
    for entry in test.environment:
        name, _, value = entry.partition("=")
        if name and name not in ("PYTHONPATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            env[name] = _rewrite(value, context)
    return env


def _program(test: Test, context: Context, logs: Path) -> Line | Unrunnable:
    """A built executable run as itself; else a program's sources through ``xrdroot run``."""
    if test.launch:
        return (_rewrite(test.launch, context), *test.args), frozenset()
    if not test.files:
        return Unrunnable("SKIP", f"runs {test.program or 'a program'} with no sources the "
                                  "harness can find")
    files = [str(context.work / name) for name in test.files]
    return _xrdroot(context, "run", *_with_args(files, test.args)), frozenset({"run"})


def _script(test: Test, context: Context, logs: Path) -> Line:
    argv = _xrdroot(context, "run", *_with_args([str(context.work / test.files[0])], test.args))
    return argv, frozenset({"run"})


def _pyunittest(test: Test, context: Context, logs: Path) -> Line:
    files = [str(context.work / name) for name in test.files]
    return (context.python, str(SHIM), "--junitxml", str(logs / "junit.xml"), *files), frozenset()


def _root(test: Test, context: Context, logs: Path) -> Line:
    """``root.exe`` and its words, as ``xrdroot root`` with the same words."""
    return _xrdroot(context, "root", *(_rewrite(arg, context) for arg in test.args)), frozenset(
        {"root"})


def _command(test: Test, context: Context, logs: Path) -> Line | Unrunnable:
    return command_line(test.command, context)


def _setup(test: Test, context: Context, logs: Path) -> Line | Unrunnable:
    """A build step: its last ``xrdroot build``, the others run first."""
    if not test.builds:
        return Unrunnable("SKIP", f"a {test.setup_of} build step with nothing to build")
    step = _step(test.builds[-1], context, False)
    return step.argv, frozenset({"build"})


_ARGV = {
    "gtest": _program,
    "program": _program,
    "script": _script,
    "pyunittest": _pyunittest,
    "root": _root,
    "command": _command,
    "setup": _setup,
}
