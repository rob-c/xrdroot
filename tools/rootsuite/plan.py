"""How xrdroot runs one test: the command line, where, with what - or why it cannot.

Every C++ program and macro goes through ``python -m xrdroot run``: a
program's sources all at once, its ctest arguments after ``--``
(``main(argc, argv)``); a gtest's sources the same way, ``xrdroot run``
recognising ``TEST(...)``; a roottest macro as ``file.C`` or
``file.C(args)``; a Python script or roottest ``.py`` macro as a PyROOT
script. A unittest file runs under pytest through :mod:`pyshim`, with
``ROOT`` meaning ``xrdroot.pyroot``. ``root.exe -b -q file.C`` command lines
become ``xrdroot run file.C``, and ``hadd``, ``rootls`` and ``rootcp`` the
xrdroot subcommands that take their command lines. Anything else - a compiled
executable, genreflex, rootcling, ``root -e`` - is a skip with its reason.

Paths into ROOT's source tree and ``<build>`` are both rewritten to the
test's working copy, so nothing a test writes can land in the sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .model import Test

__all__ = ["Invocation", "Context", "plan", "SHIM", "TOOLS"]

#: The pytest launcher a unittest file is run through, in the xrdroot under test's Python.
SHIM = Path(__file__).with_name("pyshim.py")

#: ROOT's command-line tools that xrdroot has a subcommand taking the same command line.
TOOLS = {"hadd": "merge", "rootls": "ls", "rootcp": "cp"}

#: ``root``'s flags that change nothing a batch run of one macro does.
_QUIET_FLAGS = {"-b", "-q", "-l", "-n", "-x", "-a", "--web=off", "-config", "-memstat"}


@dataclass(frozen=True)
class Invocation:
    """One command to run for a test: its argv, its directory, and how its output is caught."""

    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    merge: bool = False
    stdin: Path | None = None
    junit: Path | None = None


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


def plan(test: Test, context: Context, logs: Path) -> Invocation | str:
    """The invocation running ``test``, or the sentence saying why the harness cannot."""
    reason = _unrunnable(test)
    if reason:
        return reason
    argv = _ARGV[test.runner](test, context, logs)
    if isinstance(argv, str):
        return argv
    cwd = context.work / (test.working_dir or test.directory)
    junit = logs / "junit.xml" if test.runner == "pyunittest" else None
    return Invocation(
        argv=tuple(_rewrite(arg, context) for arg in argv), cwd=cwd, env=_environment(test),
        merge=test.check_out and test.check_err and not test.errref,
        stdin=context.work / test.input if test.input else None, junit=junit)


def _unrunnable(test: Test) -> str:
    if test.skip:
        return test.skip
    if test.runner == "setup":
        return f"a {test.setup_of} build step, which has nothing to run without a compiler"
    if "executable" in test.needs:
        return "runs an executable roottest compiles, which the harness does not build"
    return ""


def _environment(test: Test) -> dict[str, str]:
    """The test's ``ENVIRONMENT``, less what names CI's build tree; ROOT's ``ROOT_HIST=0``."""
    env = {"ROOT_HIST": "0"}
    for entry in test.environment:
        name, _, value = entry.partition("=")
        if name and name not in ("PYTHONPATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            env[name] = value
    return env


def _program(test: Test, context: Context, logs: Path) -> tuple[str, ...] | str:
    if not test.files:
        return f"runs {test.program or 'a program'} with no sources the harness can find"
    files = [str(context.work / name) for name in test.files]
    return _xrdroot(context, "run", *_with_args(files, test.args))


def _macro(test: Test, context: Context, logs: Path) -> tuple[str, ...]:
    path = str(context.work / test.files[0])
    called = f"{path}({','.join(test.args)})" if test.args else path
    return _xrdroot(context, "run", called)


def _script(test: Test, context: Context, logs: Path) -> tuple[str, ...]:
    return _xrdroot(context, "run", *_with_args([str(context.work / test.files[0])], test.args))


def _pyunittest(test: Test, context: Context, logs: Path) -> tuple[str, ...]:
    files = [str(context.work / name) for name in test.files]
    return (context.python, str(SHIM), "--junitxml", str(logs / "junit.xml"), *files)


def _root(test: Test, context: Context, logs: Path) -> tuple[str, ...] | str:
    """``root.exe -b -q [-l] file.C`` as ``xrdroot run file.C``; anything more is a skip."""
    if "-e" in test.args:
        return "runs root.exe -e expressions, which xrdroot run cannot take"
    macros = [arg for arg in test.args if arg not in _QUIET_FLAGS and not arg.startswith("--")]
    if len(macros) != 1:
        return f"runs root.exe with {len(macros)} macros, where xrdroot run takes one"
    macro = macros[0]
    if not macro.startswith(("/", "<build>")):
        macro = str(context.work / test.directory / macro)
    plain, paren, rest = macro.partition("(")
    return _xrdroot(context, "run", plain.rstrip("+") + paren + rest)


def _command(test: Test, context: Context, logs: Path) -> tuple[str, ...] | str:
    name = Path(test.command[0]).name if test.command else ""
    tool = TOOLS.get(name.removesuffix(".exe"))
    if tool is None:
        return f"runs {name or 'a command'}, which is neither ROOT nor a program xrdroot runs"
    return _xrdroot(context, tool, *test.args)


_ARGV = {
    "gtest": _program,
    "program": _program,
    "macro": _macro,
    "script": _script,
    "pyunittest": _pyunittest,
    "root": _root,
    "command": _command,
}
