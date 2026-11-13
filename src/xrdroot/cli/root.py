"""``xrdroot root`` (and ``xrdroot cint``): ROOT's ``root``, its C++ prompt and its command line.

    $ xrdroot root -l f.root              # f.root attached as _file0, then root [0]
    $ xrdroot root -l -b -q 'fit.C(1000)' # the macro run, then out with its value
    $ xrdroot root -l -b -q -e '.L lib.C+' run.C -e 'check()'
    $ echo 'TH1F h("h","h",10,0,1); h.Fill(.5); h.GetEntries()' | xrdroot root -l -b
    (double) 1.0000000

The options are ``root``'s, and the command line is taken as ``root`` takes
it: each ``-e`` line and each file in the order given. A ``.root`` file is
opened as ``_file0``, ``_file1``... - ``Attaching file f.root as _file0...``
- and a macro, ``x.C``, ``x.C(args)`` or ``x.C+`` (ACLiC), is run as ``.x``
runs it, after ``Processing x.C...``. Every file is looked for first, and
one that is not there ends ``root`` at once with status 2. A macro that fails
ends the processing of the rest; a failing ``-e`` line does not.

``-q`` leaves once they are done, exiting as ``root -q`` does: with the value
of the last line or macro run (``int x = 4;`` is 4, a pointer 255), ``1``
for code that did not compile, ``0`` after ``.q``, or the status given to
``exit(n)`` or ``gApplication->Terminate(n)``. Without ``-q`` the prompt
reads standard input, and ends with ``0`` at ``.q`` or its end. The banner
is shown when there is no ``-l`` and nothing on the command line to run.
``rootlogon.C`` in the working directory runs as the prompt starts and
``rootlogoff.C`` as it ends, unless ``-n``. There are never graphics windows
here, so ``-b`` is always so.

The Python prompt with ROOT's dot commands is ``xrdroot shell``; this one is
C++, translated into Python by :mod:`xrdroot.pycint` as it is typed.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..pycint.interactive import Interpreter, interact
from ..pycint.interactive.commands import dispatch
from ..pycint.interactive.terminal import say
from .run import LOGOFF, LOGON

__all__ = ["add_parser", "run", "banner", "Options", "options"]

#: ``root``'s flags that take no value, by what each sets.
FLAGS = {"-l": "no_banner", "-b": "batch", "-q": "quit", "-x": "exit_on_error",
         "-n": "no_logon", "-t": "threads", "-a": "splash", "-memstat": "memstat"}  # fmt: skip

#: Flags ``root`` takes with a value of their own after them.
VALUED = frozenset({"-e", "-config"})

#: The extensions of the files ``root`` attaches rather than runs.
ROOT_FILES = (".root",)

#: The exit status ``root`` gives a command line it will not run.
USAGE = 2

#: What ``-h`` prints.
HELP = """usage: root [-b B] [-x X] [-e E] [-n N] [-t T] [-q Q] [-l L] [-a A]
            [-config CONFIG] [-h HELP] [--version VERSION]
            [--web WEB] [dir] [data1.root...dataN.root] [file1.C...fileN.C]

xrdroot root: ROOT's C++ prompt, the C++ translated into Python as it is typed.
  -b         batch mode: no graphics windows (always so here)
  -x         exit at the first error
  -e E       run the line E, in its place among the files
  -n         run no rootlogon.C or rootlogoff.C
  -q         leave once the command line is done
  -l         no banner
"""


@dataclass
class Options:
    """``root``'s command line: its flags, and its ``-e`` lines and files, in their order."""

    no_banner: bool = False
    batch: bool = False
    quit: bool = False
    exit_on_error: bool = False
    no_logon: bool = False
    threads: bool = False
    splash: bool = False
    memstat: bool = False
    help: bool = False
    version: bool = False
    #: Each ``("e", line)`` and ``("file", argument)``, as given.
    items: list[tuple[str, str]] = field(default_factory=list)


def options(argv: Sequence[str]) -> Options:
    """``root``'s command line read as ``root`` reads it - flags anywhere, the rest in order."""
    found = Options()
    words = iter(argv)
    for word in words:
        if word in FLAGS:
            setattr(found, FLAGS[word], True)
        elif word in VALUED:
            value = next(words, "")
            if word == "-e":
                found.items.append(("e", value))
        elif word in ("-h", "-?", "--help"):
            found.help = True
        elif word == "--version":
            found.version = True
        elif not word.startswith("--"):  # ``--web=off``, ``--notebook``: nothing to do here
            found.items.append(("file", word))
    return found


class _RootParser(argparse.ArgumentParser):
    """``root``'s parser: the words as they come, for :func:`options` to read in order."""

    def parse_known_args(  # type: ignore[override]
        self, args: Sequence[str] | None = None, namespace: Any = None
    ) -> tuple[argparse.Namespace, list[str]]:
        parsed = namespace if namespace is not None else argparse.Namespace()
        parsed.argv = list(sys.argv[1:] if args is None else args)
        return parsed, []


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "root",
        help="ROOT's C++ prompt, as `root -l` is: C++ typed, translated and run",
        description="A C++ prompt in the style of ROOT's cling: open the ROOT files given "
        "as _file0, _file1..., run the macros and -e lines given in order, then prompt.",
        add_help=False,
    )
    parser.__class__ = _RootParser


def banner() -> str:
    """What the prompt says first without ``-l``: what it is - xrdroot, and not ROOT."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        release = version("xrdroot")
    except PackageNotFoundError:
        release = "(not installed)"
    return (
        f"xrdroot {release}: a C++ prompt in the style of ROOT's, translated into Python.\n"
        "  This is not ROOT; ROOT's names are xrdroot.pyroot's. .help lists the commands, "
        ".q leaves."
    )


def _logon(prompt: Interpreter, name: str) -> None:
    """``rootlogon.C`` or ``rootlogoff.C``, run as ``.x`` runs it - but, as in ROOT, not
    typed at the prompt, so no prompt number is used up and its value is no exit status."""
    if Path(name).is_file():
        status = prompt.last_status
        say(dispatch(prompt, f".x {name}") or [])
        prompt.last_status = status


def _missing(given: str) -> str | None:
    """What ``root`` says of a file on its command line that is not there - or ``None``."""
    from ..pycint.execute import called, exists
    from ..pycint.library.aclic import split

    spec = called(given)
    path = split(spec)[0]
    if os.path.isdir(path) or exists(spec) or "://" in path:
        return None
    if path.endswith(ROOT_FILES):
        return (f"Warning in <TApplication::GetOptions>: file {path} not found\n"
                f"root: unrecognized option '{path}'\n"
                "Try 'root --help' for more information.")  # fmt: skip
    return f"Error in <TApplication::GetOptions>: macro {path} not found"


def _checked(given: Options) -> int | None:
    """Every file looked for before anything runs, as ``root`` does; the status to end with
    if one is not there."""
    for what, text in given.items:
        said = _missing(text) if what == "file" else None
        if said is not None:
            sys.stderr.write(said + "\n")
            return USAGE
    return None


def _start(prompt: Interpreter, given: Options) -> None:
    """The ``-e`` lines and the files, in their order, until one ends the session - or a
    macro fails, which ends the processing of the command line."""
    for what, text in given.items:
        if prompt.finished or prompt.failed:
            return
        say(prompt.feed(text) if what == "e" else prompt.given(text))


def _version() -> int:
    from ..pyroot.core.troot import RELEASE

    sys.stdout.write(f"ROOT Version: {RELEASE} (xrdroot's C++ prompt)\n")
    return 0


def run(args: argparse.Namespace) -> int:
    given = options(args.argv)
    if given.help:
        sys.stdout.write(HELP)
        return 0
    if given.version:
        return _version()
    failed = _checked(given)
    if failed is not None:
        return failed
    return _session(given, args.argv)


def _session(given: Options, argv: list[str]) -> int:
    """``root`` itself: the banner, the logon macro, the command line, the prompt, logoff."""
    from ..pycint.runtime.application import started

    if not given.no_banner and not given.items:
        sys.stdout.write(banner() + "\n")
    with started(["root.exe", *argv]):
        prompt = Interpreter(exit_on_error=given.exit_on_error)
        logons = () if given.no_logon else (LOGON, LOGOFF)
        for name in logons[:1]:
            _logon(prompt, name)
        status = _running(prompt, given)
        for name in logons[1:]:
            _logon(prompt, name)
        prompt.close()
    sys.stdout.flush()
    return status


def _running(prompt: Interpreter, given: Options) -> int:
    """The command line, then - without ``-q`` - the prompt; the status ``root`` exits with.

    ``-q`` exits with the value of what ran last; a prompt left at ``.q`` or at the end of
    its input exits 0, whatever it ran.
    """
    if given.items and not given.quit:
        sys.stdout.write("\n")
    _start(prompt, given)
    return prompt.status if given.quit or prompt.finished else interact(prompt)
