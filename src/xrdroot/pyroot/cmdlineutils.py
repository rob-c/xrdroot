"""``cmdLineUtils``: the module ROOT writes ``rootls``, ``rootcp``, ``rootmv`` and ``rootrm`` on.

ROOT's command-line tools are Python programs in ``main/python``, each a thin
front over this one module: it turns ``file.root:dir/h*`` into the file and the
paths in it that the pattern names, walks the directories of a file key by key,
and offers the two stream redirectors they wrap their noise in. A test of
ROOT's - ``roottest/main/testPatternToFileNameAndPathSplitList.py`` - imports
the module itself by that name and calls the pattern function, so the module is
here, under the name ROOT installs it, rather than only as the ``xrdroot ls``
subcommand that does the same work.

The functions keep ROOT's names, arguments and return shapes - a list of
``(fileName, pathSplitList)`` tuples, each path split into its parts - because
what calls them is ROOT's own code. What reads the file is this library's
:func:`xrdroot.file.open_root`, so the patterns are answered from the key list
without ROOT.
"""

from __future__ import annotations

import fnmatch
import glob
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO, Any

from ..errors import FormatError
from ..file import Directory, ROOTFile, open_root

__all__ = [
    "fileNameListMatch",
    "pathSplitListMatch",
    "patternToFileNameAndPathSplitList",
    "patternToPathSplitList",
    "stderrRedirected",
    "stdoutRedirected",
    "streamRedirected",
]

#: Where this module's warnings go. ROOT's own module calls ``logging.basicConfig`` as it
#: imports, which would take over the logging of whatever imports the namespace; a warning
#: of this module is worth printing and not worth restructuring an application's log for.
LOGGER = logging.getLogger("cmdLineUtils")

#: The suffixes ROOT's tools read objects out of. A pattern naming anything else is refused
#: as ROOT refuses it, rather than opened to see what it holds.
EXTENSIONS = (".root", ".pcm")

#: The schemes a source may be the URL of, which are opened as they are, never globbed.
SCHEMES = ("http", "https", "root", "gs", "s3")

#: The classes a key holding a directory is written as: the only keys a path walks into.
DIRECTORIES = ("TDirectory", "TDirectoryFile")


@contextmanager
def streamRedirected(
    source: IO[str] | None = None, destination: str = os.devnull
) -> Iterator[IO[str]]:
    """Send everything written to ``source`` to ``destination`` for the length of the block.

    At the file descriptor, as ROOT's redirectors do, not by rebinding
    ``sys.stdout``: what writes to the stream may be C code that kept its own
    copy of the descriptor, and a redirect that missed it would leave the very
    output the block exists to hide. ``source`` is a stream with a descriptor;
    ``destination`` is a path, ``os.devnull`` when nothing says otherwise.
    """
    stream = sys.stdout if source is None else source
    stream.flush()
    with os.fdopen(os.dup(stream.fileno()), "wb") as saved, open(destination, "wb") as sink:
        os.dup2(sink.fileno(), stream.fileno())
        try:
            yield stream
        finally:
            stream.flush()
            os.dup2(saved.fileno(), stream.fileno())


def stdoutRedirected() -> Any:
    """Hide what is printed to standard output, as ROOT's tools hide their imports' noise."""
    return streamRedirected(sys.stdout)


def stderrRedirected() -> Any:
    """Hide what is printed to standard error, as ROOT's tools hide a warning of ROOT's own."""
    return streamRedirected(sys.stderr)


def fileNameListMatch(filePattern: str, wildcards: bool) -> list[str]:
    """The files a pattern names, ``~`` and ``$VARS`` expanded, as ROOT's tools expand them."""
    names = glob.glob(filePattern) if wildcards else [filePattern]
    return [os.path.expandvars(os.path.expanduser(name)) for name in names]


def pathSplitListMatch(fileName: str, objPattern: str, wildcards: bool) -> list[list[str]]:
    """The paths in ``fileName`` that ``objPattern`` names, split part by part.

    Without wildcards the pattern is the answer, as ROOT takes it: the file is
    not opened, so what the pattern names need not be there.
    """
    if wildcards:
        return patternToPathSplitList(fileName, objPattern)
    return [[piece for piece in objPattern.split("/") if piece]]


def patternToPathSplitList(fileName: str, pattern: str) -> list[list[str]]:
    """Every path in ``fileName`` that matches ``pattern``, as a list of name parts.

    Each ``/``-separated part is a shell pattern matched against the names of
    the keys at that depth, so ``*/*`` is the contents of every directory. A
    part that matches nothing ends the walk there, and a pattern that matched
    nothing at all is warned about, as ROOT's tools warn of it.
    """
    try:
        rootFile = open_root(fileName)
    except (OSError, FormatError) as why:
        LOGGER.warning("%s", why)
        return []

    with rootFile:
        pathSplitList: list[list[str]] = [[]]
        for piece in [part for part in pattern.split("/") if part]:
            pathSplitList = _walk(rootFile, pathSplitList, piece)

    if not pathSplitList:
        LOGGER.warning("can't find %s in %s", pattern, fileName)
    return pathSplitList


def _walk(rootFile: ROOTFile, pathSplitList: list[list[str]], piece: str) -> list[list[str]]:
    """One part of a pattern: what each path so far matches, one level deeper in its directory."""
    matched: list[list[str]] = []
    for pathSplit in pathSplitList:
        names = _keyNames(rootFile, pathSplit)
        if names is not None:
            matched.extend([*pathSplit, name] for name in names if fnmatch.fnmatch(name, piece))
    return matched


def _keyNames(rootFile: ROOTFile, pathSplit: list[str]) -> list[str] | None:
    """The names of the keys ``pathSplit`` leads to, or ``None`` if it does not lead into a
    directory: a path never walks into a histogram, whatever the pattern after it says."""
    here: Directory = rootFile
    for piece in pathSplit:
        if here.classnames().get(piece) not in DIRECTORIES:
            return None
        here = here[piece]
    return here.keys()


def patternToFileNameAndPathSplitList(
    pattern: str, wildcards: bool = True
) -> list[tuple[str, list[list[str]]]]:
    """The files ``pattern`` names, and in each the paths it names in it.

    ``test.root`` is the whole file, ``test.root:tof`` the object called
    ``tof`` in it, ``*.root:h*/*`` every object of every histogram directory of
    every ROOT file here. A URL is opened as the one file it names and is never
    globbed, since the shell has no way to look inside it.
    """
    if "://" in pattern and pattern.partition("://")[0] in SCHEMES:
        fileName, colon, objPattern = pattern.rpartition(".root:")
        if not colon:
            return [(pattern, [[]])]
        whole = fileName + ".root"
        return [(whole, pathSplitListMatch(whole, objPattern, wildcards))]

    filePattern, colon, objPattern = pattern.partition(":")
    if not any(fnmatch.fnmatch(filePattern, "*" + suffix) for suffix in EXTENSIONS):
        LOGGER.warning("%s: No such file (or extension not supported)", pattern)
        return []
    if not colon:
        return [(name, [[]]) for name in fileNameListMatch(filePattern, wildcards)]
    return [
        (name, pathSplitListMatch(name, objPattern, wildcards))
        for name in fileNameListMatch(filePattern, wildcards)
    ]


def __getattr__(name: str) -> Any:
    """ROOT's other helpers here - ``copyRootObject``, ``deleteRootObject``, the argparsers -
    are answered with the subcommand that does the same work, not a bare ``NameError``.

    It is an ``AttributeError`` and not a refusal, because the namespace finds a class by
    asking each of its modules for it and reads ``AttributeError`` as "not here"; a refusal
    would turn every such lookup into a failure. A name starting with ``_`` says nothing, as
    Python's own probes of a module expect.
    """
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    raise AttributeError(
        f"cmdLineUtils.{name} is not here. xrdroot writes that tool as a subcommand rather than "
        f"over this module, so run xrdroot ls, cp, mv, rm or print on the files instead of "
        f"calling it."
    )


# ROOT's tools import this module by its own name - ``from cmdLineUtils import *`` - because
# ``thisroot.sh`` puts it beside them, not under ``ROOT``. Registering it here makes that
# import work wherever the namespace has been imported, which is what a ROOT tool does first.
sys.modules.setdefault("cmdLineUtils", sys.modules[__name__])
