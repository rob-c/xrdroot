"""``xrdroot mv``: ROOT's ``rootmv`` - objects moved within a ROOT file or into another.

    $ xrdroot mv in.root:hpx in.root:histo          # renamed
    $ xrdroot mv in.root:hpx in.root:dir            # into a directory
    $ xrdroot mv 'in.root:h*' out.root              # into another file
    $ xrdroot mv --recreate in.root:h out.root      # into a file made anew

``rootmv`` is a Python program over ``cmdLineUtils``, and this is that
program over xrdroot: each SOURCE is shell patterns - files, then the path
in them a level at a time - and DEST a file and a path in it. What each
source matches is copied as ``rootmv`` copies it, directories and all,
replacing what is in the way, and once it is copied it is deleted where it
was; a whole file moved is removed. Its messages are ``rootmv``'s logging
warnings, and its status the number of things that went wrong.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from ..errors import ROOTError
from ..file import Directory, Key, ROOTFile, open_root
from ..merging import copy
from ..writer import WritableDirectory, WritableFile, create
from ..wupdate import update
from .sources import DIRECTORIES
from .tooling import raw_parser

__all__ = ["add_parser", "run"]

#: ``rootmv``'s description and epilog, as its ``--help`` prints them.
DESCRIPTION = "Move objects from ROOT files to another"
EPILOG = """Examples:
- rootmv source.root:hist* dest.root
  Move all histograms whose named starts with 'hist' from 'source.root' to 'dest.root'.

- rootmv source1.root:hist1 source2.root:hist2 dest.root
  Move histograms 'hist1' from 'source1.root' and 'hist2' from 'source2.root' to 'dest.root'.

- rootmv --recreate source.root:hist dest.root
  Recreate the destination file 'dest.root' and move the histogram named 'hist' from 'source.root' into it.

- rootmv -c 101 source.root:hist dest.root
  Change the compression settings of the destination file 'dest.root' to ZLIB algorithm with compression level 1 and move the histogram named 'hist' from 'source.root' into it.
  Meaning of the '-c' argument is given by 'compress = 100 * algorithm + level'.
  Other examples of usage:
    * -c 509 : ZSTD with compression level 9
    * -c 404 : LZ4 with compression level 4
    * -c 207 : LZMA with compression level 7
  For more information see https://root.cern.ch/doc/master/classTFile.html#ad0377adf2f3d88da1a1f77256a140d60
  and https://root.cern.ch/doc/master/structROOT_1_1RCompressionSetting.html

  """  # noqa: E501

#: ``cmdLineUtils``'s words for what goes wrong.
TARGET_ERROR = "target '{0}' is not a directory"
OVERWRITE_ERROR = "cannot overwrite non-directory '{0}' with directory '{1}'"
DELETE_ERROR = "object {0} was not existing, so it is not deleted"
MOVE_ERROR = "error during copy of {0}, it is not removed from {1}"
MANY_OCCURENCE_WARNING = (
    "Several versions of '{0}' are present in '{1}'. Only the most recent will be considered."
)
#: The trees, which ``rootmv`` clones rather than reads and writes.
TREES = ("TTree", "TNtuple", "TNtupleD")
#: What a source or DEST names, as ``cmdLineUtils`` tells it: a file, maybe with a path.
ROOT_FILE, ROOT_OBJECT = "*.root", "*.root:*"
REMOTE = ("htt*://*.root", "root://*.root", "s3://*.root", "gs://*.root")


def add_parser(subparsers: Any) -> None:
    raw_parser(subparsers, "mv", "move objects between ROOT files, as rootmv does")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rootmv", description=DESCRIPTION, epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("SOURCE", nargs="+", help="Source file")
    parser.add_argument("DEST", help="Destination file")
    parser.add_argument("-c", "--compress", type=int,
                        help="change the compression settings of the\n"
                        "destination file (if not already existing).")  # fmt: skip
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="prompt before every removal.")  # fmt: skip
    parser.add_argument("--recreate", action="store_true", help="recreate the destination file.")
    return parser


def _warn(message: str) -> None:
    """A warning as ``cmdLineUtils``'s logging prints one."""
    sys.stderr.write(f"WARNING: {message}\n")


def run(args: argparse.Namespace) -> int:
    try:
        parsed = _parser().parse_args(args.argv)
    except SystemExit as leaving:
        return int(leaving.code or 0)
    sources = [found for pattern in parsed.SOURCE for found in _sources(pattern)]
    dest = _sources(parsed.DEST, wildcards=False)
    if not sources or not dest:
        return 1
    name, (path,) = dest[0]
    return _Move(parsed.interactive).run(sources, name, path, parsed.compress, parsed.recreate)


# -- what the patterns name -------------------------------------------------------------


def _sources(pattern: str, wildcards: bool = True) -> list[tuple[str, list[list[str]]]]:
    """``patternToFileNameAndPathSplitList``: each file a pattern names, and its paths."""
    if any(fnmatch.fnmatch(pattern, f"{remote}:*") for remote in REMOTE):
        name, objects = pattern.rsplit(":", 1)
        return [(name, _paths(name, objects, wildcards))]
    if any(fnmatch.fnmatch(pattern, remote) for remote in REMOTE):
        return [(pattern, [[]])]
    if fnmatch.fnmatch(pattern, ROOT_OBJECT):
        files, objects = pattern.split(":")[:2]
        return [(name, _paths(name, objects, wildcards)) for name in _files(files, wildcards)]
    if fnmatch.fnmatch(pattern, ROOT_FILE):
        return [(name, [[]]) for name in _files(pattern, wildcards)]
    _warn(f"{pattern}: No such file (or extension not supported)")
    return []


def _files(pattern: str, wildcards: bool) -> list[str]:
    names = glob.glob(pattern) if wildcards else [pattern]
    return [os.path.expandvars(os.path.expanduser(name)) for name in names]


def _paths(name: str, pattern: str, wildcards: bool) -> list[list[str]]:
    if not wildcards:
        return [[part for part in pattern.split("/") if part]]
    try:
        with open_root(name) as file:
            found = _matched(file, [part for part in pattern.split("/") if part])
    except (OSError, ValueError, ROOTError):
        _warn(f"File {name} does not exist")
        return []
    if not found:
        _warn(f"can't find {pattern} in {name}")
    return _once(found, name)


def _matched(file: ROOTFile, parts: list[str]) -> list[list[str]]:
    """Each path whose every level a pattern matches, a key for every cycle of a name."""
    found: list[list[str]] = [[]]
    for part in parts:
        deeper = []
        for path in found:
            directory = _directory(file, path)
            if directory is not None:
                deeper += [[*path, key.name] for key in directory.all_keys()
                           if fnmatch.fnmatch(key.name, part)]  # fmt: skip
        found = deeper
    return found


def _once(found: list[list[str]], name: str) -> list[list[str]]:
    """``manyOccurenceRemove``: a name met for several cycles kept once, with a warning."""
    if len(found) > 1:
        for path in list(found):
            if found.count(path) != 1:
                _warn(MANY_OCCURENCE_WARNING.format("/".join(path), name))
                while path in found and found.count(path) != 1:
                    found.remove(path)
    return found


def _directory(file: Directory, path: list[str]) -> Directory | None:
    """The directory at ``path`` in a file being read, or ``None`` if it is not one."""
    here = file
    for part in path:
        try:
            key = here.key(part)
        except KeyError:
            return None
        if key.classname not in DIRECTORIES:
            return None
        here = here._subdirectory(key)
    return here


# -- the move ---------------------------------------------------------------------------


@dataclass
class _Move:
    """One ``rootmv`` run."""

    interactive: bool
    dest: WritableFile | None = None
    dest_name: str = ""
    #: What this run wrote, by directory and name: a tree's key comes only as the file closes.
    written: dict[tuple[str, str], str] = field(default_factory=dict)

    def run(
        self,
        sources: list[tuple[str, list[list[str]]]],
        name: str,
        path: list[str],
        compress: int | None,
        recreate: bool,
    ) -> int:
        if compress is not None and os.path.isfile(name):
            _warn("can't change compression settings on existing file")
            return 1
        self.dest_name = name
        try:
            self.dest = _destination(name, compress, recreate)
        except (OSError, ValueError, ROOTError):
            _warn(f"File {name} does not exist")
            return 1
        status = 0
        with self.dest:
            for each, paths in sources:
                status += self._file(each, paths, path, len(sources) == 1)
        return status

    def _file(self, name: str, paths: list[list[str]], dest: list[str], one_file: bool) -> int:
        """``_moveObjects``: each path of one source file copied, then deleted there."""
        try:
            reading = open_root(name)
        except (OSError, ValueError, ROOTError):
            _warn(f"File {name} does not exist")
            return 1
        status = 0
        assert self.dest is not None
        with reading:
            source = self.dest if name == self.dest_name else update(name)
            for path in paths:
                status += self._one(reading, source, path, dest, one_file and len(paths) == 1)
            if source is not self.dest:
                source.close()
        return status

    def _one(
        self, reading: ROOTFile, source: WritableFile, path: list[str], dest: list[str], one: bool
    ) -> int:
        failed = self._copy(reading, path, dest, one)
        if failed:
            _warn(MOVE_ERROR.format("/".join(path), reading.name))
            return failed
        return self._delete(reading.name, source, path)

    # -- copyRootObject ----------------------------------------------------------------

    def _copy(self, reading: ROOTFile, path: list[str], dest: list[str], one: bool) -> int:
        multiple = not (one and path)
        status = 0
        if multiple and dest and not self._is_directory(dest):
            _warn(TARGET_ERROR.format(dest[-1]))
            status = 1
        if not path:
            return status + self._recursive(reading, path, dest)
        named = dest[-1] if not multiple and dest and not self._exists(dest) else ""
        return status + self._placed(reading, path, dest, named)

    def _placed(self, reading: ROOTFile, path: list[str], dest: list[str], named: str) -> int:
        """One source copied where DEST says: a new name, into a directory, or over a thing."""
        if _directory(reading, path) is not None:
            return self._directory_copy(reading, path, dest, named)
        if named:
            return self._recursive(reading, path, dest[:-1], named)
        if self._is_directory(dest):
            return self._recursive(reading, path, dest)
        return self._recursive(reading, path, dest[:-1], dest[-1])

    def _directory_copy(
        self, reading: ROOTFile, path: list[str], dest: list[str], named: str
    ) -> int:
        if named:
            self._mkdir([*dest[:-1], named])
            return self._recursive(reading, path, [*dest[:-1], named])
        if not self._is_directory(dest):
            _warn(OVERWRITE_ERROR.format(dest[-1], path[-1]))
            return 1
        inside = [*dest, path[-1]]
        if not self._exists(inside):
            self._mkdir(inside)
        if not self._is_directory(inside):
            _warn(OVERWRITE_ERROR.format(path[-1], path[-1]))
            return 1
        return self._recursive(reading, path, inside)

    # -- copyRootObjectRecursive -------------------------------------------------------

    def _recursive(
        self, reading: ROOTFile, path: list[str], dest: list[str], named: str = ""
    ) -> int:
        status = 0
        listing = _directory(reading, path) is not None
        for key in _newest(reading, path):
            inner = [*path, key.name] if listing else path
            if key.classname in DIRECTORIES:
                status += self._subdirectory(reading, inner, [*dest, key.name])
                continue
            if key.classname in TREES:
                self._clear(dest, key.name)  # as rootmv has it: the tree's own name
            elif named:
                self._clear(dest, named)
            self._write(reading, inner, dest, named or key.name)
        return status

    def _subdirectory(self, reading: ROOTFile, path: list[str], dest: list[str]) -> int:
        if not self._exists(dest):
            self._mkdir(dest)
        if not self._is_directory(dest):
            _warn(OVERWRITE_ERROR.format(dest[-1], dest[-1]))
            return 1
        return self._recursive(reading, path, dest)

    def _clear(self, dest: list[str], name: str) -> None:
        """What is in the way of a replacing copy, every cycle of it, taken away."""
        if self._exists([*dest, name]):
            self._here(dest).delete(name)
            self.written.pop(("/".join(dest), name), None)

    def _write(self, reading: ROOTFile, path: list[str], dest: list[str], name: str) -> None:
        here = self._here(dest)
        copy(reading, here, {"/".join(path): name})
        parent = _directory(reading, path[:-1])
        assert parent is not None
        self.written[("/".join(dest), name)] = parent.key(path[-1]).classname

    # -- the destination ---------------------------------------------------------------

    def _here(self, dest: list[str]) -> WritableDirectory:
        assert self.dest is not None
        return self.dest.mkdir("/".join(dest)) if dest else self.dest

    def _kind(self, dest: list[str]) -> str | None:
        assert self.dest is not None
        here: WritableDirectory = self.dest
        for part in dest[:-1]:
            if here.classname_of(part) not in DIRECTORIES:
                return None
            here = here.mkdir(part)
        return here.classname_of(dest[-1]) or self.written.get(("/".join(dest[:-1]), dest[-1]))

    def _exists(self, dest: list[str]) -> bool:
        return not dest or self._kind(dest) is not None

    def _is_directory(self, dest: list[str]) -> bool:
        return not dest or self._kind(dest) in DIRECTORIES

    def _mkdir(self, dest: list[str]) -> None:
        self._here(dest)

    # -- deleteRootObject --------------------------------------------------------------

    def _delete(self, name: str, source: WritableFile, path: list[str]) -> int:
        if self.interactive:
            question = (f"remove '{'/'.join(path)}' from '{name}' ? (y/n) : " if path
                        else f"remove '{name}' ? (y/n) : ")  # fmt: skip
            sys.stdout.write(question)
            sys.stdout.flush()
            if sys.stdin.readline().strip().lower() != "y":
                return 0
        if not path:
            source.close()
            os.remove(name)
            return 0
        here = source.mkdir("/".join(path[:-1])) if path[:-1] else source
        if not here.delete(path[-1]):
            _warn(DELETE_ERROR.format(path[-1]))
            return 1
        return 0


def _destination(name: str, compress: int | None, recreate: bool) -> WritableFile:
    """DEST, as ``rootmv`` opens it: made anew, or updated - and made if not there."""
    if not recreate and os.path.exists(name):
        return update(name)
    if compress is None:
        return create(name)
    from ..merging.files import resolve_compression

    algorithm, level = resolve_compression(compress, 0)
    return create(name, compression=algorithm, level=level)


def _newest(reading: ROOTFile, path: list[str]) -> list[Key]:
    """``getKeyList``, each name once at its newest cycle: a directory's keys, or the one."""
    directory = _directory(reading, path)
    if directory is None:
        parent = _directory(reading, path[:-1])
        assert parent is not None
        return [parent.key(path[-1])]
    newest: dict[str, Key] = {}
    for key in directory.all_keys():
        if key.name not in newest or newest[key.name].cycle < key.cycle:
            newest[key.name] = key
    return list(newest.values())
