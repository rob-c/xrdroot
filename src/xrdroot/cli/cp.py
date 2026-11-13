"""``xrdroot cp``: ROOT's ``rootcp``, and ``TTree::CopyTree`` with ``--cut``.

    $ xrdroot cp in.root out.root                       # everything at the top
    $ xrdroot cp -r in.root out.root                    # directories and all
    $ xrdroot cp 'in.root:hists/*' in.root:events out.root
    $ xrdroot cp in.root:h out.root:renamed             # a new name
    $ xrdroot cp --replace in.root:h out.root:h         # over what is there
    $ xrdroot cp in.root:events skim.root:skims --cut 'nMuon >= 2' --columns nMuon,Muon_pt

Each SOURCE is a file and a pattern in it, read as :mod:`.sources` reads one.
DEST is a file - updated if it is there, made if not, or made anew with
``--recreate`` - and after its colon either a directory there, which the
copies go into, or a new name for the one thing copied. A directory is
copied only with ``-r``; an object already at the destination only with
``--replace``, which takes its every cycle away first. Each object goes
across as the record it was where it can, a tree as its baskets, the newest
cycle of each name; what is wrong is said as ``rootcp`` says it, with its
statuses - and, as ``rootcp`` does, a run that fails part-way deletes DEST
unless DEST was also a source.

Past ``rootcp``'s flags, ``--cut`` keeps only the entries of the trees
copied that pass it and ``--columns`` only the branches named, as
``TTree::CopyTree`` would, and ``--slow`` writes trees' entries afresh.
See :func:`xrdroot.copy`.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from ..errors import ROOTError
from ..merging import copy
from ..merging.files import resolve_compression
from ..writer import WritableDirectory, WritableFile, create
from ..wupdate import update
from .sources import DIRECTORIES, Matches, matching, missing_file, split_source
from .tooling import Flag, Log, OptionError, Options, raw_parser

__all__ = ["add_parser", "run", "USAGE"]

#: What ``rootcp`` says when its command line is wrong, on standard error.
USAGE = (
    "usage: rootcp [-h] [-c COMPRESS] [--recreate] [-r|--recursive] [--replace] "
    "[-v|--verbose] SOURCE [SOURCE ...] DEST\n"
)

#: The rest of ``rootcp -h``, after the usage line.
HELP = """
Copy objects from ROOT files into another

positional arguments:
  SOURCE                Source file(s)
  DEST                  Destination file

options:
  -h, --help            show this help message and exit
  -c, --compress COMPRESS
                        change the compression settings of the destination file (if not already
                        existing).
  --recreate            recreate the destination file.
  -r, --recursive       recurse inside directories
  --replace             replace object if already existing
  -v                    be verbose
  -vv                   be even more verbose

Note: If an object has been written to a file multiple times, rootcp will copy only the latest version of that object.

Source and destination files accept the syntax: `protocol://path/to/file.root:path/to/object*` to select specific
subobjects or directories in the file.

Examples:
- rootcp source.root dest.root
  Copy the latest version of each object in 'source.root' to 'dest.root'.

- rootcp source.root:hist* dest.root
  Copy all histograms whose names start with 'hist' from 'source.root' to 'dest.root'.

- rootcp source1.root:hist1 source2.root:hist2 dest.root
  Copy histograms 'hist1' from 'source1.root' and 'hist2' from 'source2.root' to 'dest.root'.

- rootcp --recreate source.root:hist dest.root
  Recreate 'dest.root' and copy the histogram named 'hist' from 'source.root' into it.

- rootcp -c 101 source.root:hist dest.root
  Change compression, if not existing, of 'dest.root' to ZLIB algorithm with compression level 1 and copy the histogram named 'hist' from 'source.root' into it.
  Meaning of the '-c' argument is given by 'compress = 100 * algorithm + level'.
  Other examples of usage:
    * -c 509 : ZSTD with compression level 9
    * -c 404 : LZ4 with compression level 4
    * -c 207 : LZMA with compression level 7
  For more information see https://root.cern.ch/doc/latest-stable/classTFile.html#ad0377adf2f3d88da1a1f77256a140d60
  and https://root.cern.ch/doc/latest-stable/structROOT_1_1RCompressionSetting.html
"""  # noqa: E501

#: ``rootcp``'s flags, then xrdroot's own: ``TTree::CopyTree``'s cut and columns.
FLAGS = (
    Flag(("-c", "--compress"), takes_value=True),
    Flag(("--recreate",)),
    Flag(("--replace",)),
    Flag(("-r", "--recursive")),
    Flag(("-h", "--help")),
    Flag(("-v",)),
    Flag(("-vv",)),
    Flag(("--cut",), takes_value=True),
    Flag(("--columns",), takes_value=True),
    Flag(("--slow",)),
)


def add_parser(subparsers: Any) -> None:
    raw_parser(subparsers, "cp", "copy objects between ROOT files, as rootcp does")


@dataclass
class _Destination:
    """Where the copies go: the file, the path after its colon, and whether that is a name."""

    filename: str
    path: str
    new_object: bool = False
    out: WritableFile | None = None


@dataclass
class _Copy:
    """One ``rootcp`` run: what it was told."""

    log: Log
    recursive: bool
    replace: bool
    cut: str | None
    columns: list[str] | None
    fast: bool
    same_file: bool = False
    sources: list[tuple[str, str]] = field(default_factory=list)
    given: list[str] = field(default_factory=list)
    #: What this run has copied, by directory and name: a tree's key is only
    #: written as the file closes, so the directory cannot say yet.
    written: dict[tuple[str, str], str] = field(default_factory=dict)


def _parsed(argv: list[str]) -> tuple[Options, int | None]:
    """The command line, or the status ``rootcp`` refuses it with."""
    try:
        options = Options(FLAGS).parse(argv)
    except OptionError as why:
        sys.stderr.write(f"{why}\n{USAGE}")
        return Options(FLAGS), 1
    if options.has("h"):
        sys.stderr.write(USAGE + HELP)
        return options, 0
    if len(options.args) < 2:
        sys.stderr.write(USAGE)
        return options, 1
    return options, None


def _job(options: Options) -> _Copy:
    """What the flags ask of the run."""
    log = Log("rootcp", 3 if options.has("vv") else 2 if options.has("v") else 1)
    columns = options.value("columns")
    return _Copy(log, options.has("r"), options.has("replace"), options.value("cut"),
                 columns.split(",") if columns else None, not options.has("slow"))  # fmt: skip


def run(args: argparse.Namespace) -> int:
    options, status = _parsed(args.argv)
    if status is not None:
        return status
    job = _job(options)
    try:
        dest = _Destination(*split_source(options.args[-1]))
        job.given = options.args[:-1]
        job.sources = [split_source(text) for text in job.given]
    except ValueError as why:
        job.log.error(str(why))
        return 1
    job.same_file = any(name == dest.filename for name, _ in job.sources)
    status = _opened(job, dest, options.has("recreate"), options.value("c"))
    return _copied_into(job, dest) if status is None else status


def _copied_into(job: _Copy, dest: _Destination) -> int:
    """The run's status, DEST open: and, as rootcp does, DEST gone if it failed, unless a source."""
    assert dest.out is not None
    with dest.out:
        failed = _checked(job, dest) or _copied(job, dest)
    if failed and not job.same_file:
        os.unlink(dest.filename)
    return 1 if failed else 0


def _opened(job: _Copy, dest: _Destination, recreate: bool, compress: str | None) -> int | None:
    """Open DEST as ``rootcp`` does - or the status it gives up with."""
    log = job.log
    if recreate and job.same_file:
        log.error("cannot recreate destination file if this is also a source file")
        return 1
    exists = os.path.exists(dest.filename)
    if compress is not None and not exists:  # sic: rootcp refuses -c for a new file, not an old
        log.error(f"can't change compression settings on existing file {dest.filename}")
        return 1
    try:
        if exists and not recreate:
            dest.out = update(dest.filename)
        else:
            dest.out = create(dest.filename, **_compression(compress, job.sources))
    except (OSError, ValueError, ROOTError):
        folder = os.path.dirname(dest.filename)
        if folder and not os.path.isdir(folder):
            where = os.path.join(os.getcwd(), dest.filename)
            sys.stderr.write(f"SysError in <TFile::TFile>: file {where} can not be opened "
                             "No such file or directory\n")  # fmt: skip
        log.error(f"File {dest.filename}does not exist.")
        return 1
    return None


def _compression(compress: str | None, sources: list[tuple[str, str]]) -> dict[str, Any]:
    """A new DEST's compression: ``-c``'s, or the first source's as it is filed."""
    from ..file import open_root

    codes = 0
    if compress is None and missing_file(sources[0][0]) is None:
        try:
            with open_root(sources[0][0]) as first:
                codes = first.compression
        except (OSError, ValueError, ROOTError):
            codes = 0
    algorithm, level = resolve_compression(None if compress is None else int(compress), codes)
    return {"compression": algorithm, "level": level}


def _checked(job: _Copy, dest: _Destination) -> bool:
    """Whether DEST's path is wrong for what is asked: taken, or a name for several things."""
    assert dest.out is not None
    if not dest.path:
        return False
    kind = _kind(dest.out, dest.path)
    if kind is not None and kind not in DIRECTORIES:
        if not job.replace:
            job.log.error(f'destination path "{dest.filename}:{dest.path}" already exists '
                          f"(as an object of type {kind}). Use the --replace flag to overwrite "
                          "it.")  # fmt: skip
            return True
        kind = None
    dest.new_object = kind is None
    if dest.new_object and len(job.sources) > 1:
        job.log.error(f'multiple sources were specified, but destination path "{dest.filename}:'
                      f'{dest.path}" is not a directory.')  # fmt: skip
        return True
    return False


def _kind(out: WritableDirectory, path: str) -> str | None:
    """The class of what ``path`` holds in the file being written, or ``None``."""
    head, _, name = path.rpartition("/")
    here = out
    for part in head.split("/") if head else []:
        if here.classname_of(part) not in DIRECTORIES:
            return None
        here = here.mkdir(part)
    return here.classname_of(name)


def _copied(job: _Copy, dest: _Destination) -> bool:
    """Every source copied in turn; whether one stopped the run."""
    for text in job.given:
        found = matching(text, job.recursive)
        try:
            if _source(job, dest, found):
                return True
        finally:
            found.close()
    return False


def _source(job: _Copy, dest: _Destination, found: Matches) -> bool:
    """One source copied; whether it stopped the run."""
    if _wrong(job, dest, found):
        return True
    chosen = [*found.leaves]
    for index in found.dirs:
        chosen += found.nodes[index].children if index == 0 else [index]
    for index in chosen:
        _node(job, dest, found, index)
    return False


def _wrong(job: _Copy, dest: _Destination, found: Matches) -> bool:
    """Whether a source matched wrongly - nothing, or several things for one new name."""
    for line in found.said:
        sys.stderr.write(line + "\n")
    for message in found.errors:
        job.log.error(message)
    if found.errors:
        return True
    if dest.new_object and len(found.leaves) + len(found.dirs) > 1:
        job.log.error(f'multiple sources were specified but destination path "{dest.filename}:'
                      f'{dest.path}" is not a directory.')  # fmt: skip
        return True
    return False


def _split(path: str) -> tuple[str, str]:
    head, _, base = path.rpartition("/")
    return head, base


def _target(dest: _Destination, source: str, name: str) -> tuple[str, str, str]:
    """Where ``CopyNode`` puts a node: its folder, its name there, and the whole path."""
    if dest.new_object or not dest.path:
        target = dest.path or source
        return (*_split(target), target)
    return dest.path, name, f"{dest.path}/{name}"


def _blocked(
    job: _Copy, dest: _Destination, here: WritableDirectory, base: str, target: str
) -> bool:
    """Whether an object is in the way; one in the way of ``--replace`` is taken away."""
    kind = here.classname_of(base) or job.written.get((here.path, base))
    if kind is None:
        return False
    if kind not in DIRECTORIES and not job.replace:
        job.log.error(f"an object of type '{kind}' already exists at {dest.filename}:{target}. "
                      "Use the --replace flag to overwrite existing objects.")  # fmt: skip
        return True
    if job.replace:  # a directory too, which is then made again, as rootcp does
        here.delete(base)
    return False


def _node(job: _Copy, dest: _Destination, found: Matches, index: int) -> None:
    """One key copied, as ``rootcp``'s ``CopyNode`` copies it - directories key by key."""
    log = job.log
    node = found.nodes[index]
    source = found.path(index)
    folder, base, target = _target(dest, source, node.name)
    if job.same_file and found.filename == dest.filename and source == target:
        log.error(f"{found.filename}:{source}: source and destination cannot be the same")
        return
    log.info(2, f"cp {found.filename}:{source} -> {dest.filename}:{target}")
    assert dest.out is not None
    here: WritableDirectory = dest.out
    if folder:
        log.info(3, f"mkdir {folder}")
        here = dest.out.mkdir(folder)
    if _blocked(job, dest, here, base, target):
        return
    shown = "TDirectoryFile" if node.classname == "TDirectory" else node.classname
    log.info(3, f'read object "{source}" of type {shown}')
    if node.is_directory:
        _directory(job, dest, found, index, here)
        return
    try:
        copy(found.file, here, {source: base}, cut=job.cut, columns=job.columns, fast=job.fast)
    except (ROOTError, KeyError, ValueError, TypeError) as why:
        log.error(f'failed to read object "{source}". {why}')
        return
    job.written[(here.path, base)] = node.classname


def _directory(
    job: _Copy, dest: _Destination, found: Matches, index: int, here: WritableDirectory
) -> None:
    """A directory: with ``-r``, made at DEST and everything in it copied there."""
    node = found.nodes[index]
    source = found.path(index)
    if not job.recursive:
        job.log.warning(f"Directory '{source}' will not be copied. Use the -r option if you "
                        "need a recursive copy.")  # fmt: skip
        return
    here.mkdir(node.name)
    inner = _Destination(dest.filename, f"{dest.path}/{node.name}" if dest.path else node.name,
                         dest.new_object, dest.out)  # fmt: skip
    for child in node.children:
        _node(job, inner, found, child)
