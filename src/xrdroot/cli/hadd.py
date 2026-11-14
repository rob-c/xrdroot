"""``xrdroot hadd``: ROOT's ``hadd``, its command line, its messages and its statuses.

    $ xrdroot hadd all.root run1.root run2.root     # as hadd
    $ xrdroot hadd -f505 all.root run*.root         # over what is there, as zstd 5
    $ xrdroot hadd -k -v 3 all.root @runs.txt       # past bad inputs; the inputs listed in a file

The merging is :func:`xrdroot.merge`'s - ``xrdroot merge`` is the same with
xrdroot's own defaults - and everything around it is ``hadd``'s: its
flags, spelled every way ``hadd`` takes them (``-n0``, ``-n 0``, ``-n=0``),
its warnings and errors, ``Info in <hadd>:`` and the ``hadd Source file``
lines, and its exit statuses. With no ``-f`` setting the output is
compressed as ROOT's 101, as ``hadd`` compresses it. ``-j`` is taken and
said, but the inputs are merged in this one process, one after another;
``-n``, ``-cachesize``, ``-d`` and ``-experimental-io-features`` are
checked as ``hadd`` checks them and change nothing here.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any

from ..compression import spelled
from ..errors import ROOTError
from ..file import open_root
from ..merging import merge
from ..writer import create
from .tooling import Log, raw_parser

__all__ = ["add_parser", "run", "USAGE", "Request", "parse"]

#: What ``hadd`` says first when its command line will not do, on standard error.
USAGE = """
usage: hadd [-a] [-f] [-f[0-9]] [-fk] [-ff] [-k] [-L] [-Ltype] [-O] [-T]
            [-v V] [-j J] [-dbg] [-d D] [-n N] [-cachesize CACHESIZE]
            [-experimental-io-features EXPERIMENTAL_IO_FEATURES]
            TARGET SOURCES
"""

#: The rest of ``hadd -h``.
HELP = """
This program will add histograms, trees and other objects from a list
of ROOT files and write them to a target ROOT file. The target file is
newly created and must not exist, or if -f ("force") is given, must
not be one of the source files.

OPTIONS:
  -a                                   Append to the output
  -f                                   Force overwriting of output file
  -f[0-9]                              Gives the ability to specify the compression algorithm i and level j
                                       of the target file, by passing the number i*100 + j, e.g. -f505. The
                                       last digit (j) can be set from 0 = uncompressed to 9 = highly
                                       compressed. The first digit (i) is 1 for ZLIB, 2 for LZMA, 4 for LZ4
                                       and 5 for ZSTD. Recommended numbers are 101 (ZLIB), 207 (LZMA), 404
                                       (LZ4), 505 (ZSTD). The default value for this flag is 101
                                       (kDefaultZLIB). See ROOT::RCompressionSetting and TFile::TFile
                                       documentation for more details.
  -fk                                  Sets the target file to contain the baskets with the same compression
                                       as the input files (unless -O is specified). Compresses the meta data
                                       using the compression level specified in the first input or the
                                       compression setting after fk (for example 206 when using -fk206)
  -ff                                  The compression level use is the one specified in the first input
  -k                                   Skip corrupt or non-existent files, do not exit
  -L                                   Read the list of objects from FILE and either only merge or skip those
                                       objects depending on the value of "-Ltype". FILE must contain one
                                       object name per line, which cannot contain whitespaces or '/'. You can
                                       also pass TDirectory names, which apply to the entire directory
                                       content. Lines beginning with '#' are ignored. If this flag is passed,
                                       "-Ltype" MUST be passed as well.
  -Ltype                               Sets the type of operation performed on the objects listed in FILE
                                       given with the "-L" flag. "SkipListed" will skip all the listed
                                       objects; "OnlyListed" will only merge those objects. If this flag is
                                       passed, "-L" must be passed as well.
  -O                                   Re-optimize basket size when merging TTree
  -T                                   Do not merge Trees
  -v                                   Explicitly set the verbosity level: 0 request no output, 99 is the
                                       default
  -j                                   Parallelize the execution in 'J' processes. If the number of processes
                                       is not specified, use the system maximum.
  -dbg                                 Enable verbosity. If -j was specified, do not not delete partial files
                                       stored inside working directory.
  -d                                   Carry out the partial multiprocess execution in the specified
                                       directory
  -n                                   Open at most 'N' files at once (use 0 to request to use the system
                                       maximum)
  -cachesize                           Resize the prefetching cache use to speed up I/O operations (use 0 to
                                       disable)
  -experimental-io-features            Used with an argument provided, enables the corresponding experimental
                                       feature for output trees. See ROOT::Experimental::EIOFeatures
  TARGET                               Target file
  SOURCES                              Source files

If TARGET and SOURCES have different compression settings a slower
method is used. For options that takes a size as argument, a decimal
number of bytes is expected. If the number ends with a ``k'', ``m'',
``g'', etc., the number is multiplied by 1000 (1K), 1000000 (1MB),
1000000000 (1G), etc. If this prefix is followed by i, the number is
multiplied by the traditional 1024 (1KiB), 1048576 (1MiB), 1073741824
(1GiB), etc. The prefix can be optionally followed by B whose casing
is ignored, eg. 1k, 1K, 1Kb and 1KB are the same.
"""  # noqa: E501

#: ``hadd``'s compression when none is asked for: ROOT's compiled default, zlib at level 1.
DEFAULT_COMPRESSION = 101
#: The verbosity ``hadd`` has unless ``-v`` says otherwise.
VERBOSITY = 2
#: The flags that are on or off, by the letters after the dash.
SWITCHES = ("T", "a", "k", "O", "dbg")
#: The flags that take a value, by name, and the value a bare one stands for, if any.
VALUED = {"d": None, "j": 0, "Ltype": None, "L": None, "cachesize": None,
          "experimental-io-features": None, "n": None, "v": VERBOSITY}  # fmt: skip
#: The flags whose value is a whole number.
NUMBERS = ("j", "n", "v")
#: The experimental I/O features ``-experimental-io-features`` knows.
FEATURES = ("kGenerateOffsetMap",)
#: How many bytes of a file ROOT reads first, for its header.
HEADER = 300
#: What ``hadd`` says of ``-ff`` beside ``-f505``, after "cannot specify".
BOTH = "both -ff and -f[0-9]. Either use the first input compression or specify it."
#: What ``TFileMerger`` says when the target will not open.
MERGER = "Error in <TFileMerger::OutputFile>: cannot open the MERGER output file \n"
#: A cache size: a number, a decimal or binary prefix, and a ``B`` or not.
SIZE = re.compile(r"^(\d+)([kKmMgGtTpPeE]?)(i?)[bB]?$")


def _warning(message: str) -> str:
    return f"Warning in <hadd>: {message}\n"


class Refused(Exception):
    """A command line ``hadd`` will not run, with what it says and the status it exits with."""

    def __init__(self, message: str, status: int = 1) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Request:
    """What one ``hadd`` command line asks for."""

    switches: set[str] = field(default_factory=set)
    values: dict[str, Any] = field(default_factory=dict)
    positional: list[str] = field(default_factory=list)
    force: bool = False
    first: bool = False
    keep: bool = False
    compression: int | None = None
    help: bool = False
    warned: list[str] = field(default_factory=list)


def add_parser(subparsers: Any) -> None:
    raw_parser(subparsers, "hadd", "merge ROOT files into one, exactly as hadd does")


# -- the command line -----------------------------------------------------------------


class _Parser:
    """``hadd``'s own reading of its command line, word by word."""

    def __init__(self, argv: list[str]) -> None:
        self.words = [word for word in argv if word]
        self.at = 0
        self.request = Request()
        self.started = False  # in the one run of positional arguments, or past it
        self.flagged = False  # a flag seen since that run started
        self.after = False  # past a ``--``

    def parse(self) -> Request:
        while self.at < len(self.words):
            word = self.words[self.at]
            self.at += 1
            if not self.after and word.startswith("-") and len(word) > 1:
                self._flag(word)
            else:
                self._positional(word)
        return self.request

    def _positional(self, word: str) -> None:
        if self.started and self.flagged:
            first = self.request.positional[0]
            raise Refused(
                f"seen a positional argument '{word}' after some flags. Positional arguments "
                f"were already parsed at this point (from '{first}' onwards), and you can only "
                "have one sequence of them, so you cannot pass more. Please group your "
                "positional arguments all together so that hadd works as you expect.\n"
                f"Cmdline: hadd {' '.join(self.words)} "
            )
        self.started = True
        self.request.positional.append(word)

    def _flag(self, word: str) -> None:
        if word == "--":
            if self.started:
                raise Refused("found `--`, but we've already parsed (or are still parsing) a "
                              "sequence of positional arguments!")  # fmt: skip
            self.after = True
            return
        self.flagged = self.started
        name = word[1:]
        if name in SWITCHES:
            self._switch(name)
        elif name in ("help", "h", "-help"):
            self.request.help = True
        elif not self._valued(name) and not self._compression(name):
            self.request.warned.append(_warning(f"unknown flag: {word}"))

    def _switch(self, name: str) -> None:
        if name in self.request.switches:
            self.request.warned.append(_warning(f"duplicate flag: {name}"))
        self.request.switches.add(name)

    def _valued(self, name: str) -> bool:
        """A flag taking a value - ``-n 0``, ``-n0``, ``-n=0`` - read; whether it was one."""
        for flag in sorted(VALUED, key=len, reverse=True):
            if name.startswith(flag):
                self.request.values[flag] = self._value(flag, name[len(flag) :].lstrip("="))
                return True
        return False

    def _value(self, flag: str, attached: str) -> Any:
        if attached:
            return self._number(flag, attached) if flag in NUMBERS else attached
        if self.at >= len(self.words):
            raise Refused(f"expected argument after '-{flag}' flag.")
        given = self.words[self.at]
        numeric = given[:1].isdigit()
        if flag in NUMBERS and not numeric:
            if VALUED[flag] is None:
                raise Refused(f"the argument after '-{flag}' flag was not of the expected type.")
            return VALUED[flag]  # the word after it is read again, as what it is
        self.at += 1
        return self._number(flag, given) if flag in NUMBERS else given

    @staticmethod
    def _number(flag: str, text: str) -> int:
        if not text[:1].isdigit():
            raise Refused(f"the argument after '-{flag}' flag was not of the expected type.")
        if not text.isdigit():
            raise Refused(f"error parsing integer argument '{text}'")
        return int(text)

    def _compression(self, name: str) -> bool:
        """The ``-f`` family: ``-f``, ``-ff``, ``-fk``, ``-f505``, ``-fk206``."""
        if not name.startswith("f"):
            return False
        letters = name[1:]
        digits = letters.lstrip("fk")
        for letter in letters[: len(letters) - len(digits)]:
            self._letter(letter, bool(digits))
        self.request.force = True
        if digits:
            self._setting(digits, name)
        return True

    def _letter(self, letter: str, numbered: bool) -> None:
        request = self.request
        if letter == "f":
            if request.compression is not None:  # -f505 first: hadd says it otherwise
                raise Refused(f"[err] Cannot specify {BOTH}")
            if numbered:
                raise Refused(f"cannot specify {BOTH}")
            if request.first:
                request.warned.append(_warning("duplicate flag: -ff"))
            request.first = True
        else:
            if request.keep:
                request.warned.append(_warning("duplicate flag: -fk"))
            request.keep = True

    def _setting(self, digits: str, name: str) -> None:
        request = self.request
        if not digits.isdigit():
            raise Refused(f"invalid flag: {name}")
        if request.first:
            raise Refused(f"cannot specify {BOTH}")
        if request.compression is not None:
            raise Refused("cannot specify -f[0-9] multiple times!")
        value = int(digits)
        if 1 <= value <= 9:
            request.warned.append(_warning(f"interpreting {value} as 10{value}. This behavior "
                                           "is deprecated, please use the full compression "
                                           "settings."))  # fmt: skip
        elif value and not (100 <= value <= 509 and value // 10 % 10 == 0):
            raise Refused(f"{value} is not a supported compression settings.")
        request.compression = value


def parse(argv: list[str]) -> Request:
    """``hadd``'s command line read as ``hadd`` reads it; :class:`Refused` if it will not do."""
    request = _Parser(argv).parse()
    _checked(request)
    return request


def _checked(request: Request) -> None:
    """The values ``hadd`` checks once the whole line is read."""
    kind = request.values.get("Ltype")
    if kind is not None and kind not in ("SkipListed", "OnlyListed"):
        raise Refused(f"invalid argument for -Ltype: '{kind}'. Can only be 'SkipListed' or "
                      "'OnlyListed' (case matters).")  # fmt: skip
    wanted = request.values.get("experimental-io-features")
    for feature in wanted.split(",") if wanted else []:
        if feature not in FEATURES:
            request.warned.append(f"Error in <Set>: Could not locate {feature} in "
                                  "TBasket::EIOBits\n")  # fmt: skip
            request.warned.append(_warning(f"ignoring unknown feature request: {feature}"))
    size = request.values.get("cachesize")
    if size is not None and SIZE.match(size) is None:
        raise Refused(f"could not parse the cache size passed after -cachesize: '{size}'")


# -- the run --------------------------------------------------------------------------


def _say(stream: Any, text: str) -> None:
    """A line out, at once: ``hadd``'s two streams are read together, in order."""
    stream.write(text)
    stream.flush()


def run(args: argparse.Namespace) -> int:
    log = Log("hadd")
    try:
        request = parse(args.argv)
    except Refused as why:
        _refused(log, why)
        return why.status
    for line in request.warned:
        _say(sys.stderr, line)
    if request.help:
        _say(sys.stderr, USAGE + HELP)
        return 0
    try:
        return _Hadd(request, log).run()
    except Refused as why:
        _refused(log, why)
        return why.status


def _refused(log: Log, why: Refused) -> None:
    text = str(why)
    if text.startswith("[err]"):
        _say(sys.stderr, text + "\n")
    else:
        log.error(text)
        sys.stderr.flush()


class _Hadd:
    """One merge, as ``hadd`` goes about it and says it."""

    def __init__(self, request: Request, log: Log) -> None:
        self.request = request
        self.log = log
        self.verbosity = request.values.get("v", VERBOSITY)
        self.skip = "k" in request.switches

    def info(self, level: int, message: str) -> None:
        if self.verbosity >= level:
            _say(sys.stderr, f"Info in <hadd>: {message}\n")

    def run(self) -> int:
        request = self.request
        processes = self._processes()
        folder = request.values.get("d")
        if folder is not None and not os.path.isdir(folder):
            raise Refused(f"could not access the directory specified: {folder}.")
        if ("L" in request.values) != ("Ltype" in request.values):
            raise Refused("-L must always be passed along with -Ltype.")
        if not request.positional:
            raise Refused(f"missing output file.\n{USAGE}")
        if len(request.positional) < 2:
            raise Refused(f"missing input file.\n{USAGE}")
        target = request.positional[0]
        self.info(2, f"target file: {target}")
        if "cachesize" in request.values:
            self.info(2, f"Using cachesize={request.values['cachesize']}")
        inputs = self._inputs(target)
        return self._merge(target, inputs, processes)

    def _processes(self) -> int:
        if "j" not in self.request.values:
            return 1
        processes = self.request.values["j"] or os.cpu_count() or 1
        self.info(2, f"parallelizing  with {processes} processes.")
        return int(processes)

    # -- the inputs ---------------------------------------------------------------------

    def _inputs(self, target: str) -> list[str]:
        found: list[str] = []
        for given in self.request.positional[1:]:
            if given.startswith("@"):
                found.extend(self._listed(given[1:], target))
            else:
                found.extend(self._valid(given, target, f'could not validate argument "{given}" '
                                         "as input file "))  # fmt: skip
        if not found:
            raise Refused("could not find any valid input file ")
        return found

    def _listed(self, path: str, target: str) -> list[str]:
        try:
            with open(path, encoding="utf-8") as lines:
                names = [line.strip() for line in lines if line.strip()]
        except OSError as why:
            if not self.skip:
                raise Refused(f"could not open indirect file {path}") from why
            self.log.error(f"could not open indirect file {path}")
            return []
        found: list[str] = []
        for name in names:
            found.extend(self._valid(name, target, f'could not validate the file name "{name}" '
                                     f"within indirect file {path}"))  # fmt: skip
        return found

    def _valid(self, name: str, target: str, refusal: str) -> list[str]:
        """An input, if it can be read and is not the target - else ``hadd``'s complaint."""
        if "://" not in name and not os.access(name, os.R_OK):
            message = refusal
        elif os.path.abspath(name) == os.path.abspath(target):
            message = f"file {name} cannot be both the target and an input!"
        else:
            return [name]
        if not self.skip:
            raise Refused(message)
        self.log.error(message)
        sys.stderr.flush()
        return []

    # -- the merge ----------------------------------------------------------------------

    def _merge(self, target: str, inputs: list[str], processes: int) -> int:
        request = self.request
        setting = self._setting(inputs[0])
        what = "meta data" if request.keep and "O" not in request.switches else "all output"
        self.info(2, f"compression setting for {what}: {request.compression or setting}")
        self._writable(target)
        self._split(len(inputs), processes)
        readable, failed = self._opened(inputs)
        if failed:
            # TFileMerger opened the target before it took any input in, so ROOT leaves
            # it behind, holding nothing, when an input stops the merge.
            _empty(target, setting, "a" in request.switches)
            return self._failed(target, len(inputs), len(readable))
        skip, only = self._filter()
        if not request.keep and "O" not in request.switches:
            self._compare(readable, setting)
        return self._merged(target, readable, setting, skip, only, len(inputs))

    def _setting(self, first: str) -> int:
        """The output's compression: ``-f``'s, the first input's (``-ff``/``-fk``), or 101."""
        request = self.request
        if request.compression is not None:
            value = request.compression
            return 100 + value if 1 <= value <= 9 else value
        if request.first or request.keep:
            try:
                with open_root(first) as opened:
                    return int(opened.compression)
            except (OSError, ValueError, ROOTError):
                return DEFAULT_COMPRESSION
        return DEFAULT_COMPRESSION

    def _writable(self, target: str) -> None:
        """The target, as ``hadd`` would open it: new unless ``-f`` or ``-a``, and placeable."""
        request = self.request
        appending = "a" in request.switches
        folder = os.path.dirname(target)
        if folder and not os.path.isdir(folder):
            _say(sys.stderr, f"SysError in <TFile::TFile>: file {os.path.abspath(target)} can "
                             "not be opened No such file or directory\n" + MERGER)  # fmt: skip
            if appending:
                raise Refused(f"error opening target file for update :{target}.", 2)
            raise Refused(f"error opening target file (does {target} exist?).")
        if not appending and not request.force and os.path.exists(target):
            _say(sys.stderr, f"Error in <TFile::TFile>: file {os.path.abspath(target)} already "
                             "exists\n" + MERGER)  # fmt: skip
            raise Refused(f"error opening target file (does {target} exist?).\npass \"-f\" "
                          "argument to force re-creation of output file.")  # fmt: skip

    def _split(self, files: int, processes: int) -> None:
        if "j" not in self.request.values:
            return
        step = math.ceil(files / processes)
        if step < 3:
            self.info(2, "each process should handle at least 3 files for efficiency. Setting "
                      f"the number of processes to: {math.ceil(files / 3)}")  # fmt: skip

    def _opened(self, inputs: list[str]) -> tuple[list[tuple[str, int]], bool]:
        """``TFileMerger::AddFile`` for each input: said, opened, and its compression read.

        What was taken in comes back, and whether an input that would not open
        stopped the merge, as it does unless ``-k`` says to skip it.
        """
        readable = []
        for number, name in enumerate(inputs, 1):
            if self.verbosity >= 2:
                _say(sys.stdout, f"hadd Source file {number}: {name}\n")
            try:
                with open_root(name) as opened:
                    readable.append((name, int(opened.compression)))
            except (OSError, ValueError, ROOTError):
                _say(sys.stderr, _unreadable(name))
                if not self.skip:
                    self.log.error(f"exiting due to error in {name}")
                    return readable, True
                _say(sys.stderr, f"Warning in <hadd>: skipping file with error: {name}\n")
        return readable, False

    def _filter(self) -> tuple[list[str] | None, list[str] | None]:
        listed = self.request.values.get("L")
        if listed is None:
            return None, None
        try:
            with open(listed, encoding="utf-8") as lines:
                names = [line.split()[0] for line in lines if line.split() and line[0] != "#"]
        except OSError as why:
            raise Refused(f"error opening filter file '{listed}'") from why
        if names:
            self.info(2, f"added {len(names)} object from filter file '{listed}'")
        else:
            _say(sys.stderr, f"Warning in <hadd>: no objects were added from filter file "
                             f"'{listed}'\n")  # fmt: skip
        skip = self.request.values["Ltype"] == "SkipListed"
        return (names, None) if skip else (None, names)

    def _compare(self, readable: list[tuple[str, int]], setting: int) -> None:
        if any(codes != setting for _, codes in readable):
            _say(sys.stderr, "Warning in <hadd>: Sources and Target have different compression "
                             "settings\nhadd merging will be slower\n")  # fmt: skip

    def _merged(
        self,
        target: str,
        readable: list[tuple[str, int]],
        setting: int,
        skip: list[str] | None,
        only: list[str] | None,
        count: int,
    ) -> int:
        request = self.request
        switches = request.switches
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                merge(target, [name for name, _ in readable], compression=setting,
                      force=request.force, fast="O" not in switches, skip_keys=skip,
                      only_keys=only, append="a" in switches, keep_compression=request.keep,
                      skip_errors=self.skip, trees="T" not in switches)  # fmt: skip
        except (OSError, ValueError, KeyError, TypeError, ROOTError) as why:
            self.log.error(str(why))
            return self._failed(target, count, len(readable))
        if self.verbosity >= 2:
            _say(sys.stdout, f"hadd Target path: {target}:/\n")
        self.info(3, f"merged {count} ({len(readable)}) input (partial) files into {target}")
        return 0

    def _failed(self, target: str, count: int, merged: int) -> int:
        self.log.error(f"failure during the merge of {count} ({merged}) input (partial) files "
                       f"into {target}")  # fmt: skip
        sys.stderr.flush()
        return 1


def _empty(target: str, setting: int, appending: bool) -> None:
    """The target as ``hadd`` leaves it when an input stops the merge: made, and empty."""
    if appending and os.path.exists(target):
        return
    algorithm, level = spelled(setting)
    create(target, compression=algorithm, level=level).close()


def _unreadable(name: str) -> str:
    """What ROOT says of an input that is no ROOT file: shorter than a header, or not one."""
    said = ""
    size = os.path.getsize(name) if os.path.isfile(name) else HEADER
    if size < HEADER:
        said = (f"Error in <TFile::ReadBuffer>: error reading all requested bytes from file "
                f"{name}, got {size} of {HEADER}\nError in <TFile::Init>: {name} failed to read "
                "the file type data.\n")  # fmt: skip
    return said + f"Error in <TFileMerger::AddFile>: cannot open file {name}\n"
