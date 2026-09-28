"""TMVA's ``MsgLogger``: how every line TMVA prints begins.

A TMVA line is a source name padded to 25 characters, ``": "``, and the
text: ``Factory                  : Booking method: BDT``. Only a header - the
first line of something - names its source; the lines after it leave the
name blank, and a warning is signed ``<WARNING>`` instead. With colour off,
as a ``Reader`` built with ``"!Color"`` has it, a header is marked
``<HEADER>`` and every other kind but information by its kind's name. The
colour and silence are TMVA's global ``gConfig()``, which the last
``Factory`` or ``Reader`` made sets, and so they are one setting here too.

A fatal message is printed, then ``***> abort program execution``, and then
:class:`TMVAError` is raised, as TMVA throws.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from ..errors import ROOTError, UnsupportedFeatureError

__all__ = [
    "CONFIG",
    "DEBUG",
    "VERBOSE",
    "INFO",
    "WARNING",
    "ERROR",
    "FATAL",
    "HEADER",
    "Config",
    "Logger",
    "TMVAError",
    "color",
]

#: ``TMVA::EMsgType``.
DEBUG, VERBOSE, INFO, WARNING, ERROR, FATAL, SILENT, HEADER = range(1, 9)
#: Each kind's name, as a line without colour is marked with it.
NAMES = {
    DEBUG: "DEBUG",
    VERBOSE: "VERBOSE",
    INFO: "INFO",
    WARNING: "WARNING",
    ERROR: "ERROR",
    FATAL: "FATAL",
    SILENT: "SILENT",
    HEADER: "HEADER",
}
#: The escape each kind is coloured with, when colour is on.
COLOURS = {
    DEBUG: "\033[34m",
    ERROR: "\033[31m",
    FATAL: "\033[37;41;1m",
    SILENT: "\033[30m",
}
#: ``Tools::Color``'s escapes, by the names TMVA asks for them.
ESCAPES = {
    "bold": "\033[1m",
    "reset": "\033[0m",
    "lightblue": "\033[0;36m",
    "red": "\033[1;31m",
    "dgreen": "\033[32m",
    "lgreenb": "\033[1;32m",
    "yellow": "\033[1;33m",
    "white": "\033[1;37m",
    "blue": "\033[34m",
}
#: How wide a source name is printed.
SOURCE_WIDTH = 25


class TMVAError(ROOTError, RuntimeError):
    """What TMVA's ``kFATAL`` throws: the message it printed last."""


class TMVAUnsupported(TMVAError, UnsupportedFeatureError):
    """A ``kFATAL`` of xrdroot's own: something TMVA does that xrdroot does not."""


@dataclass
class Config:
    """``TMVA::gConfig()``: colour, silence and the progress bar, shared by all of TMVA."""

    use_color: bool = True
    silent: bool = False
    draw_progress_bar: bool = False
    workers: int = 1

    @staticmethod
    def Instance() -> Config:
        """``TMVA::Config::Instance()``: the one configuration."""
        return CONFIG

    def SetUseColor(self, on: bool) -> None:
        self.use_color = bool(on)

    def SetSilent(self, on: bool) -> None:
        self.silent = bool(on)

    def SetDrawProgressBar(self, on: bool) -> None:
        self.draw_progress_bar = bool(on)

    def SetNumWorkers(self, count: int) -> None:
        self.workers = int(count)

    def GetNumWorkers(self) -> int:
        return self.workers

    def GetNCpu(self) -> int:
        import os

        return os.cpu_count() or 1


#: The one configuration, as TMVA has one.
CONFIG = Config()


def color(name: str) -> str:
    """``gTools().Color(name)``: the escape, or nothing when colour is off."""
    return ESCAPES.get(name, "") if CONFIG.use_color else ""


def _source(kind: int, name: str) -> str:
    """The source part of a line: the name for a header, ``<WARNING>``, or nothing."""
    shown = name if kind == HEADER else "<WARNING>" if kind == WARNING else ""
    if len(shown) > SOURCE_WIDTH:
        shown = shown[: SOURCE_WIDTH - 3] + "..."
    return shown.ljust(SOURCE_WIDTH)


def _decorated(kind: int, line: str) -> str:
    """A formatted line as ``WriteMsg`` sends it, with colour or with its kind named."""
    if CONFIG.use_color:
        if kind in (HEADER, WARNING, INFO, VERBOSE):
            return line
        return f"{COLOURS[kind]}<{NAMES[kind]}>{line}\033[0m"
    if kind == INFO:
        return line
    return f"<{NAMES[kind]}> {line}"


class Logger:
    """One TMVA class's ``Log()``: its source name, and the least kind it prints."""

    def __init__(self, source: str, minimum: int = INFO) -> None:
        self.source = source
        self.minimum = minimum

    def send(self, kind: int, text: Any = "") -> None:
        """Print ``text``, a line for each line in it, as ``MsgLogger::Send`` does."""
        if (kind < self.minimum and kind != FATAL) or (CONFIG.silent and kind != FATAL):
            return
        head = _source(kind, self.source) + ": "
        for line in str(text).split("\n"):
            sys.stdout.write(_decorated(kind, head + line) + "\n")

    def header(self, text: Any = "") -> None:
        self.send(HEADER, text)

    def info(self, text: Any = "") -> None:
        self.send(INFO, text)

    def verbose(self, text: Any = "") -> None:
        self.send(VERBOSE, text)

    def warning(self, text: Any = "") -> None:
        self.send(WARNING, text)

    def error(self, text: Any = "") -> None:
        self.send(ERROR, text)

    def refuse(self, text: Any) -> TMVAError:
        """``fatal`` for what TMVA does and xrdroot does not: the error is also a refusal."""
        self.send(FATAL, text)
        sys.stdout.write("***> abort program execution\n")
        return TMVAUnsupported(str(text))

    def fatal(self, text: Any) -> TMVAError:
        """Print a fatal message and hand back the error to raise, as TMVA throws after it."""
        self.send(FATAL, text)
        sys.stdout.write("***> abort program execution\n")
        return TMVAError(str(text))
