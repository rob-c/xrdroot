"""What ROOT's command-line tools share: their option parser, their messages, their sources.

``rootls``, ``rootrm``, ``rootmkdir``, ``rootcp`` and ``hadd`` are C++
programs built on one option parser (ROOT's ``optparse.hxx``), one way of
saying what went wrong - ``Error in <rootls>: ...`` - and one way of naming
what is inside a file, ``file.root:dir/pattern``. Their xrdroot subcommands
are to be told from them only by name, so they take the command line raw
(:class:`RawParser`) and read it as ROOT's parser does (:class:`Options`):
short flags in groups (``-1c``), long ones with ``=``, ``--`` ending them,
and ROOT's own words for every mistake.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["RawParser", "raw_parser", "Flag", "Options", "OptionError", "Log"]


class RawParser(argparse.ArgumentParser):
    """A subcommand's parser that keeps the command line as it was given, ``--`` and all.

    ROOT's tools read their own command lines, and argparse would both
    refuse flags it was not told of and drop a ``--`` that ends them; this
    hands every word to the subcommand, as ``args.argv``, untouched.
    """

    def parse_known_args(  # type: ignore[override]
        self, args: Sequence[str] | None = None, namespace: Any = None
    ) -> tuple[argparse.Namespace, list[str]]:
        parsed = namespace if namespace is not None else argparse.Namespace()
        parsed.argv = list(sys.argv[1:] if args is None else args)
        return parsed, []


def raw_parser(subparsers: Any, name: str, help: str) -> argparse.ArgumentParser:
    """The subcommand ``name``, its command line left for ROOT's parser to read."""
    parser: argparse.ArgumentParser = subparsers.add_parser(name, help=help, add_help=False)
    parser.__class__ = RawParser
    return parser


@dataclass(frozen=True)
class Flag:
    """One flag: its spellings (``-l``, ``--longListing``), and whether it takes a value."""

    names: tuple[str, ...]
    takes_value: bool = False
    repeatable: bool = False

    @property
    def canonical(self) -> str:
        return self.names[0].lstrip("-")


class OptionError(Exception):
    """A command line ROOT's parser would not take, in ROOT's words."""


@dataclass
class Options:
    """ROOT's ``RCmdLineOpts``: the flags a tool knows, and what one command line set.

    A single-dash flag of more than one letter (``hadd``'s ``-dbg``) makes
    single-dash words whole names; otherwise ``-1c`` is ``-1`` and ``-c``.
    A flag's value is the next word, if that does not start with ``-``, or
    follows ``=`` in a long one.
    """

    flags: Sequence[Flag]
    seen: dict[str, list[str]] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)

    @property
    def grouped(self) -> bool:
        return not any(
            name.startswith("-") and not name.startswith("--") and len(name) > 2
            for flag in self.flags
            for name in flag.names
        )

    def parse(self, argv: Sequence[str]) -> Options:
        """Read ``argv``; raise :class:`OptionError` at the first word ROOT would refuse."""
        words = list(argv)
        positional = False
        at = 0
        while at < len(words):
            word = words[at]
            at += 1
            if word == "--":
                positional = True
            elif positional or not word.startswith("-"):
                self.args.append(word)
            else:
                at = self._flag(word, words, at)
        return self

    def _flag(self, word: str, words: list[str], at: int) -> int:
        """One word of flags; where the next word to read is."""
        nxt = words[at] if at < len(words) and not words[at].startswith("-") else None
        text = word[1:]
        if text.startswith("-"):
            return self._long(word, text[1:], nxt, at)
        grouped = self.grouped and len(text) > 1
        for letter in text[:-1] if grouped else "":
            self._one(word, f"-{letter}", None, False)
        last = text[-1] if grouped else text
        self._one(word, f"-{last}", nxt, False)
        return self._after(at, nxt, last)

    def _long(self, word: str, text: str, nxt: str | None, at: int) -> int:
        """``--name`` or ``--name=value``; where the next word to read is."""
        name, equals, value = text.partition("=")
        if equals:
            self._one(word, f"--{name}", value, True)
            return at
        self._one(word, f"--{name}", nxt, False)
        return self._after(at, nxt, name)

    def _after(self, at: int, nxt: str | None, name: str) -> int:
        """Past the next word if the flag took it as its value."""
        return at + 1 if nxt is not None and self._took(name) else at

    def _find(self, spelled: str) -> Flag | None:
        bare = spelled.lstrip("-")
        return next((flag for flag in self.flags
                     if bare in (name.lstrip("-") for name in flag.names)), None)  # fmt: skip

    def _took(self, name: str) -> bool:
        found = self._find(name)
        return found is not None and found.takes_value

    def _one(self, word: str, spelled: str, value: str | None, explicit: bool) -> None:
        found = self._find(spelled)
        if found is None:
            raise OptionError(f"Unknown flag: {word}")
        self._once(found, spelled)
        if found.takes_value:
            if not value:
                raise OptionError(f"Missing argument for flag {spelled}")
            self.seen.setdefault(found.canonical, []).append(value)
            return
        if explicit and value:
            raise OptionError(f"Flag {spelled} does not expect an argument")
        self.seen.setdefault(found.canonical, []).append("")

    def _once(self, found: Flag, spelled: str) -> None:
        """Refuse a flag given again, unless it may be."""
        before = self.seen.get(found.canonical)
        if before is not None and not found.repeatable:
            extra = f" with the value: {before[-1]}" if found.takes_value else ""
            raise OptionError(f"Flag {spelled} appeared more than once{extra}")

    def has(self, name: str) -> bool:
        return name in self.seen

    def count(self, name: str) -> int:
        return len(self.seen.get(name, []))

    def value(self, name: str) -> str | None:
        found = self.seen.get(name)
        return found[-1] if found else None


@dataclass
class Log:
    """What a tool says on standard error, as ROOT's ``Err()`` and ``Info()`` say it."""

    tool: str
    verbosity: int = 1

    def error(self, message: str) -> None:
        sys.stderr.write(f"Error in <{self.tool}>: {message}\n")

    def warning(self, message: str) -> None:
        sys.stderr.write(f"Warning in <{self.tool}>: {message}\n")

    def info(self, level: int, message: str) -> None:
        if self.verbosity >= level:
            sys.stderr.write(f"Info in <{self.tool}>: {message}\n")
