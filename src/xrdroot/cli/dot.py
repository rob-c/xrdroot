"""ROOT's prompt commands - ``.ls``, ``.cd``, ``.x`` - turned into the Python they stand for.

    >>> translate(".cd dir1")
    "gROOT.cd('dir1')"
    >>> translate(".x fill.py(1000)")
    "gROOT.macro('fill.py', 1000)"
    >>> translate("h = gROOT['h']")
    "h = gROOT['h']"

A line is a command only when it starts - in its first column - with a dot
and one of the names below; anything else is Python and passes untouched,
including a ``.method()`` continuing a chained call on its own line, which is
indented or is not one of these names. The shell runs every line it reads
through :func:`translate`, and IPython every cell through :func:`transform`.
"""

from __future__ import annotations

import re
from collections.abc import Callable

__all__ = ["translate", "transform"]

#: A command line: a dot in the first column, a name, and whatever follows it.
COMMAND = re.compile(r"^\.(?P<name>[A-Za-z?]+)(?:\s+(?P<rest>.*?))?\s*$")

#: ``macro.py(1, "a")``: a macro's path and the arguments ``.x`` hands it.
CALL = re.compile(r"^(?P<path>[^(]+?)\s*\((?P<args>.*)\)$")


def _argument(rest: str) -> str:
    """A command's one argument as a Python string literal, quotes it came in or not."""
    text = rest.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    return repr(text)


def _ls(rest: str) -> str:
    return f"print(gROOT.ls({_argument(rest)}))" if rest else "print(gROOT.ls())"


def _cd(rest: str) -> str:
    return f"gROOT.cd({_argument(rest)})" if rest else "gROOT.cd()"


def _pwd(rest: str) -> str:
    return "print(gROOT.pwd())"


def _macro(rest: str) -> str:
    if not rest:
        return "print('.x needs a macro to run: .x macro.py, or .x macro.py(arguments)')"
    call = CALL.match(rest.strip())
    if call is None:
        return f"gROOT.macro({_argument(rest)})"
    arguments = call["args"].strip()
    tail = f", {arguments}" if arguments else ""
    return f"gROOT.macro({_argument(call['path'])}{tail})"


def _quit(rest: str) -> str:
    return "exit()"


def _help(rest: str) -> str:
    return "print(gROOT.help())"


#: Every command, and the Python it becomes from what follows its name.
COMMANDS: dict[str, Callable[[str], str]] = {
    "ls": _ls,
    "cd": _cd,
    "pwd": _pwd,
    "x": _macro,
    "q": _quit,
    "qqq": _quit,
    "exit": _quit,
    "help": _help,
    "h": _help,
    "?": _help,
}


def translate(line: str) -> str:
    """``line`` as Python: a prompt command translated, anything else as it was."""
    found = COMMAND.match(line)
    if found is None:
        return line
    name = found["name"]
    command = COMMANDS.get(name)
    if command is None:
        return f"print({f'.{name} is not a command here; .help lists the ones there are'!r})"
    return command(found["rest"] or "")


def transform(lines: list[str]) -> list[str]:
    """An IPython cell's lines with each prompt command translated: an input transformer."""
    translated = []
    for line in lines:
        body = line.rstrip("\n")
        ending = line[len(body) :]
        translated.append(translate(body) + ending)
    return translated
