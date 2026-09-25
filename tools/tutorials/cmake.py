"""Just enough CMake to read ``tutorials/CMakeLists.txt`` the way ROOT's build does.

ROOT's own list of which tutorials CI runs - and with which exit code, which
labels, which Python packages, and after which other tutorial - is not a data
file but a CMake program: vetoes are appended under ``if(NOT roofit)``,
dependencies are ``set(<test>-depends ...)`` inside a ``foreach``, and the
tests themselves are the ``ROOT_ADD_TEST`` calls a loop over a globbed and
filtered list makes. Copying its answers into a table here would be wrong the
day the file changes; so the file is run instead, by an interpreter of the
commands it uses (``set``, ``list``, ``file(GLOB)``, ``string(REPLACE)``,
``if``/``foreach``...) with the build's options supplied from the ROOT that is
installed. Commands it does not know are ignored, since none of them decide
which tutorial runs how.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NamedTuple, Union

__all__ = [
    "Token",
    "Command",
    "TestSpec",
    "Interpreter",
    "tokens",
    "commands",
    "glob_regex",
]


class Token(NamedTuple):
    """One argument as written: its text, and whether it was quoted."""

    text: str
    quoted: bool = False


#: A nested parenthesis inside a command's arguments, kept for ``if``'s grouping.
OPEN, CLOSE = Token("(", False), Token(")", False)

#: The lexical pieces of CMake's language, tried in this order at each position.
_LEXEME = re.compile(
    r"""
    (?P<comment>\#\[(?P<ceq>=*)\[.*?\](?P=ceq)\]|\#[^\n]*)
    |(?P<bracket>\[(?P<beq>=*)\[.*?\](?P=beq)\])
    |(?P<quoted>"(?:\\.|[^"\\])*")
    |(?P<open>\()
    |(?P<close>\))
    |(?P<space>\s+)
    |(?P<word>(?:\\.|[^\s()#"\\])+)
    """,
    re.X | re.S,
)

#: The escapes a quoted argument may hold, and what each stands for.
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", ";": "\\;"}


def _unescape(text: str) -> str:
    return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), m.group(1)), text)


def tokens(source: str) -> Iterator[Token]:
    """The source's arguments and parentheses, comments and whitespace dropped."""
    for match in _LEXEME.finditer(source):
        kind = match.lastgroup
        if kind in ("comment", "space", None):
            continue
        text = match.group(kind)
        if kind == "quoted":
            yield Token(_unescape(text[1:-1]), True)
        elif kind == "bracket":
            yield Token(text[len(match.group("beq")) + 2 : -len(match.group("beq")) - 2], True)
        elif kind in ("open", "close"):
            yield OPEN if kind == "open" else CLOSE
        else:
            yield Token(text)


class Command(NamedTuple):
    """One command invocation: its name in lower case, and its arguments."""

    name: str
    args: tuple[Token, ...]


def commands(source: str) -> list[Command]:
    """The commands of a CMake file, in order, each with its raw arguments."""
    found: list[Command] = []
    stream = iter(list(tokens(source)))
    for token in stream:
        if token is OPEN or token is CLOSE:
            continue
        opening = next(stream, None)
        if opening is not OPEN:
            continue
        found.append(Command(token.text.lower(), tuple(_arguments(stream))))
    return found


def _arguments(stream: Iterator[Token]) -> Iterator[Token]:
    """The tokens up to the parenthesis closing a command, nested ones kept."""
    depth = 0
    for token in stream:
        if token is CLOSE and depth == 0:
            return
        if token is OPEN:
            depth += 1
        elif token is CLOSE:
            depth -= 1
        yield token


# --- blocks ----------------------------------------------------------------


@dataclass
class _If:
    branches: list[tuple[tuple[Token, ...], list[Node]]] = field(default_factory=list)
    otherwise: list[Node] = field(default_factory=list)


@dataclass
class _Foreach:
    args: tuple[Token, ...]
    body: list[Node] = field(default_factory=list)


Node = Union[Command, _If, _Foreach]

#: The commands that end the block they belong to.
_ENDERS = {"endif", "else", "elseif", "endforeach"}


def _blocks(stream: Iterator[Command], until: set[str]) -> tuple[list[Node], Command | None]:
    """Commands grouped into ``if`` and ``foreach`` blocks, up to one of ``until``."""
    body: list[Node] = []
    for command in stream:
        if command.name in until:
            return body, command
        if command.name == "if":
            body.append(_if_block(command, stream))
        elif command.name == "foreach":
            inner, _ = _blocks(stream, {"endforeach"})
            body.append(_Foreach(command.args, inner))
        elif command.name not in _ENDERS:
            body.append(command)
    return body, None


def _if_block(opening: Command, stream: Iterator[Command]) -> _If:
    block = _If()
    condition = opening.args
    while True:
        body, ender = _blocks(stream, {"elseif", "else", "endif"})
        block.branches.append((condition, body))
        if ender is None or ender.name == "endif":
            return block
        if ender.name == "else":
            block.otherwise, _ = _blocks(stream, {"endif"})
            return block
        condition = ender.args


# --- globbing --------------------------------------------------------------


def glob_regex(pattern: str) -> str:
    """A CMake glob as a regular expression: ``*`` and ``?`` stay within a directory."""
    out = []
    for piece in re.split(r"(\[[^\]]*\]|\*|\?)", pattern):
        if piece == "*":
            out.append("[^/]*")
        elif piece == "?":
            out.append("[^/]")
        elif piece.startswith("[") and piece.endswith("]") and len(piece) > 2:
            out.append("[^" + piece[2:] if piece[1] == "!" else piece)
        else:
            out.append(re.escape(piece))
    return "".join(out)


def _recursive_regex(pattern: str) -> str:
    """``GLOB_RECURSE``'s pattern: its file part matched in any directory below its own."""
    head, _, name = pattern.rpartition("/")
    prefix = glob_regex(head) + "/" if head else ""
    return prefix + "(?:.*/)?" + glob_regex(name)


# --- tests -----------------------------------------------------------------


@dataclass(frozen=True)
class TestSpec:
    """One ``ROOT_ADD_TEST``: a name, its command line, and how it is judged."""

    name: str
    command: tuple[str, ...]
    passrc: int = 0
    failregex: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    depends: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    python_deps: tuple[str, ...] = ()
    timeout: float | None = None
    will_fail: bool = False


#: ``ROOT_ADD_TEST``'s keywords, each starting the section of arguments after it.
_TEST_KEYWORDS = {
    "COMMAND", "PASSRC", "FAILREGEX", "PASSREGEX", "LABELS", "DEPENDS", "ENVIRONMENT",
    "TIMEOUT", "PYTHON_DEPS", "WILLFAIL", "WORKING_DIR", "OUTREF", "ERRREF", "PRECMD",
    "POSTCMD", "OUTCNV", "FIXTURES_SETUP", "FIXTURES_CLEANUP", "FIXTURES_REQUIRED",
    "RESOURCE_LOCK", "COPY_TO_BUILDDIR", "RUN_SERIAL", "PROPERTIES", "DIFFCMD", "CHECKOUT",
}  # fmt: skip


def _sections(args: Sequence[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"": []}
    current = ""
    for arg in args:
        if arg in _TEST_KEYWORDS:
            current = arg
            sections.setdefault(current, [])
        else:
            sections[current].append(arg)
    return sections


def _test(args: Sequence[str]) -> TestSpec:
    parts = _sections(args[1:])
    timeout = parts.get("TIMEOUT") or [""]
    return TestSpec(
        name=args[0],
        command=tuple(parts.get("COMMAND", ())),
        passrc=int((parts.get("PASSRC") or ["0"])[0]),
        failregex=tuple(parts.get("FAILREGEX", ())),
        labels=tuple(parts.get("LABELS", ())),
        depends=tuple(parts.get("DEPENDS", ())),
        environment=tuple(parts.get("ENVIRONMENT", ())),
        python_deps=tuple(parts.get("PYTHON_DEPS", ())),
        timeout=float(timeout[0]) if timeout[0] else None,
        will_fail="WILLFAIL" in parts,
    )


# --- conditions ------------------------------------------------------------

#: The constants CMake's ``if`` reads as false; a ``-NOTFOUND`` suffix is also false.
FALSE = {"", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND"}
#: The constants it reads as true without looking up a variable.
TRUE = {"1", "ON", "YES", "TRUE", "Y"}


def truthy(value: str) -> bool:
    """A value as CMake's ``if(<constant>)`` reads it."""
    upper = value.upper()
    return not (upper in FALSE or upper.endswith("-NOTFOUND"))


def _number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text))


#: The binary tests ``if`` knows, on the two operands' values.
_BINARY: dict[str, Callable[[str, str], bool]] = {
    "STREQUAL": lambda a, b: a == b,
    "EQUAL": lambda a, b: _number(a) == _number(b),
    "LESS": lambda a, b: _number(a) < _number(b),
    "GREATER": lambda a, b: _number(a) > _number(b),
    "LESS_EQUAL": lambda a, b: _number(a) <= _number(b),
    "GREATER_EQUAL": lambda a, b: _number(a) >= _number(b),
    "VERSION_LESS": lambda a, b: _version(a) < _version(b),
    "VERSION_GREATER": lambda a, b: _version(a) > _version(b),
    "VERSION_EQUAL": lambda a, b: _version(a) == _version(b),
    "VERSION_GREATER_EQUAL": lambda a, b: _version(a) >= _version(b),
    "VERSION_LESS_EQUAL": lambda a, b: _version(a) <= _version(b),
    "MATCHES": lambda a, b: re.search(b, a) is not None,
}


class _Condition:
    """An ``if`` condition read by recursive descent: OR over AND over NOT over tests."""

    def __init__(self, args: Sequence[Token], interpreter: Interpreter) -> None:
        self.args = list(args)
        self.at = 0
        self.vars = interpreter.variables
        self.targets = interpreter.targets

    def peek(self) -> Token | None:
        return self.args[self.at] if self.at < len(self.args) else None

    def take(self) -> Token:
        token = self.args[self.at]
        self.at += 1
        return token

    def keyword(self, word: str) -> bool:
        token = self.peek()
        if token is not None and not token.quoted and token.text == word:
            self.at += 1
            return True
        return False

    def either(self) -> bool:
        value = self.both()
        while self.keyword("OR"):
            value = self.both() or value
        return value

    def both(self) -> bool:
        value = self.negated()
        while self.keyword("AND"):
            value = self.negated() and value
        return value

    def negated(self) -> bool:
        if self.keyword("NOT"):
            return not self.negated()
        return self.test()

    def test(self) -> bool:
        if self.peek() is OPEN:
            self.take()
            value = self.either()
            self.keyword(")")
            return value
        for word, unary in (("TARGET", self.targets.__contains__), ("DEFINED", self.defined)):
            if self.keyword(word):
                return unary(self.take().text)
        return self.compared(self.take())

    def defined(self, name: str) -> bool:
        return name in self.vars

    def compared(self, left: Token) -> bool:
        operator = self.peek()
        if operator is not None and operator.text in _BINARY:
            self.take()
            return _BINARY[operator.text](self.value(left), self.value(self.take()))
        if operator is not None and operator.text == "IN_LIST":
            self.take()
            listed = self.vars.get(self.take().text, "")
            return self.value(left) in listed.split(";")
        return self.bare(left)

    def value(self, token: Token) -> str:
        if not token.quoted and token.text in self.vars:
            return self.vars[token.text]
        return token.text

    def bare(self, token: Token) -> bool:
        upper = token.text.upper()
        if (
            token.quoted
            or upper in TRUE
            or upper in FALSE
            or _number(token.text) == _number(token.text)
        ):
            return truthy(token.text)
        return token.text in self.vars and truthy(self.vars[token.text])


# --- the interpreter -------------------------------------------------------

#: ``${name}``, innermost first, and ``$ENV{name}``.
_VARIABLE = re.compile(r"\$\{([^${}]*)\}")
_ENVIRONMENT = re.compile(r"\$ENV\{([^}]*)\}")


class Interpreter:
    """Runs a CMake file's commands, keeping its variables and the tests it adds.

    ``variables`` are the build's options and CMake's own (``roofit=ON``,
    ``CMAKE_CURRENT_SOURCE_DIR=...``); ``targets`` the names ``if(TARGET x)``
    finds; ``module`` answers ``ROOT_FIND_PYTHON_MODULE`` - whether a Python
    package imports. Globs are matched against the files under ``source``.
    """

    def __init__(
        self,
        source: Path,
        variables: Mapping[str, str] = (),  # type: ignore[assignment]
        targets: Iterable[str] = (),
        module: Callable[[str], bool] = lambda name: False,
        environ: Mapping[str, str] = (),  # type: ignore[assignment]
    ) -> None:
        self.source = Path(source)
        self.variables: dict[str, str] = {"CMAKE_CURRENT_SOURCE_DIR": str(self.source)}
        self.variables.update(dict(variables))
        self.targets = set(targets)
        self.module = module
        self.environ = dict(environ)
        self.tests: list[TestSpec] = []
        self._files: list[str] | None = None

    # expansion

    def expand(self, text: str) -> str:
        text = _ENVIRONMENT.sub(lambda m: self.environ.get(m.group(1), ""), text)
        while True:
            expanded = _VARIABLE.sub(lambda m: self.variables.get(m.group(1), ""), text)
            if expanded == text:
                return expanded
            text = expanded

    def arguments(self, args: Iterable[Token]) -> list[str]:
        """Arguments as a command sees them: unquoted ones split into their list items."""
        out: list[str] = []
        for token in args:
            if token is OPEN or token is CLOSE:
                out.append(token.text)
            elif token.quoted:
                out.append(self.expand(token.text))
            else:
                out.extend(item for item in self.expand(token.text).split(";") if item)
        return out

    def condition(self, args: Iterable[Token]) -> list[Token]:
        """``if``'s arguments expanded, with their quoting kept for its lookups."""
        out: list[Token] = []
        for token in args:
            if token is OPEN or token is CLOSE:
                out.append(token)
            else:
                out.append(Token(self.expand(token.text), token.quoted))
        return out

    def values(self, name: str) -> list[str]:
        return [item for item in self.variables.get(name, "").split(";") if item]

    # running

    def run_file(self, path: Path) -> Interpreter:
        return self.run(Path(path).read_text(encoding="utf-8", errors="replace"))

    def run(self, source: str) -> Interpreter:
        body, _ = _blocks(iter(commands(source)), set())
        self.execute(body)
        return self

    def execute(self, body: Iterable[Node]) -> None:
        for node in body:
            if isinstance(node, _If):
                self.execute(self.chosen(node))
            elif isinstance(node, _Foreach):
                self.loop(node)
            else:
                handler = _HANDLERS.get(node.name)
                if handler is not None:
                    handler(self, self.arguments(node.args))

    def chosen(self, block: _If) -> list[Node]:
        for condition, body in block.branches:
            if _Condition(self.condition(condition), self).either():
                return body
        return block.otherwise

    def loop(self, block: _Foreach) -> None:
        args = self.arguments(block.args)
        if not args:
            return
        name, items = args[0], args[1:]
        if items[:2] == ["IN", "LISTS"]:
            items = [value for listed in items[2:] for value in self.values(listed)]
        elif items[:2] == ["IN", "ITEMS"]:
            items = items[2:]
        for item in items:
            self.variables[name] = item
            self.execute(block.body)

    # files

    def files(self) -> list[str]:
        """Every file under the source directory, relative, with ``/`` separators."""
        if self._files is None:
            found = []
            for root, dirs, names in os.walk(self.source):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                here = Path(root).relative_to(self.source).as_posix()
                found.extend(name if here == "." else f"{here}/{name}" for name in names)
            self._files = sorted(found)
        return self._files

    def glob(self, patterns: Iterable[str], recurse: bool = False) -> list[str]:
        """The files ``file(GLOB)`` - or ``GLOB_RECURSE`` - finds, relative to the source."""
        prefix = str(self.source) + "/"
        found: dict[str, None] = {}
        for pattern in patterns:
            relative = pattern[len(prefix) :] if pattern.startswith(prefix) else pattern
            regex = re.compile(_recursive_regex(relative) if recurse else glob_regex(relative))
            found.update((name, None) for name in self.files() if regex.fullmatch(name))
        return sorted(found)


# --- the commands ----------------------------------------------------------


def _set(self: Interpreter, args: list[str]) -> None:
    if not args:
        return
    name, values = args[0], args[1:]
    if "CACHE" in values:
        values = values[: values.index("CACHE")]
    values = [value for value in values if value != "PARENT_SCOPE"]
    if values:
        self.variables[name] = ";".join(values)
    else:
        self.variables.pop(name, None)


def _unset(self: Interpreter, args: list[str]) -> None:
    for name in args[:1]:
        self.variables.pop(name, None)


def _list(self: Interpreter, args: list[str]) -> None:
    if len(args) < 2:
        return
    action = _LIST_ACTIONS.get(args[0])
    if action is not None:
        action(self, args[1], args[2:])


def _append(self: Interpreter, name: str, rest: list[str]) -> None:
    items = self.values(name) + rest
    self.variables[name] = ";".join(items)


def _remove_item(self: Interpreter, name: str, rest: list[str]) -> None:
    gone = set(rest)
    self.variables[name] = ";".join(item for item in self.values(name) if item not in gone)


def _length(self: Interpreter, name: str, rest: list[str]) -> None:
    if rest:
        self.variables[rest[0]] = str(len(self.values(name)))


def _find(self: Interpreter, name: str, rest: list[str]) -> None:
    if len(rest) >= 2:
        items = self.values(name)
        self.variables[rest[1]] = str(items.index(rest[0]) if rest[0] in items else -1)


def _filter(self: Interpreter, name: str, rest: list[str]) -> None:
    if len(rest) < 3 or rest[1] != "REGEX":
        return
    keep = rest[0] == "INCLUDE"
    regex = re.compile(rest[2])
    items = [item for item in self.values(name) if (regex.search(item) is not None) == keep]
    self.variables[name] = ";".join(items)


_LIST_ACTIONS: dict[str, Callable[[Interpreter, str, list[str]], None]] = {
    "APPEND": _append,
    "REMOVE_ITEM": _remove_item,
    "LENGTH": _length,
    "FIND": _find,
    "FILTER": _filter,
}


def _file(self: Interpreter, args: list[str]) -> None:
    if len(args) < 2 or args[0] not in ("GLOB", "GLOB_RECURSE"):
        return
    name, rest = args[1], args[2:]
    if rest[:1] == ["RELATIVE"]:
        rest = rest[2:]
    rest = [item for item in rest if item not in ("LIST_DIRECTORIES", "true", "false")]
    self.variables[name] = ";".join(self.glob(rest, recurse=args[0] == "GLOB_RECURSE"))


def _string(self: Interpreter, args: list[str]) -> None:
    if len(args) >= 4 and args[0] == "REPLACE":
        match, replacement, name = args[1], args[2], args[3]
        self.variables[name] = "".join(args[4:]).replace(match, replacement)


def _add_test(self: Interpreter, args: list[str]) -> None:
    if args:
        self.tests.append(_test(args))


def _python_module(self: Interpreter, args: list[str]) -> None:
    if args:
        found = self.module(args[0])
        self.variables[f"ROOT_{args[0].upper()}_FOUND"] = "TRUE" if found else "FALSE"


def _execute_process(self: Interpreter, args: list[str]) -> None:
    """Nothing is executed: a ``RESULT_VARIABLE`` says it failed, as it would without it."""
    if "RESULT_VARIABLE" in args[:-1]:
        self.variables[args[args.index("RESULT_VARIABLE") + 1]] = "1"


def _cmake_path(self: Interpreter, args: list[str]) -> None:
    if len(args) >= 4 and args[0] == "CONVERT":
        self.variables[args[-1]] = args[1]


_HANDLERS: dict[str, Callable[[Interpreter, list[str]], None]] = {
    "set": _set,
    "unset": _unset,
    "list": _list,
    "file": _file,
    "string": _string,
    "root_add_test": _add_test,
    "root_find_python_module": _python_module,
    "execute_process": _execute_process,
    "cmake_path": _cmake_path,
}
