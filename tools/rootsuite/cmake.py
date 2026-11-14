"""The tutorial harness's CMake, grown to read ROOT's test directories.

:class:`tools.tutorials.cmake.Interpreter` runs one flat file. ROOT's tests
are spread over hundreds of ``CMakeLists.txt`` reached through
``add_subdirectory``, and they lean on things the tutorials' file does not:
``function``/``macro`` definitions (``ROOTTEST_IO_TREE_TEST``,
``HIST_ADD_GTEST``) with their own scopes and ``cmake_parse_arguments``,
``PARENT_SCOPE``, ``return()``, ``get_filename_component``, ``if(EXISTS)``.
:class:`Directory` adds those, one interpreter per directory sharing a
:class:`~tools.rootsuite.model.Registry`, and leaves ROOT's own test macros
to the handlers :mod:`tools.rootsuite.rootmacros` installs.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from tools.tutorials.cmake import CLOSE, OPEN, Command, Interpreter, Token, commands

from .model import Registry

__all__ = ["Directory", "Definition", "Handler", "HANDLERS", "parse_arguments", "blocks"]

#: A command implemented here or in ROOT's macros: the interpreter and its expanded arguments.
Handler = Callable[["Directory", list[str]], None]

#: Commands only this interpreter knows, filled by the modules that implement them.
HANDLERS: dict[str, Handler] = {}


@dataclass
class If:
    """An ``if`` block, shaped as :meth:`Interpreter.chosen` reads it."""

    branches: list[tuple[tuple[Token, ...], list[Node]]] = field(default_factory=list)
    otherwise: list[Node] = field(default_factory=list)


@dataclass
class Foreach:
    """A ``foreach`` block, shaped as :meth:`Interpreter.loop` reads it."""

    args: tuple[Token, ...]
    body: list[Node] = field(default_factory=list)


@dataclass
class Definition:
    """A ``function`` or ``macro``: its parameters and its unexecuted body."""

    name: str
    params: tuple[str, ...]
    body: list[Node]
    macro: bool = False


Node = Command | If | Foreach | Definition

#: The commands that end a block, and the commands each block opens with.
_ENDERS = {"endif", "else", "elseif", "endforeach", "endfunction", "endmacro", "endwhile"}
_OPENERS = {"function": "endfunction", "macro": "endmacro", "while": "endwhile"}


def blocks(stream: Iterator[Command], until: set[str]) -> tuple[list[Node], Command | None]:
    """Commands grouped into blocks up to one of ``until``; ``while`` bodies are dropped."""
    body: list[Node] = []
    for command in stream:
        if command.name in until:
            return body, command
        node = _node(command, stream)
        if node is not None:
            body.append(node)
    return body, None


def _node(command: Command, stream: Iterator[Command]) -> Node | None:
    if command.name == "if":
        return _if_block(command, stream)
    if command.name == "foreach":
        return Foreach(command.args, blocks(stream, {"endforeach"})[0])
    if command.name in _OPENERS:
        inner, _ = blocks(stream, {_OPENERS[command.name]})
        return _definition(command, inner)
    return None if command.name in _ENDERS else command


def _definition(command: Command, body: list[Node]) -> Definition | None:
    names = [token.text for token in command.args if token not in (OPEN, CLOSE)]
    if command.name == "while" or not names:
        return None  # no loop in ROOT's test lists decides which test exists
    return Definition(names[0].lower(), tuple(names[1:]), body, command.name == "macro")


def _if_block(opening: Command, stream: Iterator[Command]) -> If:
    block = If()
    condition = opening.args
    while True:
        body, ender = blocks(stream, {"elseif", "else", "endif"})
        block.branches.append((condition, body))
        if ender is None or ender.name == "endif":
            return block
        if ender.name == "else":
            block.otherwise, _ = blocks(stream, {"endif"})
            return block
        condition = ender.args


class Return(Exception):
    """``return()``: ends the function, or the file, it is in."""


def _listed(arg: str) -> list[str]:
    """One argument as a CMake list expands: split at each unescaped ``;``, empties dropped."""
    if ";" not in arg:
        return [arg]
    return [part for part in re.split(r"(?<!\\);", arg) if part]


def parse_arguments(
    args: Iterable[str], options: Iterable[str], singles: Iterable[str], multis: Iterable[str]
) -> dict[str, list[str]]:
    """``cmake_parse_arguments``: each keyword's values, the rest under ``UNPARSED``.

    An option present maps to ``["TRUE"]``; a one-value keyword keeps its
    first value only, the others falling to ``UNPARSED`` as CMake's do. ROOT's
    macros parse ``${ARGN}``, which splits a quoted ``"a;b"`` into two values.
    """
    flags, one, many = set(options), set(singles), set(multis)
    parsed: dict[str, list[str]] = {"UNPARSED": []}
    current = "UNPARSED"
    for arg in (part for given in args for part in _listed(given)):
        if arg in flags:
            parsed[arg], current = ["TRUE"], "UNPARSED"
        elif arg in one or arg in many:
            current = arg
            parsed[current] = []
        else:
            parsed.setdefault(current, []).append(arg)
            if current in one:
                current = "UNPARSED"
    return parsed


#: ``if`` tests on one operand that the tutorials' condition reader does not know.
_UNARY: dict[str, Callable[[Directory, str], bool]] = {
    "EXISTS": lambda self, text: self.path(text).exists(),
    "IS_DIRECTORY": lambda self, text: self.path(text).is_dir(),
    "IS_ABSOLUTE": lambda self, text: text.startswith("/"),
    "COMMAND": lambda self, text: text.lower() in self.registry.functions,
    "TEST": lambda self, text: any(test.ctest == text for test in self.registry.tests),
    "POLICY": lambda self, text: True,
}

#: The variable ``include_directories`` appends to: a directory property, which a variable
#: models, since each subdirectory starts from a copy of its parent's.
INCLUDES = "<include_directories>"
#: The variable ``add_definitions`` appends to: the directory's ``COMPILE_DEFINITIONS``.
DEFINITIONS = "<compile_definitions>"

#: ``$<TARGET_FILE:x>``: the executable a generator expression names, by its target name.
_TARGET_FILE = re.compile(r"\$<TARGET_FILE:([^>]+)>")
_TARGET_DIR = re.compile(r"\$<TARGET_FILE_DIR:[^>]+>")


class Directory(Interpreter):
    """One directory's ``CMakeLists.txt``, with its own variables and the shared registry.

    ``relative`` is its path below ROOT's source tree, which ROOT's macros
    turn into test names; ``CMAKE_CURRENT_BINARY_DIR`` is ``<build>/<relative>``.
    """

    def __init__(
        self,
        registry: Registry,
        relative: str,
        variables: Mapping[str, str],
        module: Callable[[str], bool] = lambda name: True,
    ) -> None:
        root = Path(registry.root)
        source = root / relative if relative else root
        super().__init__(source, variables, module=module)
        self.registry = registry
        self.relative = relative
        self.targets = registry.targets
        self.parent: dict[str, str] | None = None
        #: Files ``configure_file(... COPYONLY)`` copies into the build directory under their
        #: own names, which the directory's tests find there.
        self.configured: list[str] = []
        build = f"<build>/{relative}" if relative else "<build>"
        self.variables.update(
            CMAKE_CURRENT_SOURCE_DIR=str(source),
            CMAKE_CURRENT_LIST_DIR=str(source),
            CMAKE_CURRENT_BINARY_DIR=build,
        )

    def path(self, text: str) -> Path:
        """A path as CMake reads one: relative ones from the current source directory."""
        return Path(text) if text.startswith("/") else self.source / text

    def relative_path(self, text: str) -> str:
        """A path below ROOT's source tree, as tests record their files."""
        path = self.path(text)
        try:
            return path.resolve().relative_to(Path(self.registry.root).resolve()).as_posix()
        except ValueError:
            return text

    def inside(self, text: str) -> str | None:
        """A source or build directory as a path below ROOT's tree, or ``None`` if it is not one.

        The build tree mirrors the source tree, so ``<build>/x`` is ``x``.
        """
        if text.startswith("<build>/"):
            return text[len("<build>/"):]
        found = self.relative_path(text) if "<" not in text else "/"
        return None if found.startswith("/") else found

    def includes(self) -> list[str]:
        """``include_directories`` in effect here, below ROOT's tree, this directory first."""
        found = (self.inside(entry) for entry in self.values(INCLUDES))
        return list(dict.fromkeys(entry for entry in (self.relative, *found) if entry))

    def expand(self, text: str) -> str:
        text = super().expand(text)
        return _TARGET_DIR.sub(self.variables["CMAKE_CURRENT_BINARY_DIR"],
                               _TARGET_FILE.sub(r"\1", text))

    def arguments(self, args: Iterable[Token]) -> list[str]:
        """Arguments as a command sees them, an unquoted ``with\\ space`` as ``with space``."""
        return super().arguments(_identity(token) for token in args)

    def loop(self, block: Foreach) -> None:  # type: ignore[override]
        """``foreach``, ``RANGE stop`` and ``RANGE start stop [step]`` counted as CMake counts."""
        args = self.arguments(block.args)
        if len(args) >= 3 and args[1] == "RANGE":
            items = [str(value) for value in _range(args[2:])]
            block = Foreach(tuple(Token(word) for word in (args[0], *items)), block.body)
        super().loop(block)  # type: ignore[arg-type]

    def run_file(self, path: Path) -> Directory:
        try:
            super().run_file(path)
        except Return:
            pass
        return self

    def run(self, source: str) -> Directory:
        body, _ = blocks(iter(commands(source)), set())
        self.execute(body)
        return self

    def execute(self, body: Iterable[Node]) -> None:  # type: ignore[override]
        for node in body:
            if isinstance(node, Definition):
                self.registry.functions[node.name] = node
            elif isinstance(node, If):
                self.execute(self.chosen(node))  # type: ignore[arg-type]
            elif isinstance(node, Foreach):
                self.loop(node)  # type: ignore[arg-type]
            else:
                self.command(node)

    def command(self, node: Command) -> None:
        if node.name == "return":
            raise Return()
        handler = HANDLERS.get(node.name)
        if handler is not None:
            handler(self, self.arguments(node.args))
            return
        definition = self.registry.functions.get(node.name)
        if definition is not None:
            self.call(definition, self.arguments(node.args))
        else:
            super().execute([node])

    def builtin(self, name: str, args: list[str]) -> None:
        """The tutorials' own implementation of a command, on already expanded arguments."""
        super().execute([Command(name, tuple(Token(arg, True) for arg in args))])

    def condition(self, args: Iterable[Token]) -> list[Token]:
        tokens = super().condition(args)
        out: list[Token] = []
        at = 0
        while at < len(tokens):
            token = tokens[at]
            test = _UNARY.get(token.text) if not token.quoted else None
            if test is not None and at + 1 < len(tokens):
                out.append(Token("TRUE" if test(self, tokens[at + 1].text) else "FALSE"))
                at += 2
                continue
            out.append(token)
            at += 1
        return out

    # functions, macros and scopes

    def call(self, definition: Definition, args: list[str]) -> None:
        """A function runs in a copy of the caller's variables; a macro in the caller's own."""
        bound = _bindings(definition.params, args)
        if definition.macro:
            self.variables.update(bound)
            self.execute(definition.body)
            return
        saved, parent = self.variables, self.parent
        self.parent, self.variables = saved, {**saved, **bound}
        try:
            self.execute(definition.body)
        except Return:
            pass
        finally:
            self.variables, self.parent = saved, parent

    def set_parent(self, name: str, values: list[str]) -> None:
        """``set(... PARENT_SCOPE)``: the caller's variable, or the parent directory's."""
        if self.parent is None:
            return
        if values:
            self.parent[name] = ";".join(values)
        else:
            self.parent.pop(name, None)

    def subdirectory(self, relative: str) -> Directory | None:
        """``add_subdirectory``: a child interpreter over a copy of this one's variables.

        Each directory is read once: ``add_subdirectory(${unset})`` names the
        top of the tree again, which CMake would refuse and this ignores.
        """
        if not relative or relative in self.registry.visited:
            return None
        self.registry.visited.add(relative)
        child = Directory(self.registry, relative, self.variables, self.module)
        child.parent = self.variables
        listing = child.source / "CMakeLists.txt"
        return child.run_file(listing) if listing.is_file() else None


#: CMake's identity escapes in an unquoted argument: ``\ ``, ``\(``... stand for the character.
_IDENTITY_ESCAPE = re.compile(r"\\([^A-Za-z0-9;$@])")


def _range(words: list[str]) -> range:
    """``RANGE stop`` is 0 to stop; ``RANGE start stop [step]`` start to stop, both included."""
    numbers = [int(word) for word in words[:3] if word.lstrip("-").isdigit()]
    if len(numbers) == 1:
        return range(numbers[0] + 1)
    if len(numbers) < 2:
        return range(0)
    step = numbers[2] if len(numbers) > 2 and numbers[2] else 1
    return range(numbers[0], numbers[1] + 1, step)


def _identity(token: Token) -> Token:
    if token.quoted or "\\" not in token.text:
        return token
    return Token(_IDENTITY_ESCAPE.sub(r"\1", token.text), False)


def _bindings(params: tuple[str, ...], args: list[str]) -> dict[str, str]:
    bound = {f"ARGV{index}": value for index, value in enumerate(args)}
    bound.update(zip(params, args, strict=False))
    bound.update(ARGC=str(len(args)), ARGV=";".join(args), ARGN=";".join(args[len(params):]))
    return bound


# --- CMake's own commands, beyond the tutorials' ----------------------------


def _set(self: Directory, args: list[str]) -> None:
    if "PARENT_SCOPE" not in args:
        self.builtin("set", args)
        return
    values = [value for value in args[1:] if value != "PARENT_SCOPE"]
    self.set_parent(args[0], values)


def _configure_file(self: Directory, args: list[str]) -> None:
    """``configure_file(in out COPYONLY)``: a file the build directory has a copy of.

    Only a copy keeping its name is followed - into the directory itself, or
    to the same name in it; one rewritten (``@VAR@``) or renamed is not.
    """
    if len(args) < 2 or "COPYONLY" not in args:
        return
    source, target = Path(args[0]).name, args[1]
    named = Path(target).name if target not in (".", "") and not target.endswith("/") else source
    if named == source:
        self.configured.append(self.relative_path(args[0]))


def _include_directories(self: Directory, args: list[str]) -> None:
    """``include_directories``: where this directory's (and its children's) headers are."""
    given = [arg for arg in args if arg not in ("AFTER", "BEFORE", "SYSTEM")]
    self.variables[INCLUDES] = ";".join([*self.values(INCLUDES), *given])


def _add_definitions(self: Directory, args: list[str]) -> None:
    """``add_definitions(-DX ...)``: ``X`` defined for this directory and those below it."""
    given = [arg[2:] for arg in args if arg.startswith("-D") and len(arg) > 2]
    self.variables[DEFINITIONS] = ";".join([*self.values(DEFINITIONS), *given])


def _option(self: Directory, args: list[str]) -> None:
    if len(args) >= 3 and args[0] not in self.variables:
        self.variables[args[0]] = args[2]


def _add_subdirectory(self: Directory, args: list[str]) -> None:
    if not args:
        return
    target = self.path(args[0])
    try:
        relative = target.resolve().relative_to(Path(self.registry.root).resolve()).as_posix()
    except ValueError:
        return
    self.subdirectory(relative)


def _parse_argv(self: Directory, args: list[str]) -> list[str]:
    """``cmake_parse_arguments(PARSE_ARGV N ...)`` as the plain form, on the function's ARGV."""
    if args[:1] != ["PARSE_ARGV"] or len(args) < 6:
        return args
    start = int(args[1]) if args[1].isdigit() else 0
    return [args[2], *args[3:6], *self.values("ARGV")[start:]]


def _reset(self: Directory, prefix: str, lists: list[list[str]]) -> None:
    """Every keyword's variable unset, as CMake does, then each option FALSE until given."""
    for word in (word for group in lists for word in group):
        self.variables.pop(f"{prefix}_{word}", None)
    self.variables.update(dict.fromkeys((f"{prefix}_{word}" for word in lists[0]), "FALSE"))


def _cmake_parse_arguments(self: Directory, args: list[str]) -> None:
    """Sets ``<prefix>_<KEYWORD>`` for each keyword given, options to TRUE or FALSE."""
    args = _parse_argv(self, args)
    if len(args) < 4:
        return
    prefix, lists = args[0], [[w for w in args[index].split(";") if w] for index in (1, 2, 3)]
    _reset(self, prefix, lists)
    parsed = parse_arguments(args[4:], *lists)
    unparsed = parsed.pop("UNPARSED")
    self.variables.update({f"{prefix}_{word}": ";".join(v) for word, v in parsed.items()})
    self.variables[f"{prefix}_UNPARSED_ARGUMENTS"] = ";".join(unparsed)


def _name_we(text: str) -> str:
    return Path(text).name.split(".")[0]


#: ``get_filename_component``'s modes, on a path.
_COMPONENTS: dict[str, Callable[[Directory, str], str]] = {
    "NAME": lambda self, text: Path(text).name,
    "NAME_WE": lambda self, text: _name_we(text),
    "NAME_WLE": lambda self, text: Path(text).stem,
    "EXT": lambda self, text: Path(text).name[len(_name_we(text)):],
    "DIRECTORY": lambda self, text: str(Path(text).parent) if "/" in text else "",
    "PATH": lambda self, text: str(Path(text).parent) if "/" in text else "",
    "ABSOLUTE": lambda self, text: str(self.path(text)),
    "REALPATH": lambda self, text: str(self.path(text)),
}


def _get_filename_component(self: Directory, args: list[str]) -> None:
    if len(args) >= 3 and args[2] in _COMPONENTS:
        self.variables[args[0]] = _COMPONENTS[args[2]](self, args[1])


def _string(self: Directory, args: list[str]) -> None:
    action = _STRING_ACTIONS.get(args[0]) if args else None
    if action is None:
        self.builtin("string", args)
    else:
        action(self, args[1:])


def _regex(self: Directory, args: list[str]) -> None:
    if len(args) < 4 or args[0] != "REPLACE":
        return
    subject = "".join(args[4:])
    replacement = re.sub(r"\\(\d)", r"\\g<\1>", args[2])
    try:
        self.variables[args[3]] = re.sub(args[1], replacement, subject)
    except re.error:
        self.variables[args[3]] = subject


def _case(upper: bool) -> Callable[[Directory, list[str]], None]:
    def change(self: Directory, args: list[str]) -> None:
        if len(args) >= 2:
            self.variables[args[1]] = args[0].upper() if upper else args[0].lower()

    return change


def _append(self: Directory, args: list[str]) -> None:
    if args:
        self.variables[args[0]] = self.variables.get(args[0], "") + "".join(args[1:])


def _identifier(self: Directory, args: list[str]) -> None:
    if len(args) >= 2:
        self.variables[args[1]] = re.sub(r"[^A-Za-z0-9_]", "_", args[0])


_STRING_ACTIONS: dict[str, Callable[[Directory, list[str]], None]] = {
    "REGEX": _regex,
    "TOUPPER": _case(True),
    "TOLOWER": _case(False),
    "APPEND": _append,
    "MAKE_C_IDENTIFIER": _identifier,
}


def _get(self: Directory, args: list[str]) -> None:
    items = self.values(args[1])
    indices = [int(i) for i in args[2:-1] if i.lstrip("-").isdigit()]
    picked = [items[i] for i in indices if -len(items) <= i < len(items)]
    self.variables[args[-1]] = ";".join(picked)


_LIST_ACTIONS: dict[str, Callable[[list[str], list[str]], list[str]]] = {
    "SORT": lambda items, rest: sorted(items),
    "REMOVE_DUPLICATES": lambda items, rest: list(dict.fromkeys(items)),
    "PREPEND": lambda items, rest: [*rest, *items],
    "REVERSE": lambda items, rest: items[::-1],
}


def _list(self: Directory, args: list[str]) -> None:
    if args[:1] == ["GET"] and len(args) >= 4:
        _get(self, args)
        return
    action = _LIST_ACTIONS.get(args[0]) if len(args) >= 2 else None
    if action is None:
        self.builtin("list", args)
    else:
        self.variables[args[1]] = ";".join(action(self.values(args[1]), args[2:]))


HANDLERS.update(
    {
        "set": _set,
        "include_directories": _include_directories,
        "configure_file": _configure_file,
        "add_definitions": _add_definitions,
        "option": _option,
        "add_subdirectory": _add_subdirectory,
        "cmake_parse_arguments": _cmake_parse_arguments,
        "get_filename_component": _get_filename_component,
        "string": _string,
        "list": _list,
    }
)
