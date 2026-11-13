"""Everything typed at the prompt so far, as one C++ translation unit that only grows.

Cling keeps one translation unit for a session: what each input declares is
there for every input after it. The translator reads whole units, so each
input is translated together with every input before it - which tells the
translator the type of each variable, function and class declared so far -
and of the Python that comes out only the statements that are new are run,
in a namespace that lives as long as the session. Which are new is told by
the translation's map from Python lines back to C++ ones: a statement from a
line that has run before is not run again, so ``TH1F h(...)`` makes one
histogram however many lines follow it.

Statements cling would run inside its wrapper function - ``h.Fill(1);``,
``for (...) {...}`` - go into a function of their own, which the next line
calls, as the initializer of a variable that is never used;
declarations go to the top as they are. An expression whose value is to be
printed becomes ``auto xrdroot_value_N = (expr);``, which tells both the
value and, through :func:`xrdroot.pycint.declaring.translate_declaring`, its
type; that line is never kept.

An input the translator refuses, that names something undeclared, or that
fails as it runs, is not kept either - cling rolls the transaction back -
and is reported as a :class:`Diagnostic` at the line and column typed.
"""

from __future__ import annotations

import ast
import re
import types
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ...pycxx.errors import CxxError as CintError
from ...pycxx.errors import Refusal
from ..declaring import translate_declaring
from .classify import Kind, kind
from .lexer import Piece

__all__ = ["Unit", "Chunk", "Diagnostic", "Origin", "FILE"]

#: What the session's translation unit is called; cling's is ``ROOT_prompt_N`` per input.
FILE = "ROOT_prompt"

#: The type spellings that are no class: printed as values, never through an address.
PLAIN = re.compile(
    r"^(?:(?:const|unsigned|signed|long|short)\s+)*"
    r"(?:bool|char|short|int|long|float|double|wchar_t|char16_t|char32_t)?$"
)

#: An expression naming an object: ``h``, ``*p``, ``a.b``, ``p->q``, ``v[0]``.
NAMED = re.compile(r"\*?[A-Za-z_][\w.:>\-\[\]]*")

#: An assignment, whose value is what it assigned to: ``a = b``, ``a += b``, not ``a == b``.
ASSIGNED = re.compile(r"(?<![=!<>])=(?!=)")

#: The string classes, which are printed by value and are never captured by address.
STRING_CLASSES = frozenset({"std::string", "TString"})


@dataclass(frozen=True)
class Origin:
    """Where a line of the unit was typed: the prompt's number, the line and column there."""

    prompt: int
    line: int
    column: int
    text: str = ""


class Diagnostic(Exception):
    """An input that did not run, as cling reports one: ``ROOT_prompt_3:1:5: error: ...``."""

    def __init__(self, where: str, why: str) -> None:
        self.where = where
        self.why = why
        #: Whether the translator refused the input, before any of it ran.
        self.translating = False
        super().__init__(f"{where}: error: {why}")


@dataclass
class Chunk:
    """The C++ lines one input adds to the unit, where each was typed, and its value line."""

    prompt: int
    lines: list[str] = field(default_factory=list)
    origins: list[Origin] = field(default_factory=list)
    #: How many of the lines are kept once the input has run: all but the value's.
    kept: int = 0
    #: The expression whose value is printed, and the name its value is caught in.
    expression: str | None = None
    capture: str = ""
    lvalue: bool = False
    #: Whether the value is caught only to be handed back - a declaration ending in ``;`` -
    #: and not printed.
    quiet: bool = False

    def add(self, text: str, origin: Origin) -> None:
        """Lines of C++ typed at ``origin``: the first at its column, the rest where their
        text starts, after their indentation."""
        for offset, line in enumerate(text.split("\n")):
            column = origin.column if offset == 0 else len(line) - len(line.lstrip()) + 1
            self.lines.append(line)
            self.origins.append(Origin(origin.prompt, origin.line + offset, column, origin.text))
        if self.expression is None:
            self.kept = len(self.lines)


@dataclass
class Outcome:
    """What running an input gave: the value and the type to print it as, if it had one."""

    value: Any = None
    spelled: str | None = None
    lvalue: bool = False
    printed: bool = False
    #: Whether the input had a value at all, printed or not: what ``root -q`` exits with.
    valued: bool = False


class Unit:
    """The session's translation unit, its namespace, and which of its statements have run."""

    def __init__(self, directory: Path | None = None) -> None:
        self.lines: list[str] = []
        self.origins: list[Origin] = []
        #: The statements that have run, by the C++ line they came from and their order on it.
        self.done: set[tuple[str, int, int]] = set()
        self.directory = directory or Path.cwd()
        self.namespace: dict[str, Any] = {"__name__": "__cint__", "__file__": FILE}
        #: Each run's source map, by the file name its Python was compiled under.
        self.maps: dict[str, dict[int, tuple[str, int]]] = {}
        self.serial = 0
        #: The type of each variable declared so far, as cling spells it, by C++ name.
        self.types: dict[str, str | None] = {}
        made, _ = translate_declaring("", FILE)
        exec(compile(made.python, "<ROOT_prompt header>", "exec"), self.namespace)

    def name(self, stem: str) -> str:
        """A name no other input used: for a wrapper function, or a value's capture."""
        self.serial += 1
        return f"xrdroot_{stem}_{self.serial}"

    # -- building an input's chunk -------------------------------------------------------

    def chunk(self, prompt: int, pieces: list[Piece], typed: str, quiet: bool = True) -> Chunk:
        """The lines ``pieces`` add to the unit: declarations as they are, statements wrapped.

        With ``quiet``, the value of a last expression or declaration ending in ``;`` is
        caught too, unprinted: it is what ``ProcessLine`` hands back.
        """
        chunk = Chunk(prompt)
        lines = typed.split("\n")
        group: list[Piece] = []
        pieces = _caught(chunk, pieces) if quiet else pieces
        for piece in pieces:
            what = kind(piece)
            origin = Origin(prompt, piece.line, piece.column, lines[piece.line - 1])
            if what.what == "statement" and piece.end != "open":
                group.append(piece)
                continue
            self._wrap(chunk, group, lines)
            if piece.end == "open":
                self._value(chunk, piece, what, origin)
            else:
                chunk.add(piece.text, origin)
        self._wrap(chunk, group, lines)
        if quiet and pieces:
            self._declared_value(chunk, pieces, origin)
        return chunk

    def _declared_value(self, chunk: Chunk, pieces: list[Piece], origin: Origin) -> None:
        """A last declaration ending in ``;`` - ``int x = 4;`` - caught, not printed: its value
        is what ``ProcessLine`` hands back, and what ``root -q`` exits with."""
        name = _declared_name(pieces[-1]) if chunk.expression is None else None
        if name is None:
            return
        chunk.expression, chunk.quiet = name, True
        chunk.capture = self.name("value")
        chunk.add(f"auto {chunk.capture} = ({name});", origin)

    def _wrap(self, chunk: Chunk, group: list[Piece], lines: list[str]) -> None:
        """Statements, in a function of their own that is then called: ``void f() {...} f();``.

        The group is emptied: its statements are in the chunk now."""
        if not group:
            return
        name = self.name("statements")
        first, last = group[0], group[-1]
        start = Origin(chunk.prompt, first.line, first.column, lines[first.line - 1])
        end_line = last.line + last.text.count("\n")
        end = Origin(chunk.prompt, end_line, 1, lines[end_line - 1])
        chunk.add(f"void {name}() {{", start)
        for piece in group:
            chunk.add(piece.text, Origin(chunk.prompt, piece.line, piece.column,
                                         lines[piece.line - 1]))  # fmt: skip
        chunk.add("}", end)
        # A call is no declaration, which is all C++ has at the top: an initializer makes it one.
        chunk.add(f"int {self.name('ran')} = ({name}(), 0);", end)
        group.clear()

    def _value(self, chunk: Chunk, piece: Piece, what: Kind, origin: Origin) -> None:
        """The last piece, typed without its ``;``: run it, and catch its value to print."""
        if what.what == "declaration":
            chunk.add(piece.text + ";", origin)
            if what.name is None:
                return
            expression = what.name
        else:
            # On one line, which is what the capture is: a comment ending a line would end it.
            expression = piece.text if "\n" not in piece.text else " ".join(piece.parts)
            chunk.lvalue = _lvalue(expression)
        chunk.expression = expression
        chunk.capture = self.name("value")
        chunk.add(f"auto {chunk.capture} = ({expression});", origin)

    # -- translating and running ---------------------------------------------------------

    def _translated(self, chunk: Chunk) -> tuple[Any, dict[str, tuple[str, str | None]]]:
        source = "\n".join(self.lines + chunk.lines) + "\n"
        try:
            return translate_declaring(source, FILE, [self.directory])
        except Refusal as why:
            refused = self._diagnostic(chunk, why.where, why.why)
            refused.translating = True
            raise refused from None

    def run(self, chunk: Chunk) -> Outcome:
        """Run what ``chunk`` adds, keep it if it ran, and hand back its value, if any."""
        made, declared = self._translated(chunk)
        outcome = self._outcome(chunk, declared)
        if outcome.valued and _by_address(outcome.spelled):
            made, declared = self._by_address(chunk, made, declared)
        selected, keys = self._new(made)
        self._check_names(chunk, made, selected)
        self._execute(chunk, made, selected)
        if outcome.valued:
            outcome.value = self.namespace.pop(declared[chunk.capture][0], None)
        self._keep(chunk, keys, declared)
        return outcome

    def _outcome(self, chunk: Chunk, declared: dict[str, tuple[str, str | None]]) -> Outcome:
        if chunk.expression is None:
            return Outcome()
        named = chunk.expression if chunk.expression in declared else chunk.capture
        spelled = declared.get(named, ("", None))[1]
        return Outcome(spelled=spelled, lvalue=chunk.lvalue or named != chunk.capture,
                       printed=not chunk.quiet, valued=True)  # fmt: skip

    def _by_address(self, chunk: Chunk, made: Any, declared: Any) -> tuple[Any, Any]:
        """An object held by value, caught by its address rather than copied - as cling
        prints the object itself, not a copy of it - if the translator takes that."""
        line = f"auto {chunk.capture} = ({chunk.expression});"
        index = len(chunk.lines) - 1 - chunk.lines[::-1].index(line)
        chunk.lines[index] = f"auto *{chunk.capture} = &({chunk.expression});"
        try:
            return self._translated(chunk)
        except Diagnostic:
            chunk.lines[index] = line
            return made, declared

    def _new(self, made: Any) -> tuple[list[ast.stmt], list[tuple[str, int, int]]]:
        """The top-level Python statements of ``made`` that have not run, and their keys."""
        selected: list[ast.stmt] = []
        keys: list[tuple[str, int, int]] = []
        seen: dict[tuple[str, int], int] = {}
        for statement in ast.parse(made.python).body:
            where = _first_mapped(made.source_map, statement)
            if where is None:
                key = ("", hash(ast.dump(statement)), 0)
            else:
                count = seen[where] = seen.get(where, -1) + 1
                key = (where[0], where[1], count)
            if key not in self.done:
                selected.append(statement)
                keys.append(key)
        return selected, keys

    def _check_names(self, chunk: Chunk, made: Any, selected: list[ast.stmt]) -> None:
        """Refuse, as cling does before running anything, a name nothing declares."""
        root = self.namespace["ROOT"]
        for statement in selected:
            for node in ast.walk(statement):
                if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                        and node.value.id == "ROOT"):  # fmt: skip
                    continue
                try:
                    getattr(root, node.attr)
                except AttributeError:
                    where = made.source_map.get(node.lineno)
                    raise self._diagnostic(chunk, where, f"use of undeclared identifier "
                                           f"'{node.attr}'", node.attr) from None  # fmt: skip

    def _execute(self, chunk: Chunk, made: Any, selected: list[ast.stmt]) -> None:
        label = f"<ROOT_prompt {self.serial}>"
        self.serial += 1
        self.maps[label] = made.source_map
        code = compile(ast.Module(body=selected, type_ignores=[]), label, "exec")
        try:
            exec(code, self.namespace)
        except (SystemExit, KeyboardInterrupt):
            raise
        except CintError as why:  # a line of C++ this one ran, gROOT->ProcessLine("...")
            raise self._diagnostic(chunk, why.where or self._located(why), why.why) from None
        except Exception as why:
            where = self._located(why)
            raise self._diagnostic(chunk, where, f"{type(why).__name__}: {why}") from None

    def _located(self, why: BaseException) -> tuple[str, int] | None:
        """The C++ line ``why`` was raised at: of the innermost prompt line it went through."""
        deepest = None
        trace: types.TracebackType | None = why.__traceback__
        while trace is not None:
            mapped = self.maps.get(trace.tb_frame.f_code.co_filename)
            if mapped is not None and trace.tb_lineno in mapped:
                deepest = mapped[trace.tb_lineno]
            trace = trace.tb_next
        return deepest

    def _keep(self, chunk: Chunk, keys: list[tuple[str, int, int]], declared: Any) -> None:
        self.lines.extend(chunk.lines[: chunk.kept])
        self.origins.extend(chunk.origins[: chunk.kept])
        limit = len(self.lines)
        self.done.update(key for key in keys if key[0] != FILE or key[1] <= limit)
        self.types = {name: spelled for name, (_, spelled) in declared.items()
                      if not name.startswith("xrdroot_")}  # fmt: skip

    # -- reporting -----------------------------------------------------------------------

    def _diagnostic(
        self, chunk: Chunk, where: Any, why: str, named: str | None = None
    ) -> Diagnostic:
        """``why`` placed as cling places it: ``ROOT_prompt_N:line:column``."""
        file, line = (where.file, where.line) if hasattr(where, "file") else (where or ("", 0))
        origins = self.origins + chunk.origins
        if file != FILE or not 0 < line <= len(origins):
            return Diagnostic(f"{file}:{line}:1" if file else f"ROOT_prompt_{chunk.prompt}:1:1",
                              why)  # fmt: skip
        origin = origins[line - 1]
        column = origin.column
        if named is not None:
            found = re.search(rf"\b{re.escape(named)}\b", origin.text)
            column = found.start() + 1 if found else column
        return Diagnostic(f"ROOT_prompt_{origin.prompt}:{origin.line}:{column}", why)


#: The words a statement that is no expression starts with: its value is nothing.
CONTROL = frozenset(
    "if for while do switch try return break continue goto throw delete case default else "
    "co_return co_yield".split()
)


def _expression(piece: Piece) -> bool:
    """Whether ``piece`` is an expression statement ending in ``;``, such as ``h.Fill(1);``."""
    first = piece.parts[0] if piece.parts else ""
    return (piece.end == "semicolon" and kind(piece).what == "statement"
            and first not in CONTROL and first not in ("{", ";"))  # fmt: skip


def _declared_name(piece: Piece) -> str | None:
    """The variable a declaration ending in ``;`` declares - but not an array, whose value
    is its address: nothing ``root -q`` exits with but 255."""
    found = kind(piece)
    if piece.end != "semicolon" or found.what != "declaration" or found.name is None:
        return None
    after = piece.parts.index(found.name) + 1
    return None if piece.parts[after : after + 1] == ["["] else found.name


def _caught(chunk: Chunk, pieces: list[Piece]) -> list[Piece]:
    """``pieces``, a last expression statement among them without its ``;``, so its value
    is caught: unprinted, for the chunk is then ``quiet``."""
    if not pieces or not _expression(pieces[-1]):
        return pieces
    chunk.quiet = True
    return [*pieces[:-1], _opened(pieces[-1])]


def _opened(piece: Piece) -> Piece:
    """``piece`` without its ``;``: an expression whose value is caught."""
    text = piece.text.rstrip()
    return replace(piece, text=text[:-1].rstrip(), end="open", parts=piece.parts[:-1])


def _first_mapped(source_map: dict[int, tuple[str, int]], statement: ast.stmt) -> Any:
    end = statement.end_lineno or statement.lineno
    for line in range(statement.lineno, end + 1):
        if line in source_map:
            return source_map[line]
    return None


def _lvalue(expression: str) -> bool:
    """Whether the expression names an object, as ``h``, ``*p``, ``a.b`` and ``a = b`` do."""
    text = expression.strip()
    named = bool(NAMED.fullmatch(text)) and not text.endswith(")")
    return named or bool(ASSIGNED.search(text))


def _by_address(spelled: str | None) -> bool:
    """Whether a value of this type is an object to catch by its address, not a copy."""
    if spelled is None or spelled.endswith("*") or "[" in spelled:
        return False
    return not PLAIN.match(spelled) and spelled not in STRING_CLASSES
