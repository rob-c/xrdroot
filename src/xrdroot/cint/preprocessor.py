"""The C preprocessor, as Cling runs it over a macro before reading a word of C++.

``#include`` of a header ROOT or the system provides is dropped - the names
it would declare are ROOT's, and reach the Python through
:mod:`xrdroot.pyroot` - while a ``"local.h"`` next to the macro is read in
where it stands. ``#define`` makes object-like and function-like macros
(``#`` stringising, ``##`` pasting and ``__VA_ARGS__`` included), and the
conditional sections are decided as Cling decides them: ``__CLING__`` is
defined, ``__CINT__`` and ``__ROOTCLING__`` are not, and ROOT's own
housekeeping macros - ``ClassDef``, ``ClassImp``, ``R__LOAD_LIBRARY``,
``R__ADD_INCLUDE_PATH`` - expand to nothing.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path

from .errors import Refusal, Where
from .ppexpr import evaluate
from .tokens import Token, tokenize

__all__ = ["Macro", "Preprocessor", "preprocess", "PREDEFINED"]

#: ROOT's version as Cling reports it, ``ROOT_VERSION(6, 40, 4)``.
ROOT_VERSION = (6 << 16) | (40 << 8) | 4

#: The macros defined before a macro's first line: Cling's, the platform's and ROOT's own.
PREDEFINED = f"""
#define __CLING__ 1
#define __cplusplus 201703L
#define __linux__ 1
#define __unix__ 1
#define R__UNIX 1
#define R__LINUX 1
#define R__USE_IMT 1
#define ROOT_VERSION_CODE {ROOT_VERSION}
#define ROOT_VERSION(a, b, c) (((a) << 16) + ((b) << 8) + (c))
#define R__LOAD_LIBRARY(x)
#define R__ADD_INCLUDE_PATH(x)
#define R__ADD_LIBRARY_PATH(x)
#define ClassDef(name, version)
#define ClassDefOverride(name, version)
#define ClassDefNV(name, version)
#define ClassDefInline(name, version)
#define ClassDefInlineOverride(name, version)
#define ClassDefInlineNV(name, version)
#define ClassImp(name)
#define ClassImpUnique(name, key)
#define templateClassImp(name)
#define NamespaceImp(name)
#define R__EXTERN extern
#define BIT(n) (1ULL << (n))
#define SETBIT(n, i) ((n) |= BIT(i))
#define CLRBIT(n, i) ((n) &= ~BIT(i))
#define TESTBIT(n, i) ((bool)(((n) & BIT(i)) != 0))
#define _R__UNIQUE_(X) X
#define R__DEPRECATED(maj, min, reason)
#define RQ_OBJECT(name)
#define Q_OBJECT
#define R__CLING_PTRCHECK(on)
#define _QUOTE_(name) #name
"""

#: The extensions a local file an ``#include`` names may be read from.
MAXIMUM_DEPTH = 40


class Macro:
    """One ``#define``: its name, its parameters (``None`` if object-like) and its body."""

    __slots__ = ("name", "params", "variadic", "body")

    def __init__(
        self, name: str, params: list[str] | None, variadic: bool, body: list[Token]
    ) -> None:
        self.name = name
        self.params = params
        self.variadic = variadic
        self.body = body


class _Section:
    """One level of ``#if`` nesting: is its text kept, and has a branch of it been taken?"""

    __slots__ = ("active", "taken", "outer")

    def __init__(self, active: bool, taken: bool, outer: bool) -> None:
        self.active = active
        self.taken = taken
        self.outer = outer


def _line(tokens: list[Token], at: int) -> int:
    """Where the directive line starting at ``at`` ends: at the next token that begins a line."""
    end = at + 1
    while end < len(tokens) and not tokens[end].bol and tokens[end].kind != "eof":
        end += 1
    return end


def _pasted(left: Token, right: Token) -> Token:
    text = left.text + right.text
    words = tokenize(text, left.where.file)
    if len(words) != 2:
        raise Refusal(f"## makes {text!r}, which is not one C++ token", left.where)
    word = words[0]
    return Token(word.kind, word.text, left.where, False, left.hide, left.space)


def _stringised(argument: list[Token], where: Where) -> Token:
    parts = []
    for index, token in enumerate(argument):
        gap = " " if token.space and index else ""
        parts.append(gap + token.text)
    text = "".join(parts).replace("\\", "\\\\").replace('"', '\\"')
    return Token("str", f'"{text}"', where)


class Preprocessor:
    """Runs directives and expands macros over the tokens of a macro and its local headers."""

    def __init__(self, directories: list[Path] | None = None) -> None:
        self.macros: dict[str, Macro] = {}
        self.directories = list(directories or [])
        #: Every local file read in by an ``#include``, which a translation depends on.
        self.included: list[Path] = []
        self._depth = 0
        self._sections: list[_Section] = []
        self._directives: dict[str, Callable[[list[Token], Where, list[Token]], None]] = {
            "define": self._define,
            "undef": self._undef,
            "include": self._include,
            "include_next": self._include,
            "import": self._include,
            "if": self._if,
            "ifdef": self._ifdef,
            "ifndef": self._ifndef,
            "elif": self._elif,
            "else": self._else,
            "endif": self._endif,
            "error": self._error,
        }
        self.run(tokenize(PREDEFINED, "<predefined>"))

    # -- the stream ---------------------------------------------------------

    @property
    def active(self) -> bool:
        return not self._sections or self._sections[-1].active

    def run(self, tokens: list[Token]) -> list[Token]:
        """``tokens`` with their directives run and their macros expanded, ``eof`` dropped."""
        out: list[Token] = []
        pending: list[Token] = []
        at = 0
        while at < len(tokens) and tokens[at].kind != "eof":
            token = tokens[at]
            if token.bol and token.is_("#"):
                out.extend(self.expand(pending))
                pending = []
                end = _line(tokens, at)
                self._directive(tokens[at + 1 : end], token.where, out)
                at = end
                continue
            if self.active:
                pending.append(token)
            at += 1
        out.extend(self.expand(pending))
        return out

    def finish(self, where: Where) -> None:
        """Refuse a macro whose ``#if`` sections are not all closed at its end."""
        if self._sections:
            raise Refusal("an #if is never closed by its #endif", where)

    def _directive(self, words: list[Token], where: Where, out: list[Token]) -> None:
        if not words:
            return
        name = words[0].text
        handler = self._directives.get(name)
        conditional = name in ("if", "ifdef", "ifndef", "elif", "else", "endif")
        if handler is not None and (self.active or conditional):
            handler(words[1:], where, out)

    # -- definitions --------------------------------------------------------

    def _define(self, words: list[Token], where: Where, out: list[Token]) -> None:
        if not words or words[0].kind != "id":
            raise Refusal("#define names no macro", where)
        name = words[0].text
        rest = words[1:]
        if rest and rest[0].is_("(") and not rest[0].space:
            params, variadic, body = self._parameters(rest, where)
            self.macros[name] = Macro(name, params, variadic, body)
        else:
            self.macros[name] = Macro(name, None, False, rest)

    @staticmethod
    def _parameters(rest: list[Token], where: Where) -> tuple[list[str], bool, list[Token]]:
        params: list[str] = []
        variadic = False
        at = 1
        while at < len(rest) and not rest[at].is_(")"):
            word = rest[at]
            if word.is_("..."):
                variadic = True
            elif word.kind == "id":
                params.append(word.text)
            at += 1
        if at == len(rest):
            raise Refusal("a #define's parameter list is never closed", where)
        return params, variadic, rest[at + 1 :]

    def _undef(self, words: list[Token], where: Where, out: list[Token]) -> None:
        if words:
            self.macros.pop(words[0].text, None)

    def _error(self, words: list[Token], where: Where, out: list[Token]) -> None:
        message = " ".join(word.text for word in words)
        raise Refusal(f"the macro stops itself with #error {message}", where)

    # -- includes -----------------------------------------------------------

    def _include(self, words: list[Token], where: Where, out: list[Token]) -> None:
        words = self.expand(words)
        if not words or words[0].kind != "str":
            return
        name = words[0].text[1:-1]
        found = self.find(name, where)
        if found is None:
            return
        if self._depth >= MAXIMUM_DEPTH:
            raise Refusal(f"#include {name} nests deeper than {MAXIMUM_DEPTH} files", where)
        self.included.append(found)
        self._depth += 1
        self.directories.insert(0, found.parent)
        try:
            out.extend(self.run(tokenize(_read(found), str(found))))
        finally:
            self.directories.pop(0)
            self._depth -= 1

    def find(self, name: str, where: Where) -> Path | None:
        """The local file ``"name"`` means - beside the including file first - if there is one."""
        places = [Path(where.file).parent, *self.directories]
        for place in places:
            candidate = place / name
            if candidate.is_file():
                return candidate
        return None

    # -- conditional sections -----------------------------------------------

    def _open(self, keep: bool) -> None:
        outer = self.active
        self._sections.append(_Section(outer and keep, keep, outer))

    def _if(self, words: list[Token], where: Where, out: list[Token]) -> None:
        self._open(self.active and bool(self.condition(words, where)))

    def _ifdef(self, words: list[Token], where: Where, out: list[Token]) -> None:
        self._open(bool(words) and words[0].text in self.macros)

    def _ifndef(self, words: list[Token], where: Where, out: list[Token]) -> None:
        self._open(not (words and words[0].text in self.macros))

    def _top(self, directive: str, where: Where) -> _Section:
        if not self._sections:
            raise Refusal(f"#{directive} has no #if to belong to", where)
        return self._sections[-1]

    def _elif(self, words: list[Token], where: Where, out: list[Token]) -> None:
        section = self._top("elif", where)
        if section.taken or not section.outer:
            section.active = False
            return
        keep = bool(self.condition(words, where))
        section.active = section.taken = keep

    def _else(self, words: list[Token], where: Where, out: list[Token]) -> None:
        section = self._top("else", where)
        section.active = section.outer and not section.taken
        section.taken = True

    def _endif(self, words: list[Token], where: Where, out: list[Token]) -> None:
        self._top("endif", where)
        self._sections.pop()

    def condition(self, words: list[Token], where: Where) -> int:
        """What an ``#if`` or ``#elif`` line is worth, ``defined`` and macros resolved."""
        resolved = self.expand(self._defined(words, where))
        numbers = [_zero(word) if word.kind == "id" else word for word in resolved]
        return evaluate(numbers, where)

    def _defined(self, words: list[Token], where: Where) -> list[Token]:
        out: list[Token] = []
        at = 0
        while at < len(words):
            word = words[at]
            if word.is_("defined"):
                name, at = _operand(words, at + 1, where)
                out.append(_one(name in self.macros, word.where))
                continue
            if word.is_("__has_include", "__has_include_next"):
                found, at = self._has_include(words, at + 1, where)
                out.append(_one(found, word.where))
                continue
            out.append(word)
            at += 1
        return out

    def _has_include(self, words: list[Token], at: int, where: Where) -> tuple[bool, int]:
        close = at
        while close < len(words) and not words[close].is_(")"):
            close += 1
        inside = words[at + 1 : close]
        if inside and inside[0].kind == "str":
            return self.find(inside[0].text[1:-1], where) is not None, close + 1
        # A system header, <cmath> or <vector>: Cling has it.
        return True, close + 1

    # -- expansion ----------------------------------------------------------

    def expand(self, tokens: list[Token]) -> list[Token]:
        """``tokens`` with every macro in them expanded, as many times over as it takes."""
        work = deque(tokens)
        out: list[Token] = []
        while work:
            token = work.popleft()
            replaced = self._replacement(token, work)
            if replaced is None:
                out.append(token)
            else:
                work.extendleft(reversed(replaced))
        return out

    def _replacement(self, token: Token, work: deque[Token]) -> list[Token] | None:
        if token.kind != "id" or token.text in token.hide:
            return None
        if token.text in ("__LINE__", "__FILE__"):
            return [_position(token)]
        macro = self.macros.get(token.text)
        if macro is None:
            return None
        body = self._body(macro, token, work)
        if body is None:
            return None
        hide = token.hide | {macro.name}
        return [word.moved(token.where, hide) for word in body]

    def _body(self, macro: Macro, token: Token, work: deque[Token]) -> list[Token] | None:
        """What a use of ``macro`` becomes - ``None`` for a function-like one not called."""
        if macro.params is None:
            return macro.body
        if not work or not work[0].is_("("):
            return None
        return self._substituted(macro, _arguments(work, token.where), token.where)

    def _substituted(self, macro: Macro, arguments: list[list[Token]], where: Where) -> list[Token]:
        named = _bind(macro, arguments, where)
        out: list[Token] = []
        at = 0
        while at < len(macro.body):
            at = self._substitute(macro.body, at, named, out, where)
        return out

    def _substitute(
        self,
        body: list[Token],
        at: int,
        named: dict[str, list[Token]],
        out: list[Token],
        where: Where,
    ) -> int:
        """Put what ``body[at]`` becomes on ``out``, and say where the body goes on from."""
        word = body[at]
        after = body[at + 1] if at + 1 < len(body) else None
        if after is not None and self._operator(word, after, named, out, where):
            return at + 2
        if word.kind == "id" and word.text in named:
            pasting = after is not None and after.is_("##")
            out.extend(named[word.text] if pasting else self.expand(named[word.text]))
        else:
            out.append(word)
        return at + 1

    def _operator(
        self,
        word: Token,
        after: Token,
        named: dict[str, list[Token]],
        out: list[Token],
        where: Where,
    ) -> bool:
        """``#param`` and ``a ## b``: stringise or paste, if that is what ``word`` is."""
        if word.is_("#") and after.text in named:
            out.append(_stringised(named[after.text], where))
            return True
        if word.is_("##") and out:
            right = named.get(after.text, [after]) if after.kind == "id" else [after]
            out.extend(self._paste(out.pop(), right))
            return True
        return False

    @staticmethod
    def _paste(left: Token, right: list[Token]) -> list[Token]:
        if not right:
            return [left]
        return [_pasted(left, right[0]), *right[1:]]


def _bind(macro: Macro, arguments: list[list[Token]], where: Where) -> dict[str, list[Token]]:
    params = macro.params or []
    if arguments == [[]] and not params:
        arguments = []
    if len(arguments) < len(params) or (len(arguments) > len(params) and not macro.variadic):
        raise Refusal(
            f"{macro.name}() is a macro of {len(params)} arguments, used with {len(arguments)}",
            where,
        )
    named = dict(zip(params, arguments))
    if macro.variadic:
        named["__VA_ARGS__"] = _joined_arguments(arguments[len(params) :], where)
    return named


def _joined_arguments(extra: list[list[Token]], where: Where) -> list[Token]:
    """``__VA_ARGS__``: the arguments past the named ones, commas between them again."""
    joined: list[Token] = []
    for index, argument in enumerate(extra):
        if index:
            joined.append(Token("op", ",", where))
        joined.extend(argument)
    return joined


def _arguments(work: deque[Token], where: Where) -> list[list[Token]]:
    """The arguments of a function-like macro's use, taken off the front of ``work``."""
    work.popleft()
    arguments: list[list[Token]] = [[]]
    depth = 0
    while work:
        token = work.popleft()
        if token.is_(")") and depth == 0:
            return arguments
        if token.is_(",") and depth == 0:
            arguments.append([])
            continue
        depth += token.is_("(") - token.is_(")")
        arguments[-1].append(token)
    raise Refusal("a macro's arguments are never closed by a )", where)


def _operand(words: list[Token], at: int, where: Where) -> tuple[str, int]:
    """The name after ``defined``, bracketed or not, and where reading goes on."""
    if at < len(words) and words[at].is_("("):
        if at + 2 >= len(words) or not words[at + 2].is_(")"):
            raise Refusal("defined( ) must hold one name", where)
        return words[at + 1].text, at + 3
    if at >= len(words):
        raise Refusal("defined must be followed by a name", where)
    return words[at].text, at + 1


def _one(truth: bool, where: Where) -> Token:
    return Token("num", "1" if truth else "0", where)


def _zero(word: Token) -> Token:
    # A name still standing after expansion counts as 0 - true and false as C++ spells them.
    return Token("num", "1" if word.text == "true" else "0", word.where)


def _position(token: Token) -> Token:
    if token.text == "__LINE__":
        return Token("num", str(token.where.line), token.where)
    return Token("str", '"' + token.where.file.replace("\\", "\\\\") + '"', token.where)


def _read(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def preprocess(
    text: str, file: str = "<macro>", directories: list[Path] | None = None
) -> tuple[list[Token], list[Path]]:
    """The tokens of a macro after preprocessing, and the local files it read in."""
    pre = Preprocessor(directories)
    tokens = pre.run(tokenize(text, file))
    pre.finish(Where(file, tokens[-1].where.line if tokens else 1))
    return tokens, pre.included
