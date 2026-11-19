"""A class of a translated program, described as its C++ declaration wrote it.

ROOT's ``.Class`` prints a class's layout out of a dictionary: how many bytes
one takes, the file and line it was declared at, its bases, and every member
function's C++ signature. A translation keeps none of that in Python - a Python
class is its attributes - so a description is read back off the syntax tree the
translator built, and the size off what ``sizeof`` gives (:mod:`.sizes`).
Nothing here prints anything: the layout's text is ROOT's, and belongs to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .ctype import CType
from .dialect import active, using
from .nodes import Function
from .parse_decl import parse
from .preprocessor import preprocess
from .program import ClassInfo, Program
from .sizes import size_of
from .typeids import qualified_names, spelled

__all__ = ["Description", "BaseClass", "Method", "described", "program_of"]

#: The ``*``s a type ends with, which a signature sets off by a space.
POINTER = re.compile(r"(\*+)$")


@dataclass
class BaseClass:
    """A base the class inherits: the name its declaration gave it, and how."""

    name: str
    access: str


@dataclass
class Method:
    """A member function: how it is reached, and its signature as C++ spells it."""

    #: ``public``, ``private`` or ``protected``, as the member was declared.
    access: str
    virtual: bool
    static: bool
    #: ``template <class X> `` in front of the signature, or ``""``.
    template: str
    signature: str


@dataclass
class Description:
    """A class as its declaration describes it: ``TClass::Size``, its place, bases and methods."""

    name: str
    #: ``sizeof`` of one object of the class, or ``None`` when its layout is not known here.
    size: int | None
    file: str
    line: int
    bases: list[BaseClass]
    methods: list[Method]


def program_of(text: str, file: str, directories: list[Path] | None = None) -> Program:
    """The program the C++ ``text`` of ``file`` declares, without translating it to Python.

    Cling hands a dictionary's description of a class to anything that asks, wherever the
    class was declared; the translator only knows what it has parsed, so a description of a
    macro's class is made by reading that macro's declarations again.
    """
    with using(active()):
        tokens, _ = preprocess(text, file, directories)
        return Program(parse(tokens))


def described(program: Program, name: str) -> Description | None:
    """What ``program`` says of the class ``name``, or ``None`` when it declares no such class."""
    info = program.classes.get(name)
    if info is None:
        info = program.classes.get(name.split("::")[-1])
    if info is None:
        return None
    return _described(info, program)


def _described(info: ClassInfo, program: Program) -> Description:
    """The description of one parsed class declaration and its members."""
    decl = info.decl
    qualified = qualified_names(program.unit.decls)
    functions = [func for overloads in info.methods.values() for func in overloads]
    return Description(
        qualified.get(decl.name, decl.name),
        size_of(CType(decl.name), program),
        decl.where.file,
        decl.where.line,
        [BaseClass(qualified.get(each.ctype.name, each.ctype.name), each.access)
         for each in decl.bases],
        [_method(func, qualified) for func in functions],
    )  # fmt: skip


def _method(func: Function, qualified: dict[str, str]) -> Method:
    """One member function, its signature spelled as its declaration wrote it."""
    return Method(
        "private" if func.hidden else "public",
        func.virtual,
        func.static,
        _template(func),
        _signature(func, qualified),
    )


def _template(func: Function) -> str:
    """``template <class X> `` for a member template, and nothing for a plain member."""
    if not func.template:
        return ""
    return "template <" + ", ".join(f"class {name}" for name in func.template) + "> "


def _signature(func: Function, qualified: dict[str, str]) -> str:
    """``privateOp2 &operator=(const int &)``: the function as C++ declares it."""
    params = ", ".join(_spelled(each.ctype, qualified) for each in func.params)
    const = " const" if func.const else ""
    if func.kind == "constructor":
        return f"{func.name}({params}){const}"
    if func.kind == "destructor":
        return f"{func.name}({params}){const}"
    returns = _spelled(func.returns, qualified)
    gap = "" if returns.endswith(("*", "&")) else " "  # ``privateOp2 &operator=``
    return f"{returns}{gap}{func.name}({params}){const}"


def _spelled(ctype: CType, qualified: dict[str, str]) -> str:
    """A type as a signature writes it - ``const int &``, ``StThreeVector<double> *``.

    The ``*`` and ``&`` are set off by a space, as ROOT prints them against the type they
    point at and not the name they were declared with.
    """
    text = spelled(ctype, qualified) or ctype.name
    text = POINTER.sub(r" \1", text)
    if ctype.reference:
        text += " &"
    return text
