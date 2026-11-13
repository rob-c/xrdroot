"""Operators a macro defines outside a class, and which one an expression calls.

``bool operator==(const A &, const B &)`` is a function of its own here,
``operator_eq``, and ``a == b`` calls it where the types of ``a`` and ``b``
are ones it takes - chosen as C++'s overload resolution chooses, as far as
the translator knows the types: an exact match before a conversion, a
class before one it derives from, a template's parameter last. C++ only
calls a user's operator when an operand is an object of a class or an
enum, so ``1 + 2`` and ``p == nullptr`` keep the built-in meaning they
have everywhere else. The writers ``ostream &operator<<(ostream &, const
T &)`` are the stream emitter's (see :meth:`_stream_operator`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .ctype import CType
from .nodes import Function
from .program import ClassInfo, Program

__all__ = ["FreeOperator", "free_operators", "choose", "written_type"]

#: The binary operators a free function can be, and the word its Python name is made with.
BINARY = {
    "+": "add", "-": "sub", "*": "mul", "/": "div", "%": "mod", "^": "xor", "&": "and",
    "|": "or", "<": "lt", ">": "gt", "<=": "le", ">=": "ge", "==": "eq", "!=": "ne",
    "<<": "lshift", ">>": "rshift",
}  # fmt: skip

#: The unary operators a free function of one parameter can be, and their words.
UNARY = {"+": "pos", "-": "neg", "!": "not", "~": "invert"}

#: How well an argument fits a parameter, best first; ``None`` is no fit at all. A base
#: class fits at ``BASE`` and one more for each step of derivation between.
EXACT, BASE, CONVERTED, TEMPLATED, TEMPLATE, UNKNOWN = 0, 10, 20, 30, 40, 50

#: The fits of an object to a class's parameter, which make a call a user's operator.
OBJECT_FITS = range(EXACT, CONVERTED)


@dataclass(frozen=True)
class FreeOperator:
    """One free operator function the macro defines, and the Python name it is written as."""

    func: Function
    py: str


def written_type(func: Function) -> str | None:
    """``T`` of ``ostream &operator<<(ostream &, const T &)``, if ``func`` is one."""
    if func.name != "operator<<" or len(func.params) != 2 or func.body is None:
        return None
    if "stream" not in func.params[0].ctype.name:
        return None
    kind = func.params[1].ctype
    return None if kind.pointer else (kind.enum or kind.name)


def free_operators(program: Program) -> dict[str, list[FreeOperator]]:
    """Each operator the macro defines outside a class, by name: ``operator==`` and so on."""
    found: dict[str, list[FreeOperator]] = {}
    for name, funcs in program.functions.items():
        defined = [func for func in funcs if _callable(func)]
        for index, func in enumerate(defined, start=1):
            word = (BINARY if len(func.params) == 2 else UNARY)[name[8:]]
            suffix = f"__{index}" if len(defined) > 1 else ""
            found.setdefault(name, []).append(FreeOperator(func, f"operator_{word}{suffix}"))
    return found


def _callable(func: Function) -> bool:
    """Is ``func`` a defined operator an expression calls - not a stream's writer?"""
    if func.kind != "operator" or func.body is None or written_type(func) is not None:
        return False
    table = {1: UNARY, 2: BINARY}.get(len(func.params), {})
    return func.name[8:] in table


def choose(
    candidates: Sequence[FreeOperator],
    operands: Sequence[CType | None],
    classes: dict[str, ClassInfo],
) -> FreeOperator | None:
    """The candidate the operands' types fit best, if any fits and an operand is an object."""
    best: tuple[int, FreeOperator] | None = None
    for candidate in candidates:
        score = _fits(candidate.func, operands, classes)
        if score is not None and (best is None or score < best[0]):
            best = (score, candidate)
    return best[1] if best is not None else None


def _fits(
    func: Function, operands: Sequence[CType | None], classes: dict[str, ClassInfo]
) -> int | None:
    """How far ``func`` is from taking ``operands``: the sum of each one's distance."""
    if len(func.params) != len(operands):
        return None
    template = set(func.template or [])
    total, objects = 0, 0
    for param, given in zip(func.params, operands, strict=True):
        score = _fit(param.ctype, given, template, classes)
        if score is None:
            return None
        total += score
        objects += (score in OBJECT_FITS or score == TEMPLATED) and _object(param.ctype)
    return total if objects else None


def _object(ctype: CType) -> bool:
    """Is a parameter of ``ctype`` an object of a class or an enum, or a pointer to one - what
    an operand known to be one makes the call a user's operator rather than a built-in one?

    C++ needs an object of a class or an enum among the operands; one of a
    pointer to a class is let stand for it where the other operand's type is
    not known - ``cout << &v`` of ``operator<<(ostream &, const V *)``."""
    return ctype.enum is not None or ctype.is_class


def _fit(
    param: CType, given: CType | None, template: set[str], classes: dict[str, ClassInfo]
) -> int | None:
    """How far an argument of type ``given`` is from a parameter of type ``param``."""
    if param.name in template:
        return TEMPLATE if given is None or given.pointer >= param.pointer else None
    if given is None:  # an object, perhaps, but never taken to be a pointer
        return UNKNOWN if param.is_class and not param.pointer else None
    if _plain(param) or _plain(given):
        return _plain_fit(param, given)
    return _class_fit(param, given, template, classes)


def _plain(ctype: CType) -> bool:
    """A number, an enum or a string: what is matched by kind, not by class."""
    return ctype.enum is not None or ctype.scalar or ctype.is_string


def _plain_fit(param: CType, given: CType) -> int | None:
    """An enum for its own enum, a number for any number - its own type best - a string for
    a string."""
    if param.enum is not None or given.enum is not None:
        return EXACT if param.enum == given.enum else None
    if param.scalar and given.scalar:
        return EXACT if param.name == given.name else CONVERTED
    return EXACT if param.is_string and given.is_string else None


def _class_fit(
    param: CType, given: CType, template: set[str], classes: dict[str, ClassInfo]
) -> int | None:
    """An object for a parameter of its class, a base of it, or a template of it."""
    levels = given.pointer + len(given.dims)
    if not (param.is_class and given.is_class) or param.pointer != levels:
        return None
    if _same(param.name, given.name):
        templated = any(isinstance(arg, CType) and arg.name in template for arg in param.args)
        return TEMPLATED if templated else EXACT
    steps = _steps(given.name, param.name, classes)
    return BASE + steps if steps is not None else None


def _same(one: str, other: str) -> bool:
    """One class however qualified: ``ns::Data`` is ``Data`` after ``using``."""
    return one == other or one.split("::")[-1] == other.split("::")[-1]


def _steps(name: str, base: str, classes: dict[str, ClassInfo]) -> int | None:
    """How many derivations the macro's class ``name`` is from ``base``, if it derives from it:
    the nearer base is the better fit, as C++ ranks them."""
    info = classes.get(name.split("::")[-1])
    if info is None:
        return None
    found = [0 if _same(parent.ctype.name, base) else _steps(parent.ctype.name, base, classes)
             for parent in info.decl.bases]
    reached = [steps + 1 for steps in found if steps is not None]
    return min(reached) if reached else None
