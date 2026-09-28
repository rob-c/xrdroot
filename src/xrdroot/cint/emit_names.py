"""Names and literals: what Python a C++ name or constant stands for.

A name the macro declared is its symbol's Python (``x``, ``x.value`` for a
cell, ``self.x`` for a member). One it did not declare is the standard
library's - ``std::cout`` is the runtime's ``cout``, ``std::string`` is
``str``, ``std::vector<double>`` is ``ROOT.std.vector['double']`` - or C's,
from the runtime (``printf``, ``sqrt``, ``M_PI``), or else ROOT's:
``ROOT.TH1F``, ``ROOT.TMath.Pi``, ``ROOT.kRed``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .base import TAKEN, Out, P
from .ctype import CType
from .infer import Inference
from .nodes import Literal, Name
from .operators import python_operator
from .symbols import Symbol

__all__ = ["NameEmitter", "STD", "type_text"]

#: ``std::`` names that the runtime (or Python) provides, and what they are called there.
STD = {
    **{
        name: name
        for name in """cout cerr clog endl flush fixed scientific left right
        boolalpha noboolalpha hex dec oct showpos noshowpos setw setprecision setfill
        defaultfloat ostringstream ifstream istringstream sqrt cbrt exp exp2 expm1 log log10 log2
        log1p pow sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh fabs
        floor ceil trunc fmod hypot erf erfc tgamma lgamma isnan isinf isfinite copysign
        fmin fmax to_string stoi stod min max strlen strcmp atoi atof tolower toupper""".split()
    },
    "string": "str",
    "string_view": "str",
    "abs": "cabs",
    "round": "cround",
    "stof": "stod",
    "stol": "stoi",
    "stoll": "stoi",
    "pair": "Pair",
    "make_pair": "Pair",
    "runtime_error": "RuntimeError",
    "logic_error": "RuntimeError",
    "invalid_argument": "ValueError",
    "out_of_range": "IndexError",
    "exception": "Exception",
    "stringstream": "ostringstream",
}

#: C's names that are the runtime's under another name.
C_NAMES = {
    "abs": "cabs",
    "round": "cround",
    "assert": "cassert",
    "stdout": "'stdout'",
    "stderr": "'stderr'",
}

#: ROOT's constants that are Python's own.
ROOT_CONSTANTS = {"kTRUE": "True", "kFALSE": "False", "kNPOS": "npos"}

#: What a ROOT member looks like, used in a method of a class deriving from one of ROOT's.
ROOT_MEMBER = re.compile(r"^f[A-Z]\w*$")


def type_text(ctype: Any) -> str:
    """A type as C++ spells it, for a template argument: ``double``, ``std::pair<int,float>``."""
    if not isinstance(ctype, CType):
        return str(getattr(ctype, "value", ctype))
    text = ctype.name
    if ctype.args:
        text += "<" + ",".join(type_text(arg) for arg in ctype.args) + ">"
    return text + "*" * ctype.pointer


class NameEmitter(Inference):
    """The emitter's layer for names and literals."""

    def expr(self, node: Any) -> Out:
        raise NotImplementedError

    def value(self, node: Any) -> str:
        return self.expr(node)[0]

    def at(self, node: Any, precedence: int) -> str:
        """``node`` as Python, bracketed if it binds more loosely than ``precedence``."""
        text, own = self.expr(node)
        return text if own >= precedence else f"({text})"

    # -- resolving -----------------------------------------------------------

    def symbol(self, name: Name) -> Symbol | None:
        """The macro's own declaration ``name`` refers to, or ``None`` if it is not the macro's."""
        parts = self._unqualified(name.parts)
        if len(parts) == 1:
            return self.lookup(parts[0])
        owner = self.program.classes.get(parts[-2])
        if owner is not None:
            return self._class_member(owner.name, parts[-1])
        enum = self.program.enums.get(parts[-2])
        if enum is not None:
            py = f"{parts[-2]}.{parts[-1]}" if enum.scoped else parts[-1]
            return Symbol(parts[-1], "constant", py, CType("int"))
        return None

    def _unqualified(self, parts: list[str]) -> list[str]:
        """``ns::f`` as ``f`` when ``ns`` is one of the macro's own (flattened) namespaces."""
        known = self.program.classes.keys() | self.program.enums.keys()
        while len(parts) > 1 and parts[0] not in known and self._own_namespace(parts[0]):
            parts = parts[1:]
        return parts

    def _own_namespace(self, name: str) -> bool:
        return name in self.namespaces

    #: The namespaces the macro declares.
    namespaces: set[str]

    def _class_member(self, klass: str, name: str) -> Symbol | None:
        info = self.program.classes[klass]
        if name == klass:
            return self.lookup(klass)
        if name in info.statics or name in info.constants:
            found = info.statics.get(name)
            return Symbol(name, "static", f"{klass}.{name}", found.ctype if found else None)
        if name in info.methods:
            return Symbol(name, "static", f"{klass}.{python_operator(name)}", owner=klass)
        return None

    def use(self, symbol: Symbol) -> Out:
        """What a use of ``symbol`` reads: its Python name, through a cell or ``self``."""
        if symbol.alias is not None:
            return self.expr(symbol.alias)
        if symbol.kind == "field":
            return f"self.{symbol.py}{'.value' if symbol.cell else ''}", P.POSTFIX
        if symbol.kind == "method":
            return f"self.{symbol.py}", P.POSTFIX
        if symbol.cell:
            return f"{symbol.py}.value", P.POSTFIX
        return symbol.py, P.ATOM if "." not in symbol.py else P.POSTFIX

    def name(self, node: Name) -> Out:
        symbol = self.symbol(node)
        if symbol is not None:
            return self.use(symbol)
        return self.library(node)

    def library(self, node: Name) -> Out:
        """A name the macro did not declare: the standard library's, C's, or ROOT's."""
        parts = node.parts
        if parts[0] == "std" and len(parts) == 2 and parts[1] in STD:
            return STD[parts[1]], P.ATOM
        if parts[0] == "std" and parts[-1] == "npos":
            return "npos", P.ATOM
        if len(parts) == 1:
            return self._plain(node)
        return self.root(parts) + self.targs(node.targs), P.POSTFIX

    def _plain(self, node: Name) -> Out:
        name = node.last
        if name in C_NAMES:
            return C_NAMES[name], P.ATOM
        if name in ROOT_CONSTANTS:
            return ROOT_CONSTANTS[name], P.ATOM
        if name in TAKEN:
            return name, P.ATOM
        if self.scope.klass and ROOT_MEMBER.match(name) and self.derives_from_root():
            return f"self.{name}", P.POSTFIX
        return f"ROOT.{name}{self.targs(node.targs)}", P.POSTFIX

    def derives_from_root(self) -> bool:
        """Is the class whose method is being written derived, somewhere up, from ROOT's?"""
        klass = self.scope.klass
        seen: set[str] = set()
        while klass is not None and klass not in seen:
            seen.add(klass)
            info = self.program.classes.get(klass)
            if info is None:
                return True
            bases = [base.ctype.name for base in info.decl.bases]
            if not bases:
                return False
            klass = next((b for b in bases if b not in self.program.classes), bases[0])
        return False

    def targs(self, targs: list[Any] | None) -> str:
        """``<double, 3>`` as the subscript ``['double', 3]`` a pyroot template takes."""
        if not targs:
            return ""
        items = [
            repr(type_text(arg)) if isinstance(arg, CType) else self.value(arg) for arg in targs
        ]
        return "[" + ", ".join(items) + "]"

    # -- literals ------------------------------------------------------------

    def literal(self, node: Literal) -> Out:
        return _LITERALS[node.kind](node), P.ATOM


def _character(node: Literal) -> str:
    value = node.value
    if 32 <= value < 127 and chr(value) not in "'\\":
        return f"ord({chr(value)!r})"
    return str(value)


_LITERALS: dict[str, Callable[[Literal], str]] = {
    "int": lambda node: str(node.value),
    "float": lambda node: repr(float(node.value)),
    "str": lambda node: repr(node.value),
    "char": _character,
    "bool": lambda node: "True" if node.value else "False",
    "null": lambda node: "None",
}
