"""Names and literals: what Python a C++ name or constant stands for.

A name the program declared is its symbol's Python (``x``, ``x.value`` for a
cell, ``self.x`` for a member). One it did not declare is the standard
library's - ``std::cout`` is the runtime's ``cout``, ``std::string`` is
``str``, ``std::vector<double>`` is ``CXX.std.vector['double']`` - or C's,
from the runtime (``printf``, ``sqrt``, ``M_PI``), or else the library's,
found in the dialect's host: ``CXX.name``.
"""

from __future__ import annotations

import keyword
from collections.abc import Callable
from typing import Any

from .base import Out, P, taken
from .ctype import CType
from .dialect import active
from .infer import Inference
from .limits import is_limit, limit_text
from .nodes import Binary, Literal, Name, Node, SizeOf, Unary
from .operators import python_operator
from .sizes import size_of
from .symbols import Symbol
from .traits import is_trait, trait

__all__ = ["NameEmitter", "STD", "type_text"]

#: ``std::`` names that the runtime (or Python) provides, and what they are called there.
STD = {
    **{
        name: name
        for name in """cout cerr clog endl flush fixed scientific left right
        boolalpha noboolalpha hex dec oct showpos noshowpos setw setprecision setfill
        defaultfloat ostringstream ofstream ifstream istringstream sqrt cbrt exp exp2 expm1 log
        log10 log2 log1p pow sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh fabs
        floor ceil trunc fmod hypot erf erfc tgamma lgamma isnan isinf isfinite copysign
        fmin fmax to_string stoi stod min max strlen strcmp atoi atof tolower toupper thread mutex
        recursive_mutex lock_guard unique_lock scoped_lock atomic condition_variable ref
        cref""".split()
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

#: The ``std::`` namespaces whose members the runtime has, as objects of those names.
STD_NAMESPACES = frozenset({"chrono", "this_thread"})

#: C's names that are the runtime's under another name.
C_NAMES = {
    "abs": "cabs",
    "round": "cround",
    "assert": "cassert",
    "time": "c_time",
    "stdout": "'stdout'",
    "stderr": "'stderr'",
}

#: The methods a GoogleTest fixture inherits and calls unqualified: ``GetParam()`` in a
#: ``TEST_P`` body is the fixture's own, not a free function of the library's.
INHERITED = frozenset({"GetParam", "HasFailure", "HasFatalFailure", "HasNonfatalFailure",
                       "IsSkipped", "RecordProperty"})  # fmt: skip


def type_text(ctype: Any, spellings: dict[str, str] | None = None) -> str:
    """A type as C++ spells it, for a template argument: ``double``, ``std::pair<int,float>``,
    an instance of a template as the template-id it is, ``Box<int>``."""
    if not isinstance(ctype, CType):
        return str(getattr(ctype, "value", ctype))
    text = (spellings or {}).get(ctype.name, ctype.name)
    if ctype.args:
        text += "<" + ",".join(type_text(arg, spellings) for arg in ctype.args) + ">"
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
        if len(node.parts) == 1 and node.last in self.vector_names:
            return self.vector_names[node.last], P.ATOM  # the array a fill loop draws into
        symbol = self.symbol(node)
        if symbol is not None:
            return self.use(symbol)
        return self.library(node)

    def library(self, node: Name) -> Out:
        """A name the program did not declare: the standard library's, C's, or the host's."""
        parts = node.parts
        if is_limit(parts, node.targs):
            return self.limit(node, node, ""), P.ATOM
        known = self.trait_value(node) if is_trait(node) else None
        if known is not None:
            return str(known), P.ATOM
        found = self._standard(node) if parts[0] == "std" else None
        if found is not None:
            return found
        if len(parts) == 1:
            return self._plain(node)
        return self.host_name(parts) + self.targs(node.targs), P.POSTFIX

    def trait_value(self, node: Name) -> bool | None:
        """A type trait of types all known here - not a parameter of a template being written
        for every use - as the constant it is."""
        args = [*(node.targs or []), *(node.scope_targs or [])]
        if any(self.generic_type(arg) for arg in args if isinstance(arg, CType)):
            return None
        return trait(node, self.program)

    def generic_type(self, arg: CType) -> bool:
        """Does a type name a parameter of a template being written generically?"""
        if {arg.name, arg.name.split("::")[0]} & self.template_names:
            return True
        return any(self.generic_type(inner) for inner in arg.args if isinstance(inner, CType))

    #: Template parameter names in scope, in a template written for every use.
    template_names: set[str]

    def constant_value(self, node: Any) -> Any:
        """``node``'s value, if it is a constant known here - a number, a trait, ``sizeof`` -
        or ``None``: what ``if constexpr`` chooses its branch by."""
        rule = _CONSTANTS.get(type(node))
        return rule(self, node) if rule is not None else None

    def _constant_literal(self, node: Literal) -> Any:
        return node.value if node.kind in ("int", "bool", "char") else None

    def _constant_name(self, node: Name) -> Any:
        return self.trait_value(node) if is_trait(node) else None

    def _constant_size(self, node: SizeOf) -> Any:
        if node.ctype is None or self.generic_type(node.ctype):
            return None
        return size_of(node.ctype, self.program)

    def _constant_unary(self, node: Unary) -> Any:
        inner = self.constant_value(node.operand) if node.op in ("!", "-") else None
        if inner is None:
            return None
        return not inner if node.op == "!" else -inner

    def _constant_binary(self, node: Binary) -> Any:
        operation = CONSTANT_OPERATIONS.get(node.op)
        left, right = self.constant_value(node.left), self.constant_value(node.right)
        if operation is None or left is None or right is None:
            return None
        return operation(left, right)

    def limit(self, name: Name, node: Node, called: str) -> str:
        """``std::numeric_limits<T>::member``, the constant it is - or refused, when ``T`` is
        no arithmetic type or not one known yet."""
        assert name.targs
        kind = name.targs[0]
        found = limit_text(kind, name.last)
        if found is None:
            why = f"std::numeric_limits<{type_text(kind)}>::{name.last}{called}"
            raise self.refuse(why, node)
        return found

    def _standard(self, node: Name) -> Out | None:
        """A ``std::`` name the runtime has - ``cout``, ``npos``, ``chrono::seconds`` - if it is."""
        parts = node.parts
        if len(parts) == 2 and parts[1] in STD:
            return STD[parts[1]], P.ATOM
        if parts[-1] == "npos":
            return "npos", P.ATOM
        if len(parts) < 3 or parts[1] not in STD_NAMESPACES:
            return None
        # ``std::chrono::milliseconds``, ``std::this_thread::sleep_for``: the runtime's.
        text = parts[1]
        for part in parts[2:]:
            text = f"getattr({text}, {part!r})" if keyword.iskeyword(part) else f"{text}.{part}"
        return text + self.targs(node.targs), P.POSTFIX

    def _plain(self, node: Name) -> Out:
        name = node.last
        dialect = active()
        if name in C_NAMES:
            return C_NAMES[name], P.ATOM
        if name in dialect.functions:
            return dialect.functions[name], P.ATOM
        if name in dialect.constants:
            return dialect.constants[name], P.ATOM
        if name in taken():
            return name, P.ATOM
        inherited = dialect.member_pattern.match(name) or name in INHERITED
        if self.scope.klass and inherited and self.derives_from_library():
            return f"self.{name}", P.POSTFIX
        return self.host_name([name]) + self.targs(node.targs), P.POSTFIX

    def inherited_call(self, func: Name) -> str | None:
        """What ``f`` names, called in a method of a class derived from a library's class: as
        C++ looks in the class before the namespace, the object's own ``f`` - its class's, or
        one the library's base gives it - if it has one, else the library's function ``f``.
        ``Base::f`` of a library's base is ``Base``'s ``f`` called on the object."""
        function = self.scope.function
        if not self.scope.klass or function is None or getattr(function, "static", False):
            return None
        if func.targs or not self.derives_from_library():
            return None
        if len(func.parts) == 1:
            return self._inherited_name(func.last)
        owner = func.parts[:-1]
        if "::".join(owner) not in self.library_bases():
            return None
        return f"base_member({self.host_name(owner)}, {func.last!r}, self)"

    def _inherited_name(self, name: str) -> str | None:
        dialect = active()
        known = (C_NAMES, dialect.functions, dialect.constants, INHERITED)
        if any(name in names for names in known) or name in taken():
            return None
        if dialect.member_pattern.match(name):
            return None  # a member of the library's class's, which ``self`` has
        return f"inherited(self, {name!r}, {dialect.host})"

    def library_bases(self) -> set[str]:
        """The library's classes the class whose method is being written derives from."""
        found: set[str] = set()
        queue = [self.scope.klass]
        seen: set[str] = set()
        while queue:
            klass = queue.pop()
            if klass is None or klass in seen:
                continue
            seen.add(klass)
            info = self.program.classes.get(klass)
            for base in info.decl.bases if info is not None else ():
                name = base.ctype.name
                (queue.append if name in self.program.classes else found.add)(name)
        return found

    def derives_from_library(self) -> bool:
        """Is the class whose method is being written derived, somewhere up, from a class of
        the library's - one the program did not declare?"""
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

    def spelled(self, ctype: CType) -> str:
        """``ctype`` as a library's template is given it: instances of the program's templates
        by their C++ names."""
        return type_text(ctype, self.program.templates.spellings)

    def targs(self, targs: list[Any] | None) -> str:
        """``<double, 3>`` as the subscript ``['double', 3]`` a library's template takes."""
        if not targs:
            return ""
        items = [
            repr(self.spelled(arg)) if isinstance(arg, CType) else self.value(arg) for arg in targs
        ]
        return "[" + ", ".join(items) + "]"

    # -- literals ------------------------------------------------------------

    def literal(self, node: Literal) -> Out:
        return _LITERALS[node.kind](node), P.ATOM


#: The operators a constant condition is written with, as Python computes them.
CONSTANT_OPERATIONS: dict[str, Callable[[Any, Any], Any]] = {
    "&&": lambda a, b: bool(a and b),
    "||": lambda a, b: bool(a or b),
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}


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


#: How each kind of expression gives its constant value, if it has one.
_CONSTANTS: dict[type, Callable[[NameEmitter, Any], Any]] = {
    Literal: NameEmitter._constant_literal,
    Name: NameEmitter._constant_name,
    SizeOf: NameEmitter._constant_size,
    Unary: NameEmitter._constant_unary,
    Binary: NameEmitter._constant_binary,
}
