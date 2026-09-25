"""What C++ type an expression has, as far as the macro says: the type-inference pass.

C's arithmetic depends on types Python does not track: ``i / 2`` truncates
when ``i`` is an ``int``. The translator works out the type of every
expression it can - from declarations, literals, casts, the functions the
macro defines and the ``<cmath>`` ones - and ``None`` where it cannot, which
is most of ROOT's methods; there the runtime decides (:func:`div`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, ClassVar

from .base import EmitterBase
from .ctype import FLOATING, INTEGRAL, CType
from .nodes import (
    Assign,
    Binary,
    Call,
    Cast,
    Comma,
    Expr,
    Index,
    InitList,
    Literal,
    Member,
    Name,
    New,
    SizeOf,
    Ternary,
    This,
    Unary,
)
from .symbols import Symbol

__all__ = ["Inference", "DOUBLE", "INT", "BOOL", "arithmetic_result"]

DOUBLE = CType("double")
INT = CType("int")
BOOL = CType("bool")

#: The results of the C and C++ functions whose return type the translation needs.
RETURNS = {
    **dict.fromkeys(
        """sqrt cbrt exp exp2 expm1 log log10 log2 log1p pow sin cos tan asin acos atan atan2
        sinh cosh tanh asinh acosh atanh fabs floor ceil trunc round fmod hypot erf erfc
        tgamma lgamma atof stod stof copysign fmin fmax""".split(),
        DOUBLE,
    ),
    **dict.fromkeys("atoi atol stoi strcmp strncmp printf sprintf rand abs".split(), INT),
    "strlen": CType("unsigned long"),
    "isnan": BOOL,
    "isinf": BOOL,
    "isfinite": BOOL,
    "to_string": CType("std::string"),
    "Form": CType("char", pointer=1),
}

#: The standard's smart pointers and their makers, and the smart pointer each gives.
SMART_MAKERS = {
    "unique_ptr": "std::unique_ptr",
    "make_unique": "std::unique_ptr",
    "shared_ptr": "std::shared_ptr",
    "make_shared": "std::shared_ptr",
}

#: TMath's functions that return an integer; the rest of TMath returns a double.
TMATH_INTEGERS = frozenset(
    "Nint Abs Sign Min Max Floor Ceil FloorNint CeilNint BinarySearch LocMin LocMax "
    "Hypot Even Odd Permute".split()
)

#: The rank of each integer type in C's usual arithmetic conversions.
RANKS = {name: bits + (1 if name.startswith("unsigned") else 0) for name, bits in INTEGRAL.items()}


def arithmetic_result(left: CType | None, right: CType | None) -> CType | None:
    """The type C gives ``left op right`` for an arithmetic ``op``: the usual conversions."""
    if left is None or right is None or not (left.scalar and right.scalar):
        return None
    if left.floating or right.floating:
        widest = max(FLOATING.get(left.name, 0), FLOATING.get(right.name, 0))
        return CType("float") if widest == 32 else DOUBLE
    rank = max(RANKS[left.name], RANKS[right.name], RANKS["int"])
    for name, value in RANKS.items():
        if value == rank:
            return CType(name)
    return INT  # pragma: no cover - every rank is some type's


class Inference(EmitterBase):
    """The emitter's type pass: :meth:`typeof` of any expression, or ``None``."""

    def symbol(self, name: Name) -> Symbol | None:
        raise NotImplementedError

    def typeof(self, node: Any) -> CType | None:
        rule = self._RULES.get(type(node))
        return rule(self, node) if rule is not None else None

    def _type_literal(self, node: Literal) -> CType | None:
        if node.ctype == "char*":
            return CType("char", pointer=1, const=True)
        if node.kind == "null":
            return CType("nullptr_t", pointer=1)
        return CType(node.ctype)

    def _type_name(self, node: Name) -> CType | None:
        symbol = self.symbol(node)
        if symbol is None or symbol.ctype is None:
            return None
        return symbol.ctype.value() if not symbol.ctype.dims else symbol.ctype

    def _type_unary(self, node: Unary) -> CType | None:
        inner = self.typeof(node.operand)
        if node.op == "!":
            return BOOL
        if node.op == "*":
            return inner.element() if inner is not None else None
        if node.op == "&":
            return inner.pointed() if inner is not None else None
        if node.op in ("-", "+", "~"):
            return arithmetic_result(inner, INT) if inner is not None and inner.scalar else inner
        return inner

    def _type_binary(self, node: Binary) -> CType | None:
        if node.op in ("==", "!=", "<", ">", "<=", ">=", "&&", "||"):
            return BOOL
        left, right = self.typeof(node.left), self.typeof(node.right)
        if node.op in ("<<", ">>"):
            return arithmetic_result(left, INT) if left is not None and left.scalar else left
        if node.op in ("+", "-") and left is not None and (left.is_pointer or left.is_array):
            return left
        return arithmetic_result(left, right)

    def _type_assign(self, node: Assign) -> CType | None:
        return self.typeof(node.target)

    def _type_ternary(self, node: Ternary) -> CType | None:
        yes, no = self.typeof(node.yes), self.typeof(node.no)
        if yes is not None and no is not None and yes.scalar and no.scalar:
            return arithmetic_result(yes, no)
        return yes if yes is not None and no is not None and yes.name == no.name else None

    def _type_cast(self, node: Cast) -> CType | None:
        return node.ctype

    def _type_sizeof(self, node: SizeOf) -> CType | None:
        return CType("unsigned long")

    def _type_new(self, node: New) -> CType | None:
        return node.ctype.pointed()

    def _type_index(self, node: Index) -> CType | None:
        inner = self.typeof(node.obj)
        if inner is None:
            return None
        if inner.is_string and not inner.dims:
            return CType("char")
        if inner.is_array or inner.is_pointer:
            return inner.element()
        return None

    def _type_comma(self, node: Comma) -> CType | None:
        return self.typeof(node.items[-1])

    def _type_this(self, node: This) -> CType | None:
        return CType(self.scope.klass, pointer=1) if self.scope.klass else None

    def _type_init_list(self, node: InitList) -> CType | None:
        return node.ctype

    def _type_call(self, node: Call) -> CType | None:
        func = node.func
        if isinstance(func, Name):
            return self._named_call(func, node)
        if isinstance(func, Member):
            return self._method_call(func)
        return None

    def _named_call(self, func: Name, node: Call) -> CType | None:
        symbol = self.symbol(func)
        if symbol is not None and symbol.kind == "class":
            return CType(symbol.name)
        if symbol is not None and symbol.kind in ("function", "method", "static"):
            funcs = self.function_named(symbol)
            return funcs[0].returns if funcs else None
        if symbol is not None:
            return None
        return self._library_call(func, node)

    def _library_call(self, func: Name, node: Call) -> CType | None:
        parts = func.parts[1:] if func.parts[0] == "std" else func.parts
        if len(parts) == 2 and parts[0] == "TMath":
            return INT if parts[1] in TMATH_INTEGERS else DOUBLE
        if len(parts) != 1:
            return None
        name = parts[0]
        smart = SMART_MAKERS.get(name)
        if smart is not None:
            return CType(smart, list(func.targs or []))
        return self._numeric_call(name, node)

    def _numeric_call(self, name: str, node: Call) -> CType | None:
        """The type of ``min``, ``max`` and ``abs`` - their arguments' - and the fixed ones."""
        if name in ("min", "max") and len(node.args) == 2:
            return arithmetic_result(self.typeof(node.args[0]), self.typeof(node.args[1]))
        if name == "abs" and node.args:
            return self.typeof(node.args[0])
        return RETURNS.get(name)

    def _method_call(self, func: Member) -> CType | None:
        owner = self.typeof(func.obj)
        if owner is not None and owner.is_string and func.name in ("size", "length"):
            return CType("unsigned long")
        if owner is None:
            return None
        info = self.program.classes.get(owner.name)
        methods = info.methods.get(func.name, []) if info is not None else []
        return methods[0].returns if methods else None

    def _type_member(self, node: Member) -> CType | None:
        owner = self.typeof(node.obj)
        if owner is None:
            return None
        info = self.program.classes.get(owner.name)
        if info is None:
            return None
        found = info.fields.get(node.name) or info.statics.get(node.name)
        return found.ctype if found is not None else None

    def function_named(self, symbol: Symbol) -> list[Any]:
        """The functions a function or method symbol stands for."""
        raise NotImplementedError

    _RULES: ClassVar[dict[type, Callable[[Inference, Any], CType | None]]] = {
        Literal: _type_literal,
        Name: _type_name,
        Unary: _type_unary,
        Binary: _type_binary,
        Assign: _type_assign,
        Ternary: _type_ternary,
        Cast: _type_cast,
        SizeOf: _type_sizeof,
        New: _type_new,
        Index: _type_index,
        Comma: _type_comma,
        This: _type_this,
        InitList: _type_init_list,
        Call: _type_call,
        Member: _type_member,
    }

    def is_integral(self, node: Expr) -> bool:
        found = self.typeof(node)
        return found is not None and found.integral

    def is_floating(self, node: Expr) -> bool:
        found = self.typeof(node)
        return found is not None and found.floating

    def is_pointer(self, node: Expr) -> bool:
        found = self.typeof(node)
        return found is not None and (found.is_object_pointer or found.name == "nullptr_t")

    @staticmethod
    def deref_type(ctype: CType) -> CType:
        return replace(ctype, pointer=max(ctype.pointer - 1, 0))
