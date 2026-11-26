"""What C++ type an expression has, as far as the macro says: the type-inference pass.

C's arithmetic depends on types Python does not track: ``i / 2`` truncates
when ``i`` is an ``int``. The translator works out the type of every
expression it can - from declarations, literals, casts, the functions the
program defines and the ``<cmath>`` ones - and ``None`` where it cannot, which
is most of a library's methods; there the runtime decides (:func:`div`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, ClassVar

from .base import EmitterBase
from .ctype import FLOATING, INTEGRAL, CType, canonical
from .dialect import active
from .freeops import UNARY as FREE_UNARY
from .freeops import FreeOperator, choose
from .limits import is_limit, limit_type
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
    TypeId,
    Unary,
)
from .symbols import Symbol
from .traits import is_trait

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
    "string": CType("std::string"),
    #: The C string searches give back text, or C's null pointer - which is ``None`` here, so
    #: ``strstr(s, t) != 0`` is a test of whether the text is there, as ROOT's macros write it.
    **dict.fromkeys("strstr strchr strrchr getenv".split(), CType("char", pointer=1)),
}

#: The standard's smart pointers and their makers, and the smart pointer each gives.
SMART_MAKERS = {
    "unique_ptr": "std::unique_ptr",
    "make_unique": "std::unique_ptr",
    "shared_ptr": "std::shared_ptr",
    "make_shared": "std::shared_ptr",
}

#: The rank of each integer type in C's usual arithmetic conversions.
RANKS = {name: bits + (1 if name.startswith("unsigned") else 0) for name, bits in INTEGRAL.items()}


#: The operators whose result is true or false - of each element, for an element-wise vector.
LOGICAL = frozenset({"==", "!=", "<", ">", "<=", ">=", "&&", "||"})


def is_elementwise(ctype: CType | None) -> bool:
    """Is ``ctype`` the dialect's element-wise vector held by value - or one of its aliases -
    whose operators work element by element?"""
    vector = active().vector
    if ctype is None or ctype.pointer or vector is None:
        return False
    return ctype.name.split("::")[-1].startswith(vector)


def _vector_result(op: str, left: CType | None, right: CType | None) -> CType | None:
    """What ``left op right`` is when either is an element-wise vector: one, of each element's
    result."""
    vector = left if is_elementwise(left) else right if is_elementwise(right) else None
    if vector is None or op in ("<<", ">>"):  # a stream's, not a vector's
        return None
    return CType(str(active().vector_bool)) if op in LOGICAL else vector


def _operated(op: str, left: CType | None, right: CType | None) -> CType | None:
    """What an arithmetic, shift or pointer ``left op right`` is."""
    if op in ("<<", ">>"):
        return arithmetic_result(left, INT) if left is not None and left.scalar else left
    if op in ("+", "-") and left is not None and (left.is_pointer or left.is_array):
        return left
    return arithmetic_result(left, right)


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


def _named_type(text: str) -> CType:
    """The type a dialect's ``method_returns`` names, as a declaration would spell it.

    A table entry is a class's name (``TMatrixD``) or a pointer to one written with the ``*`` a
    declaration has - ``Double_t*``, which a graph's ``GetX`` answers - and a pointer is what
    makes a range of it: ``std::is_sorted(g.GetX(), g.GetX() + n)`` needs the return type to say
    so, or the range is refused. A typedef is looked through as a declaration's is, so an entry
    naming ``Int_t`` is the ``int`` every arithmetic rule here is written against.
    """
    starred = text.rstrip().endswith("*")
    base = text.rstrip().rstrip("*").strip() if starred else text
    return CType(canonical(base), pointer=int(starred))


def _made_field(func: Member) -> CType | None:
    """A dialect's field maker ``MakeField<T>``: the ``shared_ptr<T>`` a value is written
    through."""
    first = func.targs[0] if func.targs else None
    if func.name not in active().field_makers or not isinstance(first, CType):
        return None
    return replace(first, pointer=first.pointer + 1)


#: The C++ names a pointer to text is spelled with: a null ``char *`` is ``None``, not 0.
CHAR_NAMES = ("char", "signed char", "unsigned char")

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
        if symbol is None and is_limit(node.parts, node.targs):
            return limit_type(node.targs[0] if node.targs else None, node.last)
        if symbol is None and is_trait(node):
            return BOOL
        if symbol is None or symbol.ctype is None:
            return None
        return symbol.ctype.value() if not symbol.ctype.dims else symbol.ctype

    def free_operator(self, op: str, operands: list[CType | None]) -> FreeOperator | None:
        """The macro's own ``operator<op>`` defined outside a class that operands of these
        types call, if there is one."""
        candidates = self.free_ops.get("operator" + op)
        if not candidates:
            return None
        return choose(candidates, operands, self.program.classes)

    def _free_result(self, op: str, operands: list[CType | None]) -> CType | None:
        """What the macro's own operator called by ``op`` returns, ``auto`` being unknown."""
        chosen = self.free_operator(op, operands)
        returns = chosen.func.returns if chosen is not None else None
        return returns.value() if returns is not None and not returns.is_auto else None

    def _type_unary(self, node: Unary) -> CType | None:
        inner = self.typeof(node.operand)
        own = self._free_result(node.op, [inner]) if node.op in FREE_UNARY else None
        return own if own is not None else self._builtin_unary(node, inner)

    def _builtin_unary(self, node: Unary, inner: CType | None) -> CType | None:
        if node.op == "!":
            return BOOL
        if node.op == "*":
            return self._dereferenced_type(inner)
        if node.op == "&":
            return inner.pointed() if inner is not None else None
        if node.op in ("-", "+", "~"):
            return arithmetic_result(inner, INT) if inner is not None and inner.scalar else inner
        return inner

    def _dereferenced_type(self, inner: CType | None) -> CType | None:
        if inner is not None and (inner.pointer or inner.dims):
            return inner.element()
        held = inner.args[0] if inner is not None and inner.is_smart and inner.args else None
        if isinstance(held, CType):
            return held  # ``*p`` of a ``std::unique_ptr<T>`` is the ``T`` it holds
        return self.reader_element(inner, "value") or inner

    def _type_binary(self, node: Binary) -> CType | None:
        left, right = self.typeof(node.left), self.typeof(node.right)
        own = self._free_result(node.op, [left, right])
        if own is not None:
            return own
        vector = _vector_result(node.op, left, right)
        if vector is not None:
            return vector
        if node.op in LOGICAL:
            return BOOL
        if node.op == "+" and left is not None and left.name == "std::string" and left.is_string:
            return left  # ``s + x``: a ``std::string``, joined to whatever ``x`` is written as
        return _operated(node.op, left, right)

    def _type_assign(self, node: Assign) -> CType | None:
        return self.typeof(node.target)

    def _type_ternary(self, node: Ternary) -> CType | None:
        yes, no = self.typeof(node.yes), self.typeof(node.no)
        if yes is not None and no is not None and yes.scalar and no.scalar:
            # one type both ways is that type - two chars a char, not an int
            return yes if yes.name == no.name else arithmetic_result(yes, no)
        return yes if yes is not None and no is not None and yes.name == no.name else None

    def _type_cast(self, node: Cast) -> CType | None:
        return node.ctype

    def _type_sizeof(self, node: SizeOf) -> CType | None:
        return CType("unsigned long")

    def _type_typeid(self, node: TypeId) -> CType | None:
        return CType("std::type_info", const=True)

    def _type_new(self, node: New) -> CType | None:
        """``new T`` is a ``T*``; ``new T[n][m]``, rows of an array, is taken as a ``T**``:
        indexed twice, as the array of arrays it is."""
        ctype = node.ctype
        if ctype.dims:
            return CType(ctype.name, pointer=ctype.pointer + 1 + len(ctype.dims))
        return ctype.pointed()

    def _type_index(self, node: Index) -> CType | None:
        inner = self.typeof(node.obj)
        if inner is None:
            return None
        if inner.is_string and not inner.dims:
            return CType("char")
        if inner.is_array or inner.is_pointer:
            return inner.element()
        return self.reader_element(inner, "array")

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
        if symbol is None and is_limit(func.parts, func.targs):
            return limit_type(func.targs[0] if func.targs else None, func.last)
        if symbol is None:
            return self._library_call(func, node)
        if symbol.kind == "class":
            return CType(symbol.name)
        if symbol.kind in ("function", "method", "static", "member"):
            return self._function_result(func, node, symbol)
        return None

    def _function_result(self, func: Name, node: Call, symbol: Symbol) -> CType | None:
        """What a function of the program's returns - an instance's, for a template."""
        funcs = self.function_named(symbol)
        if funcs and funcs[0].template is not None:
            return self.instance_returns(func, node, symbol)
        return funcs[0].returns if funcs else None

    def _library_call(self, func: Name, node: Call) -> CType | None:
        parts = func.parts[1:] if func.parts[0] == "std" else func.parts
        numeric = active().numeric_namespaces.get(parts[0]) if len(parts) == 2 else None
        if numeric is not None:
            integers, overloaded = numeric
            if parts[1] in overloaded:
                return self._overloaded(node)
            return INT if parts[1] in integers else DOUBLE
        if len(parts) != 1:
            # ``T::f(...)``, a static call written through its class, asks the same question
            # as ``t.f(...)`` on one of its objects - what the class gives back by value.
            kept = active().method_returns.get((parts[-2], parts[-1]))
            return CType(kept) if kept is not None else None
        name = parts[0]
        smart = SMART_MAKERS.get(name)
        if smart is not None:
            return CType(smart, list(func.targs or []))
        return self._numeric_call(name, node)

    def _overloaded(self, node: Call) -> CType | None:
        """``Abs(x)``, ``Min(a, b)`` of a numeric namespace: the type of the argument, or the
        arguments' common type, which the overload chosen returns."""
        if len(node.args) == 1:
            return self.typeof(node.args[0])
        if len(node.args) == 2:
            return arithmetic_result(self.typeof(node.args[0]), self.typeof(node.args[1]))
        return None

    def _numeric_call(self, name: str, node: Call) -> CType | None:
        """The type of ``min``, ``max`` and ``abs`` - their arguments' - and the fixed ones."""
        if name in ("min", "max") and len(node.args) == 2:
            return arithmetic_result(self.typeof(node.args[0]), self.typeof(node.args[1]))
        if name == "abs" and node.args:
            return self.typeof(node.args[0])
        found = RETURNS.get(name)
        return found if found is not None else self._class_or_return(name, node)

    def _class_or_return(self, name: str, node: Call) -> CType | None:
        """What a name the program did not declare is: the dialect's known return, or - a name
        ROOT spells as a class - the class whose object the call makes, which is what an
        initialisation from it then converts."""
        named = active().returns.get(name)
        if named is not None or not active().class_pattern.match(name):
            return named
        targs = list(node.func.targs or []) if isinstance(node.func, Name) else []
        return CType(name, targs)

    def reader_element(self, ctype: CType | None, kind: str) -> CType | None:
        """What a dialect's reader reads, by ``kind``: ``*r`` of a value reader ``R<T>``, or
        ``r[i]`` of an array reader - ``T`` - or ``None`` when ``ctype`` is no such reader."""
        # Raw pointers/arrays dereference to the reader itself before its operator* applies.
        if ctype is None or ctype.pointer or ctype.dims:
            return None
        dialect = active()
        readers = dialect.value_readers if kind == "value" else dialect.array_readers
        if ctype.name not in readers:
            return None
        if ctype.name in self.program.classes or len(ctype.args) != 1:
            return None
        argument = ctype.args[0]
        return argument if isinstance(argument, CType) else None

    def reader_method(self, func: Member, kind: str) -> CType | None:
        owner = self.typeof(func.obj)
        if owner is not None and func.arrow and owner.pointer == 1:
            owner = owner.element()
        return self.reader_element(owner, kind)

    def _reader_method_type(self, func: Member) -> CType | None:
        read = self.reader_method(func, "value") if func.name == "Get" else None
        if read is not None:
            return read.pointed()
        read = self.reader_method(func, "array") if func.name == "At" else None
        if read is not None:
            return read
        return None

    def _method_call(self, func: Member) -> CType | None:
        owner = self.typeof(func.obj)
        read = self._reader_method_type(func)
        if read is not None:
            return read
        if owner is not None and owner.is_string and func.name in ("size", "length"):
            return CType("unsigned long")
        return self._class_method_type(owner, func)

    def _class_method_type(self, owner: CType | None, func: Member) -> CType | None:
        held = self._smart_pointer(owner, func.name)
        if held is not None:
            return held
        info = self.program.classes.get(owner.name) if owner is not None else None
        if info is None:
            kept = active().method_returns.get((owner.name, func.name)) if owner else None
            return _named_type(kept) if kept is not None else _made_field(func)
        methods = info.methods.get(func.name, [])
        return methods[0].returns if methods else None

    def _smart_pointer(self, owner: CType | None, name: str) -> CType | None:
        """``p.get()`` and ``p.release()``: the pointer to what a smart pointer holds.

        The type matters where the answer is used as a pointer - ``buf.get() + n`` is the
        buffer from ``n`` on, not ``n`` added to every element of it - so it is the held
        type with one more level of pointer.
        """
        if owner is None or not owner.is_smart or name not in ("get", "release"):
            return None
        held = next(iter(owner.args), None)
        assert isinstance(held, CType), "a smart pointer's argument is the type it holds"
        return replace(held, pointer=held.pointer + 1, dims=[])

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

    def instance_returns(self, func: Name, node: Call, symbol: Symbol | None) -> CType | None:
        """What a call of a function template returns, when its arguments are known."""
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
        TypeId: _type_typeid,
        New: _type_new,
        Index: _type_index,
        Comma: _type_comma,
        This: _type_this,
        InitList: _type_init_list,
        Call: _type_call,
        Member: _type_member,
    }

    def is_pointer(self, node: Expr) -> bool:
        """Is this a pointer whose null this kit holds as ``None`` - an object's, a ``char *``'s.

        A ``char *`` is text, and C's null pointer is the ``None`` beside it, so
        ``strstr(s, t) != 0`` - which ROOT's own macros select on - asks whether the text is
        there at all. The pointers that are addresses stay numbers, and compare as numbers.
        """
        found = self.typeof(node)
        if found is None:
            return False
        if found.is_object_pointer or found.name == "nullptr_t":
            return True
        return found.is_pointer and found.name in CHAR_NAMES
