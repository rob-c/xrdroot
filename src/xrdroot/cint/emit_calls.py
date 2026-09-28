"""Calls, and the stores C converts: where a macro meets ROOT, the library and itself.

A call to one of the macro's functions is a Python call, with a variable
handed to a ``double&`` parameter handed as its cell. A call to a ROOT
method is ``obj.Method(args)`` - ROOT's names are kept exactly - with the
variables ROOT writes through (``Rannor(px, py)``) handed as cells. The
standard library's calls that have no pyroot counterpart are rewritten:
``std::string``'s members become Python's, ``std::make_unique<T>(a)`` is
``T(a)``, ``std::sort(v.begin(), v.end())`` sorts ``v`` in place.

:meth:`CallEmitter.store` is C's conversion on assignment, written out: an
``int`` stored from a ``double`` is truncated, a ``float`` rounded, an
``unsigned`` wrapped, an object copied.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from .base import Out, P
from .ctype import CType
from .cursor import looks_like_type
from .emit_expr import ExprEmitter, zero
from .emit_names import type_text
from .literals import SUFFIXES
from .nodes import Assign, Binary, Call, Expr, Index, InitList, Literal, Member, Name, Unary

__all__ = ["CallEmitter", "WRAPS"]

#: The integer types a store must wrap to their width, and the runtime helper that does.
WRAPS = {
    "unsigned int": "u32",
    "unsigned long": "u64",
    "unsigned long long": "u64",
    "unsigned short": "u16",
    "unsigned char": "u8",
    "short": "i16",
    "char": "i8",
    "signed char": "i8",
}

#: A container's iterator functions, and the index each stands for in a slice.
ITERATOR_ENDS = {"begin": "0", "cbegin": "0", "end": "None", "cend": "None"}

#: ``std::numeric_limits<T>``'s members, for the floating types and the integer ones.
LIMITS = {
    ("double", "max"): "1.7976931348623157e+308",
    ("double", "min"): "2.2250738585072014e-308",
    ("double", "lowest"): "-1.7976931348623157e+308",
    ("double", "epsilon"): "2.220446049250313e-16",
    ("double", "infinity"): "float('inf')",
    ("double", "quiet_NaN"): "float('nan')",
    ("float", "max"): "3.4028234663852886e+38",
    ("float", "min"): "1.1754943508222875e-38",
    ("float", "lowest"): "-3.4028234663852886e+38",
    ("float", "epsilon"): "1.1920928955078125e-07",
    ("float", "infinity"): "float('inf')",
    ("float", "quiet_NaN"): "float('nan')",
    ("int", "max"): "2147483647",
    ("int", "min"): "-2147483648",
    ("int", "lowest"): "-2147483648",
    ("unsigned int", "max"): "4294967295",
    ("long", "max"): "9223372036854775807",
    ("long", "min"): "-9223372036854775808",
    ("long long", "max"): "9223372036854775807",
    ("unsigned long", "max"): "18446744073709551615",
}


def _tuple(items: list[str]) -> str:
    """Python's spelling of a tuple of ``items``: ``()``, ``(a,)``, ``(a, b)``."""
    return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"


def _plain(node: Expr) -> bool:
    """An argument evaluated twice to the same effect: a name, a literal, arithmetic on them."""
    if isinstance(node, Binary):
        return _plain(node.left) and _plain(node.right)
    return isinstance(node, (Name, Literal))


def _lvalue(node: Expr) -> bool:
    return isinstance(node, (Name, Member, Index)) or (isinstance(node, Unary) and node.op == "*")


class CallEmitter(ExprEmitter):
    """The emitter's layer for calls and converting stores."""

    # -- arguments -------------------------------------------------------------

    def arguments(self, node: Call) -> str:
        """A call's arguments, with those passed by reference passed as something writable."""
        return self.argument_list(node.args, self.program.reference_positions(node))

    def constructor_arguments(self, ctype: CType, args: list[Expr]) -> str:
        """Arguments to a constructor of the macro's own, references passed as writable."""
        positions = self.program.constructor_positions(ctype.name, len(args))
        return self.argument_list(args, positions)

    def argument_list(self, args: list[Expr], at: tuple[int, ...]) -> str:
        positions = set(at)
        items = []
        for index, arg in enumerate(args):
            if index in positions and _lvalue(arg) and not self._object(arg):
                items.append(self.reference(arg))
            else:
                items.append(self.value(arg))
        return ", ".join(items)

    def _object(self, arg: Expr) -> bool:
        found = self.typeof(arg)
        return found is not None and found.is_class and not found.pointer

    # -- the calls -------------------------------------------------------------

    def call(self, node: Call) -> Out:
        func = node.func
        if isinstance(func, Name):
            return self.named_call(func, node)
        if isinstance(func, Member):
            return self.method_call(func, node)
        return f"{self.at(func, P.POSTFIX)}({self.arguments(node)})", P.POSTFIX

    def named_call(self, func: Name, node: Call) -> Out:
        symbol = self.symbol(func)
        if symbol is not None:
            return f"{self.use(symbol)[0]}({self.arguments(node)})", P.POSTFIX
        special = self._LIBRARY.get(func.last)
        if special is not None and (len(func.parts) == 1 or func.parts[0] in ("std", "TString")):
            found = special(self, func, node)
            if found is not None:
                return found
        return f"{self.library(func)[0]}({self.arguments(node)})", P.POSTFIX

    def _make(self, func: Name, node: Call) -> Out | None:
        if not func.targs or not isinstance(func.targs[0], CType):
            return None
        return f"{self.class_expr(func.targs[0])}({self.arguments(node)})", P.POSTFIX

    def _smart(self, func: Name, node: Call) -> Out | None:
        """``std::unique_ptr<T>(p)``: the pointer it holds, which is all a Python name is."""
        if not node.args:
            return "None", P.ATOM
        return self.expr(node.args[0])

    def _limits(self, func: Name, node: Call) -> Out | None:
        if len(func.parts) < 2 or func.parts[-2] != "numeric_limits" or not func.targs:
            return None
        kind = type_text(func.targs[0])
        value = LIMITS.get((kind, func.last))
        if value is None:
            raise self.refuse(f"std::numeric_limits<{kind}>::{func.last}()", node)
        return value, P.ATOM

    def _format(self, func: Name, node: Call) -> Out | None:
        if func.parts != ["TString", "Format"]:
            return None
        return f"ROOT.TString(cformat({self.arguments(node)}))", P.POSTFIX

    def _exit(self, func: Name, node: Call) -> Out | None:
        return f"c_exit({self.arguments(node)})", P.POSTFIX

    def _suffixed(self, func: Name, node: Call) -> Out | None:
        """``0.1_normal``, ``100us``: the value the library's ``operator""`` makes of a number."""
        suffix = func.last[len('operator""') :]
        return f"user_literal({suffix!r}, {self.value(node.args[0])})", P.POSTFIX

    def _unsupported(self, func: Name, node: Call) -> Out | None:
        raise self.refuse(f"{func.text}() where its result is used", node)

    def _sort(self, func: Name, node: Call) -> Out | None:
        helper = "sort_range" if func.last in ("sort", "stable_sort") else "reverse_range"
        if len(node.args) < 2:
            raise self.refuse(f"std::{func.last} without a range", node)
        container, start, stop = self.iterator_range(node.args[0], node.args[1], node)
        extra = "".join(f", {self.value(arg)}" for arg in node.args[2:3])
        return f"{helper}({container}, {start}, {stop}{extra})", P.POSTFIX

    def _accumulate(self, func: Name, node: Call) -> Out | None:
        if len(node.args) != 3:
            raise self.refuse("std::accumulate with an operation of its own", node)
        container, start, stop = self.iterator_range(node.args[0], node.args[1], node)
        stop_text = "" if stop == "None" else stop
        items = f"{container}[{start}:{stop_text}]"
        return f"sum({items}, {self.value(node.args[2])})", P.POSTFIX

    def iterator_range(self, first: Expr, last: Expr, node: Expr) -> tuple[str, str, str]:
        """``v.begin(), v.end()`` or ``a, a + n``: the container, start and stop they mean."""
        container, start = self._iterator(first, node, "begin")
        other, stop = self._iterator(last, node, "end")
        if other != container:
            raise self.refuse("a range whose two ends are in different containers", node)
        return container, start, stop

    def _iterator(self, it: Expr, node: Expr, end: str) -> tuple[str, str]:
        ends = self._container_end(it)
        if ends is not None:
            return ends
        if isinstance(it, Binary) and it.op == "+":
            base, _ = self._iterator(it.left, node, end)
            return base, self.value(it.right)
        found = self.typeof(it)
        if found is not None and (found.is_array or found.is_pointer):
            return self.value(it), "0"
        raise self.refuse("an iterator this translator cannot follow back to its container", node)

    def _container_end(self, it: Expr) -> tuple[str, str] | None:
        """``v.begin()`` as ``(v, 0)`` and ``v.end()`` as ``(v, None)``: where a range starts."""
        if not isinstance(it, Call) or it.args or not isinstance(it.func, Member):
            return None
        position = ITERATOR_ENDS.get(it.func.name)
        return (self.value(it.func.obj), position) if position is not None else None

    _LIBRARY: ClassVar[dict[str, Callable[[CallEmitter, Name, Call], Out | None]]] = {
        "unique_ptr": _smart,
        "shared_ptr": _smart,
        "make_unique": _make,
        "make_shared": _make,
        "max": _limits,
        "min": _limits,
        "lowest": _limits,
        "epsilon": _limits,
        "infinity": _limits,
        "quiet_NaN": _limits,
        "Format": _format,
        "exit": _exit,
        "abort": _exit,
        "sort": _sort,
        "stable_sort": _sort,
        "reverse": _sort,
        "accumulate": _accumulate,
        "sprintf": _unsupported,
        "snprintf": _unsupported,
        "strcpy": _unsupported,
        "strcat": _unsupported,
        "swap": _unsupported,
        "max_element": _unsupported,
        "min_element": _unsupported,
        "find_if": _unsupported,
        "getline": _unsupported,
        **dict.fromkeys(['operator""' + suffix for suffix in SUFFIXES], _suffixed),
    }

    # -- methods -----------------------------------------------------------------

    def method_call(self, func: Member, node: Call) -> Out:
        owner = self.typeof(func.obj)
        if owner is not None:
            special = self._typed_method(owner, func, node)
            if special is not None:
                return special
        return f"{self.value(func)}({self.arguments(node)})", P.POSTFIX

    def _typed_method(self, owner: CType, func: Member, node: Call) -> Out | None:
        """A member of a string, a smart pointer or an exception, rewritten as Python's."""
        if owner.is_string and not owner.dims:
            return self._string_method(func, node)
        if owner.is_smart and not func.arrow:
            return self._smart_method(func, node)
        if func.name == "what" and "exception" in owner.name:
            return f"str({self.value(func.obj)})", P.POSTFIX
        return None

    def _string_method(self, func: Member, node: Call) -> Out:
        obj = self.value(func.obj)
        args = [self.value(arg) for arg in node.args]
        rewrite = STRING_METHODS.get(func.name)
        if rewrite is None:
            raise self.refuse(f"std::string::{func.name}()", node)
        return rewrite(obj, args), P.POSTFIX

    def _smart_method(self, func: Member, node: Call) -> Out:
        if func.name in ("get", "release", "operator->"):
            return self.expr(func.obj)
        raise self.refuse(f"the smart pointer's {func.name}() where its result is used", node)

    # -- assigning through a returned reference ------------------------------------

    def call_assignment(self, node: Assign) -> str:
        """``f(i) = v``: what the call returns a reference to, written through the runtime."""
        target = node.target
        assert isinstance(target, Call)
        if node.op != "=" and not all(_plain(arg) for arg in target.args):
            raise self.refuse("a compound assignment to a call whose arguments change things", node)
        value = self.assigned_value(node)
        func = target.func
        args = _tuple([self.value(arg) for arg in target.args])
        if isinstance(func, Member):
            return self._method_assignment(func, target, args, value)
        self._own_reference(func, "its operator()", node)
        if isinstance(func, Name) and self._names_class(func):
            # ``TMatrixDColumn(A, 0) = 1.0``: a temporary built, and assigned into.
            return f"assign_into({self.value(target)}, {value})"
        return f"assign_call({self.at(func, P.POSTFIX)}, {args}, {value})"

    def _method_assignment(self, func: Member, target: Call, args: str, value: str) -> str:
        """``obj.m(args) = v``: an element for ``at``, else through the runtime's ``Set``/copy."""
        self._own_reference(func.obj, f"its method {func.name}", target)
        obj = self.value(func.obj)
        if func.name == "at" and len(target.args) == 1:
            return f"set_item({obj}, {self.value(target.args[0])}, {value})"
        return f"assign_method({obj}, {func.name!r}, {args}, {value})"

    def _names_class(self, func: Name) -> bool:
        symbol = self.symbol(func)
        if symbol is not None:
            return symbol.kind == "class"
        return looks_like_type(func.parts, set(self.program.classes))

    def _own_reference(self, obj: Expr, what: str, node: Expr) -> None:
        """Refuse a reference one of the macro's own classes returns: Python returns a value."""
        found = self.typeof(obj)
        if found is not None and found.name in self.program.classes:
            why = f"assigning to what the macro's class {found.name} returns from {what}"
            raise self.refuse(why, node)

    # -- converting stores ---------------------------------------------------------

    def store(self, ctype: CType | None, node: Expr) -> str:
        """``node`` as stored into a ``ctype``: converted as C converts on assignment."""
        if ctype is None or ctype.is_array:
            return self.value(node)
        if isinstance(node, InitList) and node.ctype is None and ctype.scalar:
            return self.store(ctype, node.items[0]) if node.items else zero(ctype)
        if ctype.scalar:
            return self._arithmetic_store(ctype, node)
        if self._copied(ctype, node):
            return f"value_copy({self.value(node)})"
        return self.value(node)

    def _copied(self, ctype: CType, node: Expr) -> bool:
        """Is ``T b = a`` a copy of an object - which Python would otherwise share, not copy?"""
        if not ctype.is_class or ctype.pointer or ctype.reference or not _lvalue(node):
            return False
        found = self.typeof(node)
        return found is not None and found.is_class and not found.pointer

    def _arithmetic_store(self, ctype: CType, node: Expr) -> str:
        source = self.typeof(node)
        if ctype.is_bool:
            return self._bool_store(source, node)
        if ctype.integral:
            return self._integral_store(ctype, source, node)
        if ctype.name == "float":
            value = self.value(node)
            return value if self.single(node) else f"f32({value})"
        return self._double_store(source, node)

    def _bool_store(self, source: CType | None, node: Expr) -> str:
        if source is not None and source.is_bool:
            return self.value(node)
        return f"bool({self.condition(node)})"

    def _double_store(self, source: CType | None, node: Expr) -> str:
        if isinstance(node, Literal) and node.kind in ("int", "char"):
            return repr(float(node.value))
        if source is not None and source.integral:
            return f"float({self.value(node)})"
        return self.value(node)

    def _integral_store(self, ctype: CType, source: CType | None, node: Expr) -> str:
        text = self.value(node)
        if source is not None and source.name == ctype.name and source.scalar:
            return text
        if isinstance(node, Literal) and node.kind in ("int", "char", "bool"):
            return str(_wrapped(int(node.value), ctype))
        wrap = WRAPS.get(ctype.name)
        if wrap is not None:
            return f"{wrap}({text})"
        return text if _plain_integer(source) else f"int({text})"


def _plain_integer(source: CType | None) -> bool:
    return source is not None and source.integral and not source.is_bool


def _wrapped(value: int, ctype: CType) -> int:
    bits = {"u8": 8, "i8": 8, "u16": 16, "i16": 16, "u32": 32, "u64": 64}
    wrap = WRAPS.get(ctype.name)
    if wrap is None:
        return value
    width = bits[wrap]
    value &= (1 << width) - 1
    if wrap.startswith("i") and value >> (width - 1):
        value -= 1 << width
    return value


def _runtime_method(name: str) -> Callable[[str, list[str]], str]:
    """A string member the runtime has a function of the same name for, the string first."""
    return lambda obj, args: f"{name}({', '.join([obj, *args])})"


#: ``std::string``'s members, as the Python over a ``str`` that does what each does.
STRING_METHODS: dict[str, Callable[[str, list[str]], str]] = {
    "c_str": lambda obj, args: obj,
    "data": lambda obj, args: obj,
    "size": lambda obj, args: f"len({obj})",
    "length": lambda obj, args: f"len({obj})",
    "empty": lambda obj, args: f"(len({obj}) == 0)",
    "substr": lambda obj, args: f"substr({', '.join([obj, *args])})",
    "find": lambda obj, args: f"find({', '.join([obj, *args])})",
    "rfind": lambda obj, args: f"rfind({', '.join([obj, *args])})",
    **{
        name: _runtime_method(name)
        for name in ("find_first_of", "find_last_of", "find_first_not_of", "find_last_not_of")
    },
    "compare": lambda obj, args: f"strcmp({obj}, {args[0]})",
    "at": lambda obj, args: f"char_at({obj}, {args[0]})",
    "front": lambda obj, args: f"char_at({obj}, 0)",
    "back": lambda obj, args: f"char_at({obj}, len({obj}) - 1)",
    "Data": lambda obj, args: obj,
}
