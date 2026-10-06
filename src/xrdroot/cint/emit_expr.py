"""C++ operators as Python, with C's semantics written out where Python's differ.

The operators Python shares with C are written as they are, bracketed by
Python's precedence rather than C's. The ones that differ get the runtime's
help, chosen by the types :mod:`xrdroot.cint.infer` found: ``/`` of two
integers is ``idiv``, ``%`` is ``imod``, ``&&`` is ``and`` - wrapped in
``bool`` where its value, not its truth, is used - a pointer compared with
``0`` is compared with ``None``, ``p++`` inside an expression is a walrus.

Arithmetic C does in ``float`` - ``+``, ``-``, ``*`` and ``/`` of floats, or
of a float and an integer - is rounded to single precision after every
operation, as ROOT's compiled macros round it: ``pz = px*px + py*py`` of
three ``Float_t`` is ``f32(f32(px * px) + f32(py * py))``, not the sum in
double rounded once, which differs in the last bit a fifth of the time - and
``hsimple.C``'s profile errors, built on sums of ``pz`` squared, by more.
Python's double arithmetic on two single-precision values rounds exactly as
single precision would, so rounding each result is all it takes.
"""

from __future__ import annotations

import keyword
from collections.abc import Callable
from typing import Any, ClassVar

from .base import Out, P
from .ctype import CType
from .emit_names import NameEmitter, type_text
from .nodes import (
    Assign,
    Binary,
    Call,
    Cast,
    Comma,
    Delete,
    Expr,
    Index,
    InitList,
    Lambda,
    Literal,
    Member,
    Name,
    New,
    SizeOf,
    Ternary,
    This,
    Throw,
    Unary,
)
from .operators import DUNDERS, python_operator

__all__ = ["ExprEmitter", "COMPARISONS", "zero"]

#: C's comparison operators, which are Python's.
COMPARISONS = frozenset({"==", "!=", "<", ">", "<=", ">="})

#: The runtime helper for ``/`` and ``%`` by what the operands are known to be.
DIVISIONS = {
    ("/", "integral"): "idiv",
    ("/", "unknown"): "div",
    ("%", "integral"): "imod",
    ("%", "floating"): "fmod",
    ("%", "unknown"): "mod",
}

#: The operators whose result in ``float`` is rounded to a float.
SINGLE = frozenset({"+", "-", "*", "/"})

#: The operators whose Python is C's own, and their Python precedence.
PLAIN = {
    "+": P.ADD,
    "-": P.ADD,
    "*": P.MUL,
    "&": P.BITAND,
    "|": P.BITOR,
    "^": P.XOR,
    "<<": P.SHIFT,
    ">>": P.SHIFT,
}


def _is_null(node: Expr) -> bool:
    return isinstance(node, Literal) and (
        node.kind == "null" or (node.kind == "int" and node.value == 0)
    )


class ExprEmitter(NameEmitter):
    """The emitter's layer for expressions."""

    def call(self, node: Call) -> Out:
        raise NotImplementedError

    def lambda_(self, node: Lambda) -> Out:
        raise NotImplementedError

    def constructor_arguments(self, ctype: CType, args: list[Expr]) -> str:
        raise NotImplementedError

    def expr(self, node: Any) -> Out:
        return self._EMIT[type(node)](self, node)

    # -- truth ---------------------------------------------------------------

    def cond(self, node: Expr) -> Out:
        """``node`` where C++ tests its truth: pointers against ``None``, ``&&`` as ``and``."""
        if isinstance(node, Binary) and node.op in ("&&", "||"):
            op, level = ("and", P.AND) if node.op == "&&" else ("or", P.OR)
            left = self._truth_at(node.left, level)
            right = self._truth_at(node.right, level + 1)
            return f"{left} {op} {right}", level
        if isinstance(node, Unary) and node.op == "!":
            return self._negated(node.operand)
        if self.is_pointer(node) and not isinstance(node, (Binary, Literal)):
            return f"{self.at(node, P.COMPARE + 1)} is not None", P.COMPARE
        return self.expr(node)

    def _truth_at(self, node: Expr, level: int) -> str:
        text, own = self.cond(node)
        return text if own >= level else f"({text})"

    def _negated(self, node: Expr) -> Out:
        if self.is_pointer(node):
            return f"{self.at(node, P.COMPARE + 1)} is None", P.COMPARE
        return f"not {self._truth_at(node, P.NOT)}", P.NOT

    def condition(self, node: Expr) -> str:
        return self.cond(node)[0]

    # -- binary ----------------------------------------------------------------

    def _binary(self, node: Binary) -> Out:
        if node.op in SINGLE and self.single(node):
            return f"f32({self._operation(node)[0]})", P.POSTFIX
        return self._operation(node)

    def single(self, node: Expr) -> bool:
        """Is ``node`` an operation C does in ``float``, whose result the Python rounds?"""
        if not isinstance(node, Binary) or node.op not in SINGLE:
            return False
        found = self.typeof(node)
        return found is not None and found.floating and found.name == "float"

    def _operation(self, node: Binary) -> Out:
        op = node.op
        if op in ("&&", "||"):
            return f"bool({self.condition(node)})", P.POSTFIX
        if op in COMPARISONS:
            return self._compare(node)
        if op in ("/", "%"):
            return self._divide(node)
        special = self._arithmetic_special(node)
        if special is not None:
            return special
        if op in PLAIN:
            level = PLAIN[op]
            return f"{self.at(node.left, level)} {op} {self.at(node.right, level + 1)}", level
        raise self.refuse(f"the operator {op}", node)

    def _arithmetic_special(self, node: Binary) -> Out | None:
        """``+``, ``-``, ``<<`` whose Python is not Python's own: on pointers, strings, streams."""
        if node.op in ("+", "-") and self._pointer_arithmetic(node):
            return self._offset(node)
        if self._concatenation(node):
            right = f"cstr({self.value(node.right)})"
            return f"{self.at(node.left, P.ADD)} + {right}", P.ADD
        if node.op in ("<<", ">>") and self._streams(node):
            return self._shift(node)
        return None

    def _compare(self, node: Binary) -> Out:
        left, right = node.left, node.right
        if node.op in ("==", "!=") and self._null_test(left, right):
            subject = right if _is_null(left) else left
            test = "is" if node.op == "==" else "is not"
            return f"{self.at(subject, P.COMPARE + 1)} {test} None", P.COMPARE
        level = P.COMPARE + 1
        return f"{self.at(left, level)} {node.op} {self.at(right, level)}", P.COMPARE

    def _null_test(self, left: Expr, right: Expr) -> bool:
        if isinstance(left, Literal) and left.kind == "null":
            return True
        if isinstance(right, Literal) and right.kind == "null":
            return True
        return (_is_null(right) and self.is_pointer(left)) or (
            _is_null(left) and self.is_pointer(right)
        )

    def _divide(self, node: Binary) -> Out:
        """``/`` and ``%``: C's integer ones where both sides are integers, Python's otherwise."""
        kind = self._division_kind(node)
        if node.op == "/" and kind == "floating":
            return f"{self.at(node.left, P.MUL)} / {self.at(node.right, P.MUL + 1)}", P.MUL
        helper = DIVISIONS[node.op, kind]
        return f"{helper}({self.value(node.left)}, {self.value(node.right)})", P.POSTFIX

    def _division_kind(self, node: Binary) -> str:
        """``integral`` if both operands are, ``floating`` if either is, else ``unknown``."""
        left, right = self.typeof(node.left), self.typeof(node.right)
        if left is not None and right is not None and left.integral and right.integral:
            return "integral"
        if any(side is not None and side.floating for side in (left, right)):
            return "floating"
        return "unknown"

    def _pointer_arithmetic(self, node: Binary) -> bool:
        left = self.typeof(node.left)
        if left is None or not (left.is_pointer or left.is_array):
            return False
        if left.is_string:
            right = self.typeof(node.right)
            return right is not None and right.integral
        return not left.is_object_pointer or left.is_array

    def _concatenation(self, node: Binary) -> bool:
        """``"text" + s``: a C string joined to a string, which C++ does through ``s``'s class."""
        left, right = self.typeof(node.left), self.typeof(node.right)
        if node.op != "+" or left is None or not left.is_string or left.name == "std::string":
            return False
        return right is None or not right.is_string

    def _offset(self, node: Binary) -> Out:
        left = self.typeof(node.left)
        assert left is not None
        if left.is_string and node.op == "+" and not left.dims:
            # ``slash + 1``: the C string from there on, which is what reading it gives.
            return f"{self.at(node.left, P.POSTFIX)}[{self.value(node.right)}:]", P.POSTFIX
        if node.op == "-" or left.is_string or not left.element().scalar:
            raise self.refuse(f"pointer arithmetic on a {_spelled(left)}", node)
        return f"{self.at(node.left, P.POSTFIX)}[{self.value(node.right)}:]", P.POSTFIX

    def _streams(self, node: Binary) -> bool:
        root = node.left
        while isinstance(root, Binary) and root.op == node.op:
            root = root.left
        found = self.typeof(root)
        if isinstance(root, Name) and root.parts[0] == "std" and root.last in ("cout", "cerr"):
            return True
        return found is not None and "stream" in found.name

    def _shift(self, node: Binary) -> Out:
        if node.op == ">>":
            return self.extraction(node)
        right = self.at(node.right, P.SHIFT + 1)
        found = self.typeof(node.right)
        writer = self.stream_writers.get(_written_as(found))
        if writer is not None:
            return f"{writer}({self.value(node.left)}, {self.value(node.right)})", P.POSTFIX
        if found is not None and found.is_char:
            right = f"chr({self.value(node.right)})"
        return f"{self.at(node.left, P.SHIFT)} << {right}", P.SHIFT

    # -- unary ---------------------------------------------------------------

    def _unary(self, node: Unary) -> Out:
        op = node.op
        if op in ("++", "--"):
            return self.increment(node)
        if op == "!":
            return self._negated(node.operand)
        if op == "*":
            return self.dereference(node.operand)
        if op == "&":
            return self.address(node.operand)
        return f"{op}{self.at(node.operand, P.UNARY)}", P.UNARY

    def dereference(self, operand: Expr) -> Out:
        """``*p``: the object itself for a pointer to one, element 0 for a pointer to numbers."""
        if isinstance(operand, This):
            return "self", P.ATOM
        found = self.typeof(operand)
        if found is None:
            return f"deref({self.value(operand)})", P.POSTFIX
        own = self.own_operator(operand, "operator*", 0)
        if own is not None:
            return own, P.POSTFIX
        if _holds_value(found):  # a shared_ptr<int>: a number, or something holding one
            return f"deref({self.value(operand)})", P.POSTFIX
        if found.is_object_pointer or (found.is_class and not found.pointer):
            return self.expr(operand)
        return f"{self.at(operand, P.POSTFIX)}[0]", P.POSTFIX

    def address(self, operand: Expr) -> Out:
        """``&x``: the cell ``x`` lives in, a view of an array from an element, or the object."""
        if isinstance(operand, Name):
            return self._name_address(operand)
        if isinstance(operand, Index):
            return self._element_address(operand)
        if isinstance(operand, Member):
            return self._member_address(operand)
        if isinstance(operand, Unary) and operand.op == "*":
            return self.expr(operand.operand)
        return self.expr(operand)

    def _name_address(self, operand: Name) -> Out:
        symbol = self.symbol(operand)
        if symbol is None:
            return self.expr(operand)
        found = self.typeof(operand)
        if symbol.cell and symbol.alias is None and not _object_value(found):
            prefix = "self." if symbol.kind == "field" else ""
            return prefix + symbol.py, P.POSTFIX
        if symbol.kind == "field" and found is not None and found.scalar:
            return f"AttrRef(self, {symbol.py!r})", P.POSTFIX
        return self.expr(operand)

    def _member_address(self, operand: Member) -> Out:
        found = self.typeof(operand)
        if found is None or found.scalar or found.is_string:
            return f"AttrRef({self.value(operand.obj)}, {operand.name!r})", P.POSTFIX
        return self.expr(operand)

    def _element_address(self, operand: Index) -> Out:
        found = self.typeof(operand.obj)
        array = self.at(operand.obj, P.POSTFIX)
        index = self.value(operand.index)
        if found is not None and (found.is_array or found.is_pointer) and found.element().scalar:
            return f"{array}[{index}:]", P.POSTFIX
        return f"ItemRef({self.value(operand.obj)}, {index})", P.POSTFIX

    def reference(self, target: Expr) -> str:
        """What reads and writes ``target`` as ``.value``: a cell, an ItemRef or an AttrRef."""
        if isinstance(target, Index):
            return f"ItemRef({self.value(target.obj)}, {self.value(target.index)})"
        if isinstance(target, Member):
            return f"AttrRef({self.value(target.obj)}, {target.name!r})"
        found = self._name_reference(target) if isinstance(target, Name) else None
        if found is not None:
            return found
        if isinstance(target, Unary) and target.op == "*":
            return f"ItemRef({self.value(target.operand)}, 0)"
        raise self.refuse("changing this in the middle of an expression", target)

    def _name_reference(self, target: Name) -> str | None:
        symbol = self.symbol(target)
        if symbol is None:
            return None
        if symbol.cell:
            return ("self." if symbol.kind == "field" else "") + symbol.py
        return f"AttrRef(self, {symbol.py!r})" if symbol.kind == "field" else None

    def local_name(self, target: Expr) -> str | None:
        """The Python name a walrus can assign, when ``target`` is a plain local or global."""
        if not isinstance(target, Name):
            return None
        symbol = self.symbol(target)
        if symbol is None or symbol.cell or symbol.alias is not None:
            return None
        if symbol.kind not in ("local", "param", "global"):
            return None
        self.assigned(symbol)
        return symbol.py

    def assigned(self, symbol: Any) -> None:
        """Note that the function being written assigns ``symbol``: a global needs declaring."""

    def own_operator(self, operand: Expr, name: str, operands: int) -> str | None:
        """The method of the macro's class ``operand`` is, for its ``name`` operator, if it has one.

        ``operands`` is how many the method takes besides the object: ``it++``
        is ``operator++(int)``, ``++it`` and ``*it`` take none.
        """
        if not any(len(func.params) == operands for func in self._operators(operand, name)):
            return None
        binary, unary = DUNDERS[name]
        chosen = binary if operands else unary
        return f"{self.at(operand, P.POSTFIX)}.{chosen}({'0' if operands else ''})"

    def _operators(self, operand: Expr, name: str) -> list[Any]:
        """The ``name`` operators of the macro's class an object ``operand`` is of, if any."""
        found = self.typeof(operand)
        if found is None or found.pointer:
            return []
        info = self.program.classes.get(found.name)
        return info.methods.get(name, []) if info is not None else []

    def increment(self, node: Unary) -> Out:
        own = self.own_operator(node.operand, "operator" + node.op, int(node.postfix))
        if own is not None:
            return own, P.POSTFIX
        delta = "1" if node.op == "++" else "-1"
        name = self.local_name(node.operand)
        if name is not None:
            step = f"({name} := {name} + {delta})"
            return (f"({step} - {delta})", P.ADD) if node.postfix else (step, P.ATOM)
        helper = "postinc" if node.postfix else "preinc"
        return f"{helper}({self.reference(node.operand)}, {delta})", P.POSTFIX

    # -- assignment inside an expression -------------------------------------

    def call_assignment(self, node: Assign) -> str:
        raise NotImplementedError

    def extraction(self, node: Binary) -> Out:
        raise NotImplementedError

    def stored_through(self, node: Assign) -> str | None:
        """``f(i) = v`` and ``*p = v`` of a pointer to an object or of unknown type, if it is one.

        What ``*p`` is cannot be rebound in Python, so the store goes through
        the runtime: into the cell ``p`` is (a smart pointer to a value that
        ROOT made, say), or into the object it points at.
        """
        target = node.target
        if isinstance(target, Call):
            return self.call_assignment(node)
        if not self._held_through(target):
            return None
        return self.store_expression(target, self.assigned_value(node))[0]

    def _held_through(self, target: Expr) -> bool:
        """Is ``target`` ``*p`` of a pointer to an object, or to what this translator cannot say?"""
        if not (isinstance(target, Unary) and target.op == "*"):
            return False
        found = self.typeof(target.operand)
        return found is None or found.is_object_pointer

    def _assign(self, node: Assign) -> Out:
        stored = self.stored_through(node)
        if stored is not None:
            return stored, P.POSTFIX
        return self.store_expression(node.target, self.assigned_value(node))

    def store_expression(self, target: Expr, value: str) -> Out:
        """Python that stores ``value`` in ``target`` and is worth it, as C++'s ``=`` is."""
        name = self.local_name(target)
        if name is not None:
            return f"({name} := {value})", P.ATOM
        if isinstance(target, Index):
            obj, index = self.value(target.obj), self.value(target.index)
            return f"set_item({obj}, {index}, {value})", P.POSTFIX
        if isinstance(target, Member):
            return f"set_attr({self.value(target.obj)}, {target.name!r}, {value})", P.POSTFIX
        if self._held_through(target):
            assert isinstance(target, Unary)
            return f"store_through({self.value(target.operand)}, {value})", P.POSTFIX
        reference = self.reference(target)
        return f"set_attr({reference}, 'value', {value})", P.POSTFIX

    def assigned_value(self, node: Assign) -> str:
        """What an assignment stores, compound operators applied and C's conversion made."""
        value: Expr = node.value
        if node.op != "=":
            value = Binary(node.where, node.op[:-1], node.target, node.value)
        return self.store(self.typeof(node.target), value)

    def store(self, ctype: CType | None, node: Expr) -> str:
        raise NotImplementedError

    def _ternary(self, node: Ternary) -> Out:
        yes = self.at(node.yes, P.TERNARY + 1)
        test = self._truth_at(node.cond, P.TERNARY + 1)
        return f"{yes} if {test} else {self.at(node.no, P.TERNARY)}", P.TERNARY

    # -- postfix ---------------------------------------------------------------

    def _member(self, node: Member) -> Out:
        if node.name.startswith("~"):
            raise self.refuse(f"calling the destructor {node.name} by hand", node)
        obj = "self" if isinstance(node.obj, This) else self.at(node.obj, P.POSTFIX)
        return self.attribute(obj, node.name) + self.targs(node.targs), P.POSTFIX

    @staticmethod
    def attribute(obj: str, name: str) -> str:
        """``obj.name``, or ``getattr(obj, 'import')`` when the name is one of Python's keywords."""
        name = python_operator(name)
        if keyword.iskeyword(name):
            return f"getattr({obj}, {name!r})"
        return f"{obj}.{name}"

    def _index(self, node: Index) -> Out:
        found = self.typeof(node.obj)
        if found is not None and found.is_string and not found.dims:
            return f"char_at({self.value(node.obj)}, {self.value(node.index)})", P.POSTFIX
        return f"{self.at(node.obj, P.POSTFIX)}[{self.value(node.index)}]", P.POSTFIX

    def _cast(self, node: Cast) -> Out:
        target = node.ctype
        if node.kind == "dynamic" and target.is_class:
            kind = self.class_expr(target.element() if target.pointer else target)
            return f"dynamic_cast({kind}, {self.value(node.operand)})", P.POSTFIX
        if target.scalar:
            return self.store(target, node.operand), P.POSTFIX
        return self._held_cast(target, node.operand)

    def _held_cast(self, target: CType, operand: Expr) -> Out:
        """A cast to a pointer or a string: ``None`` from ``0``, text from what is not text."""
        if target.pointer and _is_null(operand):
            return "None", P.ATOM
        if target.is_string and not target.is_array and self.typeof(operand) is None:
            return f"cstr({self.value(operand)})", P.POSTFIX
        return self.expr(operand)

    def _sizeof(self, node: SizeOf) -> Out:
        ctype = node.ctype if node.ctype is not None else self.typeof(node.operand)
        size = ctype.size if ctype is not None else None
        if size is None:
            raise self.refuse("sizeof of a type whose size this translator does not know", node)
        return str(size), P.ATOM

    def _new(self, node: New) -> Out:
        ctype = node.ctype
        if node.place is not None:
            return self._placed(node), P.POSTFIX
        if node.count is not None:
            return self.new_array(ctype, node.count), P.POSTFIX
        args = self.constructor_arguments(ctype, node.args or [])
        if ctype.scalar or ctype.pointer:
            first = node.args[0] if node.args else None
            initial = self.store(ctype, first) if first is not None else zero(ctype)
            return f"Cell({initial}, {ctype.name!r})", P.POSTFIX
        return f"{self.class_expr(ctype)}({args})", P.POSTFIX

    def _placed(self, node: New) -> str:
        """``new (clones[i]) T(args)``: a ``T`` built and put in slot ``i`` of the array."""
        place = node.place
        if not isinstance(place, Index) or not node.ctype.is_class or node.ctype.pointer:
            why = "placement new of anything but an object into an element of an array"
            raise self.refuse(why, node)
        args = self.constructor_arguments(node.ctype, node.args or [])
        built = f"{self.class_expr(node.ctype)}({args})"
        return f"construct_at({self.value(place.obj)}, {self.value(place.index)}, {built})"

    def new_array(self, ctype: CType, count: Expr) -> str:
        """``new T[n]``: an array of ``n`` zeros, or of ``None`` for pointers and objects."""
        if ctype.scalar:
            return f"array({ctype.name!r}, {self.value(count)})"
        if ctype.pointer:
            return f"[None] * {self.at(count, P.MUL)}"
        return f"array({type_text(ctype)!r}, {self.value(count)}, make={self.class_expr(ctype)})"

    def _delete(self, node: Delete) -> Out:
        raise self.refuse("delete inside an expression", node)

    def _init_list(self, node: InitList) -> Out:
        items = ", ".join(self.value(item) for item in node.items)
        if node.ctype is not None:
            if node.ctype.scalar:
                return self.store(node.ctype, node.items[0]) if node.items else "0", P.POSTFIX
            return f"{self.class_expr(node.ctype)}({items})", P.POSTFIX
        return f"[{items}]", P.ATOM

    def _comma(self, node: Comma) -> Out:
        return f"comma({', '.join(self.value(item) for item in node.items)})", P.POSTFIX

    def _this(self, node: This) -> Out:
        return "self", P.ATOM

    def _throw(self, node: Throw) -> Out:
        value = self.value(node.operand) if node.operand is not None else ""
        return f"throw({value})", P.POSTFIX

    def _literal(self, node: Literal) -> Out:
        return self.literal(node)

    def _name(self, node: Name) -> Out:
        return self.name(node)

    def _call(self, node: Call) -> Out:
        return self.call(node)

    def _lambda(self, node: Lambda) -> Out:
        return self.lambda_(node)

    def class_expr(self, ctype: CType) -> str:
        """The Python that builds a ``ctype``: the macro's class, ``str``, or ROOT's class."""
        name = "::".join(self._unqualified(ctype.name.split("::")))
        if name in self.program.classes:
            symbol = self.class_symbols.get(name) or self.lookup(name)
            return symbol.py if symbol is not None else name
        if ctype.is_string:
            return "str"
        parts = name.split("::")
        return self.library(Name(self.scope_where, parts, ctype.args or None))[0]

    #: Where the emitter is, for the nodes it makes up for itself.
    scope_where: Any

    _EMIT: ClassVar[dict[type, Callable[[ExprEmitter, Any], Out]]] = {
        Literal: _literal,
        Name: _name,
        Unary: _unary,
        Binary: _binary,
        Assign: _assign,
        Ternary: _ternary,
        Call: _call,
        Member: _member,
        Index: _index,
        Cast: _cast,
        SizeOf: _sizeof,
        New: _new,
        Delete: _delete,
        Lambda: _lambda,
        InitList: _init_list,
        Comma: _comma,
        This: _this,
        Throw: _throw,
    }


def zero(ctype: CType) -> str:
    """What a C++ variable of ``ctype`` starts as when the macro gives it nothing."""
    if ctype.pointer or ctype.is_smart:
        return "None"
    if ctype.is_bool:
        return "False"
    if ctype.floating:
        return "0.0"
    return "0"


def _written_as(ctype: CType | None) -> str:
    """The type a free ``operator<<`` would be for, to write a value of ``ctype``."""
    if ctype is None or ctype.pointer:
        return ""
    return ctype.enum or ctype.name


def _holds_value(ctype: CType) -> bool:
    """Is ``ctype`` a smart pointer to a number or a string?

    ``std::make_shared<int>(5)`` is the number itself here, but an RNTuple
    model's ``shared_ptr<int>`` is the field's value holder, so reading
    through one is :func:`~.runtime.objects.deref`, which takes either.
    """
    held = ctype.args[0] if ctype.is_smart and ctype.args else None
    return isinstance(held, CType) and (held.scalar or held.is_string)


def _object_value(ctype: CType | None) -> bool:
    """An object held by value - whose address is the object, even when a cell holds it."""
    return ctype is not None and ctype.is_class and not ctype.pointer and not ctype.is_smart


def _spelled(ctype: CType) -> str:
    stars = "*" * ctype.pointer or "[]"
    return f"{ctype.name}{stars}"
