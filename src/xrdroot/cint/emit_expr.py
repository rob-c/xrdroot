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
from .errors import Refusal
from .infer import is_rvec
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
            return self._logical(node)
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

    def _logical(self, node: Binary) -> Out:
        """``&&`` and ``||``: of each element, ``&`` and ``|``, for an ``RVec``; else Python's."""
        if not is_rvec(self.typeof(node)):
            return f"bool({self.condition(node)})", P.POSTFIX
        level = P.BITAND if node.op == "&&" else P.BITOR
        bitwise = "&" if node.op == "&&" else "|"
        return f"{self.at(node.left, level)} {bitwise} {self.at(node.right, level + 1)}", level

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
        if node.op == "/" and kind == "floating" and isinstance(node.right, Literal):
            return f"{self.at(node.left, P.MUL)} / {self.at(node.right, P.MUL + 1)}", P.MUL
        if node.op == "/" and kind == "floating":  # by a value that may be zero: C's infinity
            return f"fdiv({self.value(node.left)}, {self.value(node.right)})", P.POSTFIX
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
        held = self._read_through(operand, found)
        if held is not None:
            return held, P.POSTFIX
        if found.is_object_pointer or (found.is_class and not found.pointer):
            return self.expr(operand)
        return f"{self.at(operand, P.POSTFIX)}[0]", P.POSTFIX

    def _read_through(self, operand: Expr, found: CType) -> str | None:
        """``*p`` of what holds a value rather than being it: a ``shared_ptr<int>`` - a number,
        or something holding one - or a library class that ``__deref__`` reads through."""
        if _holds_value(found):
            return f"deref({self.value(operand)})"
        if self.reader_element(found, "TTreeReaderValue") is not None:
            return f"{self.at(operand, P.POSTFIX)}.__deref__()"
        return None

    def address(self, operand: Expr) -> Out:
        """``&x``: the cell ``x`` lives in, a view of an array from an element, or the object."""
        if isinstance(operand, Name):
            return self._name_address(operand)
        if isinstance(operand, Index):
            return self._element_address(operand)
        if isinstance(operand, Member):
            return self._member_address(operand)
        borrowed = self._borrowed_address(operand)
        if borrowed is not None:
            return borrowed, P.POSTFIX
        if isinstance(operand, Unary) and operand.op == "*":
            return self.expr(operand.operand)
        return self.expr(operand)

    def _borrowed_address(self, operand: Expr) -> str | None:
        borrowed = self.borrowed_reference(operand) if isinstance(operand, Assign) else None
        if isinstance(operand, Unary) and operand.op in ("++", "--") and not operand.postfix:
            borrowed = self.borrowed_reference(operand)
        borrowed = borrowed if borrowed is not None else self.reader_reference(operand)
        if borrowed is None and _is_unaddressable_update(operand):
            # An update whose destination has no reference at all has no address to take.
            raise Refusal(
                f"{operand.where}: taking the address of an assignment or increment only "
                "works when its destination is addressable; assign to a variable whose "
                "address is taken first"
            )
        return borrowed

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
        borrowed = self.reader_reference(operand)
        if borrowed is not None:
            return borrowed, P.POSTFIX
        found = self.typeof(operand.obj)
        array = self.at(operand.obj, P.POSTFIX)
        index = self.value(operand.index)
        if found is not None and (found.is_array or found.is_pointer) and found.element().scalar:
            return f"{array}[{index}:]", P.POSTFIX
        return f"ItemRef({self.value(operand.obj)}, {index})", P.POSTFIX

    def reference(self, target: Expr) -> str:
        """What reads and writes ``target`` as ``.value``: a cell, an ItemRef or an AttrRef."""
        borrowed = self.reader_reference(target)
        if borrowed is not None:
            return borrowed
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

    def reader_reference(self, target: Expr) -> str | None:
        if isinstance(target, Unary) and target.op == "*":
            inner = self.reader_element(self.typeof(target.operand), "TTreeReaderValue")
            if inner is not None and inner.scalar:
                return f"{self.at(target.operand, P.POSTFIX)}.Get()"
        if isinstance(target, Index):
            inner = self.reader_element(self.typeof(target.obj), "TTreeReaderArray")
            if inner is not None and inner.scalar:
                return f"{self.at(target.obj, P.POSTFIX)}._cint_ref({self.value(target.index)})"
        return self._reader_call_reference(target)

    def _reader_call_reference(self, target: Expr) -> str | None:
        if not isinstance(target, Call) or not isinstance(target.func, Member):
            return None
        func = target.func
        inner = self.reader_method(func, "TTreeReaderArray")
        if func.name != "At" or len(target.args) != 1 or inner is None or not inner.scalar:
            return None
        return f"{self.at(func.obj, P.POSTFIX)}._cint_ref({self.value(target.args[0])})"

    def borrowed_reference(self, source: Expr) -> str | None:
        # Bind once, including assignment/prefix lvalues; postfix remains a value.
        updated = self._borrowed_update(source)
        if updated is not None:
            return updated
        borrowed = self.reader_reference(source)
        if borrowed is not None:
            return borrowed
        if isinstance(source, Unary) and source.op == "*":
            found = self.typeof(source.operand)
            if found is not None and found.is_pointer and found.element().scalar:
                return self.reference(source)
        return None

    def _borrowed_update(self, source: Expr) -> str | None:
        if isinstance(source, Assign):
            reference = self._update_reference(source.target)
            if reference is not None:
                return self.assigned_reference(source, reference)
        if isinstance(source, Unary) and source.op in ("++", "--") and not source.postfix:
            reference = self._update_reference(source.operand)
            if reference is not None:
                delta = 1 if source.op == "++" else -1
                return f"increment_reference({reference}, {delta})"
        return None

    def _update_reference(self, target: Expr) -> str | None:
        reference = self.borrowed_reference(target)
        if reference is None and isinstance(target, Name):
            reference = self._name_reference(target)
        return reference

    def assigned_reference(self, node: Assign, reference: str) -> str:
        # Python arguments evaluate RHS first, then capture the destination only once.
        right = self.value(node.value)
        conversion = self._reference_conversion(node)
        compound = ", True" if node.op != "=" else ""
        return f"store_reference({right}, {reference}, {conversion}{compound})"

    def _reference_conversion(self, node: Assign) -> str:
        with self.scoped():
            left = self.declare("__reader_left", "local", self.typeof(node.target))
            right = self.declare("__reader_right", "local", self.typeof(node.value))
            lhs = Name(node.where, [left.name])
            rhs = Name(node.where, [right.name])
            value = rhs if node.op == "=" else Binary(node.where, node.op[:-1], lhs, rhs)
            converted = self.store(self.typeof(node.target), value)
        return f"lambda {left.py}, {right.py}: {converted}"

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

    def stored_through(self, node: Assign, result: bool = True) -> str | None:
        """``f(i) = v`` and ``*p = v`` of a pointer to an object or of unknown type, if it is one.

        What ``*p`` is cannot be rebound in Python, so the store goes through
        the runtime: into the cell ``p`` is (a smart pointer to a value that
        ROOT made, say), or into the object it points at.
        """
        target = node.target
        borrowed = self.reader_reference(target)
        if borrowed is not None:
            return self.assigned_reference(node, borrowed) + (".value" if result else "")
        numeric = self._numeric_vector_assignment(node)
        if numeric is not None:
            return numeric
        if isinstance(target, Call):
            return self.call_assignment(node)
        if node.op == "=" and isinstance(target, Name) and _assigned_into(self.typeof(target)):
            return f"assign_into({self.value(target)}, {self.value(node.value)})"
        if not self._held_through(target):
            return None
        return self.store_expression(target, self.assigned_value(node))[0]

    def _numeric_vector_assignment(self, node: Assign) -> str | None:
        if node.op != "=":
            return None
        right = self.typeof(node.value)
        if not _vector_lvalue(node.value, right):
            return None
        destination, pointer = self._vector_destination(node.target)
        if destination is None:
            return None
        left = self.typeof(node.target)
        if left is None or not _numeric_vector_pair(left, right, self.program.classes):
            return None
        flag = _vector_copy_arguments(pointer, _copy_store(left, node.value, right))
        return f"assign_vector({self.value(destination)}, {self.value(node.value)}{flag})"

    def _vector_destination(self, target: Expr) -> tuple[Expr | None, bool]:
        if isinstance(target, Name):
            return target, False
        if isinstance(target, Unary) and self._held_through(target):
            return target.operand, True
        return None, False

    def _held_through(self, target: Expr) -> bool:
        """Is ``target`` ``*p`` of a pointer to an object, or to what this translator cannot say?"""
        if not (isinstance(target, Unary) and target.op == "*"):
            return False
        found = self.typeof(target.operand)
        inner = self.reader_element(found, "TTreeReaderValue")
        return found is None or found.is_object_pointer or (inner is not None and inner.scalar)

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
    """Whether ``operator*`` reads a booked result or a smart pointer's number or string.

    ``std::make_shared<int>(5)`` is the number itself here, but an RNTuple
    model's ``shared_ptr<int>`` is the field's value holder, so reading
    through one is :func:`~.runtime.objects.deref`, which takes either.
    """
    results = ("RResultPtr", "RDF::RResultPtr", "ROOT::RDF::RResultPtr")
    if ctype.name in results and not ctype.pointer and not ctype.dims:
        return True
    held = ctype.args[0] if ctype.is_smart and ctype.args else None
    return isinstance(held, CType) and (held.scalar or held.is_string)


#: What a view of part of a matrix is called after: ``TMatrixDRow``, ``TMatrixTDiag<double>``.
VIEWS = ("Row", "Column", "Diag", "Sub", "Flat", "Row_const", "Column_const", "Diag_const")


def _is_view(ctype: CType | None) -> bool:
    """Is ``ctype`` a view of part of a matrix, which ``view = value`` writes through?"""
    name = ctype.name.split("::")[-1] if ctype is not None and not ctype.pointer else ""
    return name.startswith("TMatrix") and name.endswith(VIEWS)


#: ROOT's classes a variable of is set by ``=`` - its ``operator=`` of a number or a label - and
#: not made anew: ``RooRealVar x(...); x = 3;`` (rf402).
SET_BY_ASSIGNMENT = frozenset({"RooRealVar", "RooCategory"})


def _assigned_into(ctype: CType | None) -> bool:
    """Is ``ctype`` a matrix's view, or a class set by ``=``, which ``target = value`` writes
    into in place rather than rebinding the name?"""
    if ctype is None or ctype.pointer:
        return False
    return _is_view(ctype) or ctype.name.split("::")[-1] in SET_BY_ASSIGNMENT


def _is_unaddressable_update(operand: Expr) -> bool:
    """An assignment or prefix increment, in an address context, with no reference."""
    if isinstance(operand, Assign):
        return True
    return isinstance(operand, Unary) and operand.op in ("++", "--") and not operand.postfix


def _object_value(ctype: CType | None) -> bool:
    """An object held by value - whose address is the object, even when a cell holds it."""
    return ctype is not None and ctype.is_class and not ctype.pointer and not ctype.is_smart


def _spelled(ctype: CType) -> str:
    stars = "*" * ctype.pointer or "[]"
    return f"{ctype.name}{stars}"


#: Whole-copy assignment is limited to the ordinary native integer and floating families.
_VECTOR_NUMBERS = frozenset({
    "char", "signed char", "unsigned char", "short", "unsigned short", "int", "unsigned int",
    "long", "unsigned long", "long long", "unsigned long long", "float", "double", "long double",
})
#: ROOT's numeric aliases identify the same RVec specializations as their template spellings.
_VECTOR_ALIASES = {"RVecC": "char", "RVecD": "double", "RVecF": "float", "RVecI": "int",
                   "RVecL": "long", "RVecLL": "long long", "RVecU": "unsigned int",
                   "RVecUL": "unsigned long", "RVecULL": "unsigned long long"}


def _numeric_vector_type(ctype: CType | None) -> tuple[str, str] | None:
    if ctype is None or ctype.pointer or ctype.dims or ctype.callable:
        return None
    return _numeric_vector_name(ctype)


def _numeric_vector_name(ctype: CType) -> tuple[str, str] | None:
    family, _, name = ctype.name.rpartition("::")
    if family not in ("", "std", "ROOT", "ROOT::VecOps", "VecOps"):
        return None
    if name in _VECTOR_ALIASES and family != "std":
        return "rvec", _VECTOR_ALIASES[name]
    if name not in ("vector", "RVec") or len(ctype.args) != 1:
        return None
    return _numeric_vector_element(name, ctype.args[0])


def _numeric_vector_element(name: str, element: Any) -> tuple[str, str] | None:
    if not isinstance(element, CType) or not element.scalar or element.name not in _VECTOR_NUMBERS:
        return None
    return ("std" if name == "vector" else "rvec"), element.name


def _vector_lvalue(node: Expr, found: CType | None) -> bool:
    if isinstance(node, (Name, Member, Index)):
        return True
    if isinstance(node, Assign):
        return node.op == "="
    return (isinstance(node, Unary) and node.op == "*") or (
        isinstance(node, Call) and found is not None and found.reference)



def _numeric_vector_pair(left: CType | None, right: CType | None, classes: Any) -> bool:
    if left is None or right is None or left.const:
        return False
    if left.name in classes or right.name in classes:
        return False
    signature = _numeric_vector_type(left)
    return signature is not None and signature == _numeric_vector_type(right)



def _lvalue(node: Expr) -> bool:
    return isinstance(node, (Name, Member, Index)) or (isinstance(node, Unary) and node.op == "*")


def _copy_store(ctype: CType, node: Expr, found: CType | None) -> bool:
    if not ctype.is_class or ctype.pointer or ctype.reference or not _lvalue(node):
        return False
    return found is not None and found.is_class and not found.pointer


def _vector_copy_arguments(pointer: bool, copy: bool) -> str:
    if not copy:
        return f", {pointer}, False"
    return ", True" if pointer else ""
