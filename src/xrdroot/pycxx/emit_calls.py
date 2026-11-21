"""Calls, and the stores C converts: where a program meets its library and itself.

A call to one of the program's functions is a Python call, with a variable
handed to a ``double&`` parameter handed as its cell. A call to a library
method is ``obj.Method(args)`` - the library's names are kept exactly - with
the variables the dialect says it writes through handed as cells. The
standard library's calls that have no Python counterpart are rewritten:
``std::string``'s members become Python's, ``std::make_unique<T>(a)`` is
``T(a)``, ``std::sort(v.begin(), v.end())`` sorts ``v`` in place.

:meth:`CallEmitter.store` is C's conversion on assignment, written out: an
``int`` stored from a ``double`` is truncated, a ``float`` rounded, an
``unsigned`` wrapped, an object copied.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .base import Out, P
from .ctype import SMART, CType
from .cursor import looks_like_type
from .dialect import active
from .emit_expr import ExprEmitter, _copy_store, _is_null, _lvalue, zero
from .limits import FUNCTIONS as LIMIT_FUNCTIONS
from .limits import is_limit
from .literals import suffixes
from .nodes import (
    Assign,
    Binary,
    Call,
    Expr,
    Function,
    InitList,
    Literal,
    Member,
    Name,
)
from .operators import conversion
from .symbols import Symbol

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

#: ``std::max_element`` and ``std::min_element``, and the Python of what the one found holds.
EXTREMES = {"max_element": "max", "min_element": "min"}

#: C's functions that write text into character arrays, and the index their first such
#: argument has: a scan writes into each of its targets, ``fgets`` into the buffer it fills.
TEXT_TARGETS = {"fgets": 0, "scanf": 1, "fscanf": 2, "sscanf": 2}

#: A container's iterator functions, and the index each stands for in a slice.
ITERATOR_ENDS = {"begin": "0", "cbegin": "0", "end": "None", "cend": "None"}

def _tuple(items: list[str]) -> str:
    """Python's spelling of a tuple of ``items``: ``()``, ``(a,)``, ``(a, b)``."""
    return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"


def _plain(node: Expr) -> bool:
    """An argument evaluated twice to the same effect: a name, a literal, arithmetic on them."""
    if isinstance(node, Binary):
        return _plain(node.left) and _plain(node.right)
    return isinstance(node, (Name, Literal))


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

    def argument_list(self, args: list[Expr], at: tuple[int, ...],
                      replaced: dict[int, str] | None = None) -> str:
        """``args``, with those at ``at`` passed as something writable and those ``replaced``
        - by index - passed as what the caller worked out instead."""
        positions = set(at)
        items = []
        for index, arg in enumerate(args):
            if replaced is not None and index in replaced:
                items.append(replaced[index])
            elif index in positions and _lvalue(arg) and not self._object(arg):
                items.append(self.reference(arg))
            else:
                items.append(self.value(arg))
        return ", ".join(items)

    def _object(self, arg: Expr) -> bool:
        """An object, or an array: what a callee writes into is already what is handed over."""
        found = self.typeof(arg)
        if found is not None and found.dims:
            return True
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
        instance = self._instance_call(func, node, symbol)
        if instance is not None:
            return instance, P.POSTFIX
        if symbol is not None:
            # A member function named through its class is called on the object C++ gives it
            # implicitly, ``Base::Init()`` in a method being ``Base.Init(self)``: the plain
            # function is not bound to anything, so the call has to hand ``self`` over.
            own = ["self"] if symbol.kind == "member" else []
            given = [*own, self.arguments(node), self._template_values(func, symbol)]
            return f"{self.use(symbol)[0]}({', '.join(filter(None, given))})", P.POSTFIX
        if is_limit(func.parts, func.targs) and func.last in LIMIT_FUNCTIONS:
            return self.limit(func, node, "()"), P.ATOM
        if self._smart_alias(func):
            return self._pointer_held(node)
        special = self._special(func)
        if special is not None:
            found = special(self, func, node)
            if found is not None:
                return found
        return self.library_call(func, node)

    def library_call(self, func: Name, node: Call) -> Out:
        """A function the program did not declare: an inherited member's, a C text writer's
        that has to give its character array its text back, or the library's own call."""
        inherited = self.inherited_call(func)
        if inherited is not None:
            return f"{inherited}({self.arguments(node)})", P.POSTFIX
        written = self.text_written(func, node)
        if written is not None:
            return written, P.POSTFIX
        return f"{self.library(func)[0]}({self.arguments(node)})", P.POSTFIX

    def text_written(self, func: Name, node: Call) -> str | None:
        """``sscanf(line, "%s %d", name, &n)`` where ``name`` is a ``char[]`` the translation
        holds as a ``str``: C writes the field into that array's memory, which Python cannot
        write into, so a cell takes its place in the call and the variable is given the text
        once the call has run - by :func:`~xrdroot.cppstd.cstdio.written_after`, whose value
        is the call's own: the fields stored, or the buffer ``fgets`` filled."""
        start = self._text_start(func)
        if start is None:
            return None
        targets = [(index, arg) for index, arg in enumerate(node.args)
                   if index >= start and self._text_target(arg)]
        if not targets:
            return None
        cells = {index: self.fresh("text") for index, _ in targets}
        replaced = {index: f"{cells[index]} := Cell({self.value(arg)})" for index, arg in targets}
        positions = self.program.reference_positions(node)
        given = self.argument_list(node.args, positions, replaced)
        assigned = [f"{self.value(arg)} := {cells[index]}.value" for index, arg in targets]
        return f"written_after({self.library(func)[0]}({given}), {', '.join(assigned)})"

    def _text_start(self, func: Name) -> int | None:
        """Where the character arrays a C function writes into start among its arguments, if
        the call is C's own function: a library's ``scanf`` is not this one."""
        return TEXT_TARGETS.get(func.last) if len(func.parts) == 1 else None

    def _text_target(self, arg: Expr) -> bool:
        """Is ``arg`` a ``char[]`` variable the translation holds as a ``str`` - one a callee
        writing text into it has to give the text back to through the variable? An array
        handed to a library to write into is memory, which is written in place, and a
        parameter's array belongs to the caller, who is the one to give it its text."""
        if not isinstance(arg, Name):
            return False
        symbol = self.symbol(arg)
        if symbol is None or symbol.kind != "local" or len(arg.parts) != 1:
            return False
        found = self.typeof(arg)
        if found is None or found.name != "char" or len(found.dims) != 1:
            return False
        return arg.last not in self.program.buffers

    def _instance_call(self, func: Name, node: Call, symbol: Symbol | None) -> str | None:
        """A call of a function template's instance, a member template's handed ``self``."""
        instance = self.function_instance(func, node, symbol)
        if instance is None:
            return None
        own = "self" if symbol is not None and symbol.kind in ("method", "member") else ""
        return f"{instance}({', '.join(filter(None, (own, self.arguments(node))))})"

    def function_instance(self, func: Name, node: Call, symbol: Symbol | None) -> str | None:
        """The instance of a function template a call calls, if it is one."""
        raise NotImplementedError

    def method_instance(self, owner: CType | None, func: Member, node: Call) -> str | None:
        """The instance of a member template a call of a method calls, if it is one."""
        raise NotImplementedError

    def _special(self, func: Name) -> Callable[[CallEmitter, Name, Call], Out | None] | None:
        """How a library function is written when it is not a plain call: the standard's,
        a user-defined literal's, or one the dialect writes - each by its unqualified name, or
        qualified by ``std`` or one of the dialect's classes."""
        dialect = active()
        if len(func.parts) > 1 and func.parts[0] != "std":
            if func.parts[0] not in dialect.call_classes:
                return None
        name = func.last
        found = self._LIBRARY.get(name) or dialect.calls.get(name)
        if found is None and name.startswith('operator""') and name[10:] in suffixes():
            return CallEmitter._suffixed
        return found

    def _template_values(self, func: Name, symbol: Symbol) -> str:
        """``f<3>(a)``: the value template arguments, as the keywords the function takes."""
        chosen = self._valued(symbol) if func.targs else None
        if chosen is None:
            return ""
        values = {str(param.name) for param in chosen.values}
        pairs = zip(chosen.template or [], func.targs or [], strict=False)
        return ", ".join(self._keyword(name, arg) for name, arg in pairs if name in values)

    def _keyword(self, name: str, arg: Any) -> str:
        """``N=3`` for a value argument; a type, written where a value is, passes nothing."""
        return "" if isinstance(arg, CType) else f"{name}={self.value(arg)}"

    def _valued(self, symbol: Symbol) -> Function | None:
        """The function template with value parameters that ``symbol`` names, if it is one."""
        if symbol.kind not in ("function", "static", "method", "member"):
            return None
        return next((f for f in self.function_named(symbol) if f.values), None)

    def _make(self, func: Name, node: Call) -> Out | None:
        """``std::make_shared<T>(args)``: a ``T`` built - a number in a cell, which ``*p``
        reads and writes through, as it does the number a smart pointer holds."""
        kind = func.targs[0] if func.targs else None
        if not isinstance(kind, CType):
            return None
        if kind.scalar:
            first = self.store(kind, node.args[0]) if node.args else zero(kind)
            return f"Cell({first}, {kind.name!r})", P.POSTFIX
        return f"{self.class_expr(kind)}({self.arguments(node)})", P.POSTFIX

    def _smart_alias(self, func: Name) -> bool:
        """Is ``func`` a name ``using`` or ``typedef`` gave a smart pointer - ``Upvd_t(p)``?"""
        alias = self.program.aliases.get(func.last) if len(func.parts) == 1 else None
        return alias is not None and (alias.name in SMART or alias.is_pointer)

    def _smart(self, func: Name, node: Call) -> Out | None:
        """``std::unique_ptr<T>(p)``: the pointer it holds, which is all a Python name is."""
        return self._pointer_held(node)

    def _pointer_held(self, node: Call) -> Out:
        if not node.args:
            return "None", P.ATOM
        return self.expr(node.args[0])

    def _exit(self, func: Name, node: Call) -> Out | None:
        return f"c_exit({self.arguments(node)})", P.POSTFIX

    def _suffixed(self, func: Name, node: Call) -> Out | None:
        """``0.1_normal``, ``100us``: the value the library's ``operator""`` makes of a number."""
        suffix = func.last[len('operator""') :]
        return f"user_literal({suffix!r}, {self.value(node.args[0])})", P.POSTFIX

    def _getline_value(self, func: Name, node: Call) -> Out | None:
        """``while (std::getline(in, line))``: the line stored, and the stream, to be tested."""
        if len(node.args) < 2:
            raise self.refuse("std::getline without a string to read into", node)
        stream = self._stream(node.args[0])
        delimiter = "".join(self.value(arg) for arg in node.args[2:3])
        stored = self.store_expression(node.args[1], f"{stream}.getline({delimiter})")[0]
        return f"stream_after({stream}, {stored})", P.POSTFIX

    def extraction(self, node: Binary) -> Out:
        """``while (in >> a >> b)``: each value read and stored in turn, then the stream."""
        targets: list[Expr] = []
        root: Expr = node
        while isinstance(root, Binary) and root.op == ">>":
            targets.insert(0, root.right)
            root = root.left
        stream = self._stream(root)
        stores = []
        for target in targets:
            kind = self.typeof(target)
            read = f"{stream}.extract({kind.name if kind is not None else 'double'!r})"
            stores.append(self.store_expression(target, read)[0])
        return f"stream_after({stream}, {', '.join(stores)})", P.POSTFIX

    def _stream(self, node: Expr) -> str:
        """The stream a read in an expression reads from, which the Python names twice."""
        if not isinstance(node, Name):
            why = "reading inside an expression from a stream that is not a variable"
            raise self.refuse(why, node)
        return self.value(node)

    def _find_if(self, func: Name, node: Call) -> Out | None:
        """``std::find_if(first, last, pred)`` over library iterators: what it finds, or None."""
        if len(node.args) != 3 or self._container_end(node.args[0]) is not None:
            why = "std::find_if over a container, whose iterator Python has none of"
            raise self.refuse(why, node)
        args = ", ".join(self.value(arg) for arg in node.args)
        return f"find_if({args})", P.POSTFIX

    def _unsupported(self, func: Name, node: Call) -> Out | None:
        raise self.refuse(f"{func.text}() where its result is used", node)

    def extreme_element(self, operand: Expr) -> str | None:
        func = operand.func if isinstance(operand, Call) else None
        if not isinstance(func, Name) or func.last not in EXTREMES or self.symbol(func):
            return None
        assert isinstance(operand, Call)
        if len(operand.args) != 2:
            raise self.refuse(f"std::{func.last} with a comparison of its own", operand)
        container, start, stop = self.iterator_range(operand.args[0], operand.args[1], operand)
        return f"{EXTREMES[func.last]}({container}[{start}:{'' if stop == 'None' else stop}])"

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
        "find_if": _find_if,
        "getline": _getline_value,
    }

    # -- methods -----------------------------------------------------------------

    def method_call(self, func: Member, node: Call) -> Out:
        if node is self.vector_draw:  # the draw a fill loop takes all at once (see counted)
            given = ", ".join(filter(None, (self.arguments(node), f"n={self.vector_count}")))
            return f"{self.value(func)}({given})", P.POSTFIX
        owner = self.typeof(func.obj)
        instance = self.method_instance(owner, func, node)
        if instance is not None:
            return f"{instance}({self.value(func.obj)}, {self.arguments(node)})", P.POSTFIX
        special = (
            self._typed_method(owner, func, node)
            if owner is not None
            else self._untyped_method(func, node)
        )
        if special is not None:
            return special
        return f"{self.value(func)}({self.arguments(node)})", P.POSTFIX

    def _untyped_method(self, func: Member, node: Call) -> Out | None:
        """A method of an object whose type is not known, that only one kind of object has."""
        if node.args or func.arrow:
            return None
        if func.name == "c_str":
            # Only a std::string has c_str(), and a std::string is a str here, whoever made it.
            return f"cstr({self.value(func.obj)})", P.POSTFIX
        if func.name in ("release", "get"):
            # A unique_ptr's: what it holds, unless the object has a method of its own by name.
            return f"pointee({self.value(func.obj)}, {func.name!r})", P.POSTFIX
        if func.name in ("empty", "size", "length"):
            # A std::string or container a library handed back is a str or a list here.
            return f"std_member({self.value(func.obj)}, {func.name!r})", P.POSTFIX
        return None

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
        rewrite = STRING_METHODS.get(func.name) or active().string_methods.get(func.name)
        if rewrite is None:
            raise self.refuse(f"std::string::{func.name}()", node)
        return rewrite(obj, args), P.POSTFIX

    def _smart_method(self, func: Member, node: Call) -> Out:
        if func.name in ("get", "release", "operator->"):
            return self.expr(func.obj)
        raise self.refuse(f"the smart pointer's {func.name}() where its result is used", node)

    # -- assigning through a returned reference ------------------------------------

    def call_assignment(self, node: Assign, result: bool = True) -> str:
        """``f(i) = v``: what the call returns a reference to, written through the runtime."""
        target = node.target
        assert isinstance(target, Call)
        if node.op != "=" and not all(_plain(arg) for arg in target.args):
            return self._compound_call(node, target, result)
        value = self.assigned_value(node)
        func = target.func
        args = _tuple([self.value(arg) for arg in target.args])
        if isinstance(func, Member):
            return self._method_assignment(func, target, args, value)
        self._own_reference(func, "its operator()", node)
        if isinstance(func, Name) and self._names_class(func):
            # ``Column(a, 0) = 1.0``: a temporary built, and assigned into.
            return f"assign_into({self.value(target)}, {value})"
        return f"assign_call({self.at(func, P.POSTFIX)}, {args}, {value})"

    def _compound_call(self, node: Assign, target: Call, result: bool) -> str:
        """``v(v.GetUpb()) += 1``: the call's arguments evaluated once, not once to read and
        again to write - through a reference to what it returns, which the runtime reads and
        writes back. The right side goes first, as C++17 sequences ``E1 op= E2``."""
        reference = self._call_reference(target, node)
        return self.assigned_reference(node, reference) + (".value" if result else "")

    def _call_reference(self, target: Call, node: Assign) -> str:
        """What reads and writes, as ``.value``, the reference ``target`` returns."""
        func = target.func
        args = _tuple([self.value(arg) for arg in target.args])
        if isinstance(func, Member):
            self._own_reference(func.obj, f"its method {func.name}", target)
            obj = self.value(func.obj)
            if self._element_method(func, target):
                return f"ItemRef({obj}, {self.value(target.args[0])})"
            return f"CallRef({obj}, {args}, {func.name!r})"
        self._own_reference(func, "its operator()", node)
        if isinstance(func, Name) and self._names_class(func):
            return f"HeldRef({self.value(target)})"
        return f"CallRef({self.at(func, P.POSTFIX)}, {args})"

    def _method_assignment(self, func: Member, target: Call, args: str, value: str) -> str:
        """``obj.m(args) = v``: an element for ``at``, else through the runtime's ``Set``/copy."""
        self._own_reference(func.obj, f"its method {func.name}", target)
        obj = self.value(func.obj)
        if self._element_method(func, target):
            return f"set_item({obj}, {self.value(target.args[0])}, {value})"
        return f"assign_method({obj}, {func.name!r}, {args}, {value})"

    def _element_method(self, func: Member, target: Call) -> bool:
        """``v.at(i)`` or a reader's ``At(i)``: an element of the object, stored as one."""
        reader = self.reader_method(func, "array")
        at_reader = func.name == "At" and reader is not None and reader.scalar
        return (func.name == "at" or at_reader) and len(target.args) == 1

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
        return self._object_store(ctype, node)

    def _object_store(self, ctype: CType, node: Expr) -> str:
        """``node`` stored into an object: a string class made of a C string, built from a
        pointer, copied - which Python would otherwise share - or shared as it is; a pointer
        to an object set to ``0`` is null, which ``if (p)`` then finds it to be."""
        if ctype.is_object_pointer and _is_null(node):
            return "None"
        if self._made_string(ctype, node):
            return f"{self.host_name(ctype.name.split('::'))}({self.value(node)})"
        if self._converted_from_pointer(ctype, node):
            return f"{self.class_expr(ctype)}({self.value(node)})"
        if self._copied(ctype, node):
            return f"value_copy({self.value(node)})"
        if ctype.is_object_pointer and isinstance(node, Name) and self.symbol(node) is None:
            # ``TDirectory *d = gDirectory;``: the pointer as it is now, not a library global
            # that follows what is current.
            return f"pointer_now({self.value(node)})"
        return self.value(node)

    def _made_string(self, ctype: CType, node: Expr) -> bool:
        """Is a C string stored into one of the dialect's string classes - ``TString s;
        s = "a"`` - which builds one of the class rather than make the variable a ``str``?"""
        if ctype.pointer or ctype.name not in active().string_classes:
            return False
        found = self.typeof(node)
        return found is not None and found.is_string

    def _converted_from_pointer(self, ctype: CType, node: Expr) -> bool:
        """Is ``T x = p`` - or ``x = p`` - an object built from a pointer by ``T``'s converting
        constructor or assignment? A pointer can be nothing else to an object held by value."""
        held = ctype.pointer or ctype.is_smart or ctype.reference or ctype.callable
        if held or not ctype.is_class or ctype.dims or ctype.is_auto:
            return False
        found = self.typeof(node)
        return found is not None and found.is_object_pointer and not found.is_smart

    def _copied(self, ctype: CType, node: Expr) -> bool:
        """Is ``T b = a`` a copy of an object - which Python would otherwise share, not copy?"""
        return _copy_store(ctype, node, self.typeof(node))

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
        if source is not None and (source.integral or self._converts(source)):
            return f"float({self.value(node)})"
        return self.value(node)

    def _converts(self, source: CType) -> bool:
        """Is ``source`` a class of the program's that converts itself to a number?"""
        info = self.program.classes.get(source.name) if not source.pointer else None
        return info is not None and any(conversion(name) for name in info.methods)

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
}
