"""C++ classes and enums as Python classes and constants.

A class is a Python class deriving from the same bases - the macro's own,
or the library's (``class Mine : public Base`` of an undeclared ``Base`` derives
from the host's ``Base``). Its constructor is ``__init__``, running the base
constructors and the member initialisers C++ would, in the order C++ does;
a struct with no constructor takes its members positionally, as aggregate
initialisation does. Operators become the dunder methods Python calls, the
destructor ``_destruct`` (which ``delete`` calls). ``ClassDef`` and
``ClassImp`` have already gone, in the preprocessor.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .ctype import CType
from .emit_funcs import FunctionEmitter
from .errors import Refusal
from .nodes import ClassDecl, EnumDecl, Expr, Function, Literal, Name, Unary, VarDecl
from .operators import DUNDERS, conversion
from .program import ClassInfo
from .symbols import Scope, Symbol, member_name, python_name

__all__ = ["ClassEmitter"]


class ClassEmitter(FunctionEmitter):
    """The emitter's layer for classes and enums."""

    def top(self, decl: object) -> None:
        raise NotImplementedError

    #: The lines of module-level holders the static locals being written need.
    module_statics: list[tuple[str, Any]]

    # -- enums -------------------------------------------------------------------

    def enum_values(self, decl: EnumDecl) -> list[tuple[str, str]]:
        """Each enumerator and its value: counted on from the last, as C++ counts."""
        values: list[tuple[str, str]] = []
        last: int | str = -1
        for name, value in decl.items:
            if value is not None:
                known = _constant(value)
                last = known if known is not None else self.value(value)
            elif isinstance(last, int):
                last += 1
            else:
                last = f"{values[-1][0]} + 1"
            values.append((name, str(last)))
        return values

    def enum_def(self, decl: EnumDecl) -> None:
        values = self.enum_values(decl)
        if decl.scoped and decl.name:
            self.out.blank(2 if self.scope.kind == "module" else 1)
            self.out.line(f"class {python_name(decl.name)}:", decl.where)
            with self.out.indented():
                for name, value in values:
                    self.out.line(f"{name} = {value}", decl.where)
        else:
            for name, value in values:
                symbol = self.lookup(name)
                self.out.line(f"{symbol.py if symbol else name} = {value}", decl.where)
        self.declarators(decl.declarators)

    def local_enum(self, decl: EnumDecl) -> None:
        home = python_name(decl.name) if decl.scoped and decl.name else ""
        if home:
            self.scope.add(Symbol(str(decl.name), "class", home))
        for name, _ in decl.items:
            py = f"{home}.{name}" if home else python_name(name)
            self.scope.add(Symbol(name, "constant", py, CType("int")))
        self.enum_def(decl)

    def local_class(self, decl: ClassDecl) -> None:
        self.class_def(decl)

    # -- classes -------------------------------------------------------------------

    def class_def(self, decl: ClassDecl) -> None:
        if decl.template is None:
            self._class_def(decl)
            return
        self.generic(
            lambda: self._class_def(decl),
            lambda why: self._uninstantiated_class(decl, why),
        )

    def generic(self, write: Callable[[], None], instead: Callable[[str], None]) -> None:
        """A template written once for every use whose arguments are not known - or, where
        its body cannot be written without them, ``instead``: a stand-in refusing its use."""
        mark, statics, names = self.out.mark(), len(self.module_statics), set(self.template_names)
        try:
            write()
        except Refusal as why:
            self.out.truncate(mark)
            del self.module_statics[statics:]
            self.template_names = names
            instead(str(why))

    def _uninstantiated_class(self, decl: ClassDecl, why: str) -> None:
        py = self._class_symbol(decl.name).py
        self.out.blank(2 if self.scope.kind == "module" else 1)
        self.out.line(f"class {py}:", decl.where)
        with self.out.indented():
            self.out.line("def __init__(self, *args, **kwargs):", decl.where)
            with self.out.indented():
                what = f"the class template {decl.name}"
                self.out.line(f"uninstantiated({what!r}, {why!r})", decl.where)

    def _class_def(self, decl: ClassDecl) -> None:
        info = self.program.classes[decl.name]
        for nested in info.types:
            self.top(nested)
        py = self._class_symbol(decl.name).py
        bases = [self._base(base.ctype) for base in decl.bases]
        self.out.blank(2 if self.scope.kind == "module" else 1)
        self.out.line(f"class {py}({', '.join(bases)}):" if bases else f"class {py}:", decl.where)
        template = set(decl.template or [])
        self.template_names |= template
        with self.out.indented(), self.scoped("class", klass=decl.name) as scope:
            self._members(info, scope.symbols)
            self._class_body(info)
        self.template_names -= template
        if any(conversion(name) for name in info.methods):  # ``operator int()`` and kin
            self.out.line(f"convertible({py})", decl.where)
        self.declarators(decl.declarators)

    def declarators(self, decls: list[VarDecl]) -> None:
        """``struct P {...} p;``: the variables declared with a class or enum."""
        raise NotImplementedError

    def _class_symbol(self, name: str) -> Symbol:
        symbol = self.class_symbols.get(name) or self.lookup(name)
        assert symbol is not None, f"every class is declared before it is written: {name}"
        return symbol

    def _base(self, ctype: CType) -> str:
        if ctype.name in self.program.classes:
            return self._class_symbol(ctype.name).py
        return self.class_expr(ctype)

    def _members(self, info: ClassInfo, symbols: dict[str, Symbol]) -> None:
        """The names a method of ``info`` sees: its members and its bases' - the nearest winning."""
        for klass in reversed(self._lineage(info)):
            for symbol in [*_data_symbols(klass), *_method_symbols(klass)]:
                symbols[symbol.name] = symbol

    def _lineage(self, info: ClassInfo) -> list[ClassInfo]:
        """``info`` and the macro's classes it derives from, nearest first."""
        found = [info]
        for base in info.decl.bases:
            other = self.program.classes.get(base.ctype.name)
            if other is not None and other not in found:
                found.extend(self._lineage(other))
        return found

    def _class_body(self, info: ClassInfo) -> None:
        spellings = self.program.templates.spellings
        values = _enumerators(self._lineage(info))
        self.out.line(f"_cxx_layout_ = {_layout(info, spellings, values)!r}", info.decl.where)
        self.out.line(f"_cxx_methods_ = {_methods_layout(info, spellings)!r}", info.decl.where)
        self.out.line("_cxx_destroyed = False", info.decl.where)
        self.out.line("_cxx_released = False", info.decl.where)
        # An enumerator's value and a static's initialiser are written in the class's own
        # Python body, which runs before the class's name is bound: a member read there is the
        # body's own name - ``N = Nbins + 2`` - and ``A.Nbins`` would be the ``NameError`` of a
        # class that is not there yet. The enumerators come first, since a static may name one.
        with self.scoped() as body:
            self._body_names(info, body)
            self._enumerators_and_statics(info)
        self._static_writes(info)
        self._constructors(info)
        for name, funcs in info.methods.items():
            self._method(info, name, [func for func in funcs if func.body is not None])
        self._finalizer(info)

    def _body_names(self, info: ClassInfo, scope: Scope) -> None:
        """The names a class's Python body can read by themselves: its enumerators as the body
        writes them, and its static members as an attribute is spelled."""
        for name in info.constants:
            scope.add(Symbol(name, "constant", name, CType("int")))
        for name, var in info.statics.items():
            scope.add(Symbol(name, "static", member_name(name), self.declared_type(var)))

    def _enumerators_and_statics(self, info: ClassInfo) -> None:
        """An enumerator's line and a static member's initialiser, in the order the class
        declares them - which is the order C++ needs, a static naming only what came before."""
        for decl in info.decl.members:
            if isinstance(decl, EnumDecl):
                for name, value in self.enum_values(decl):
                    self.out.line(f"{name} = {value}", decl.where)
        for name, var in info.statics.items():
            ctype = self.declared_type(var)
            self.out.line(f"{member_name(name)} = {self.initial(var, ctype)}", var.where)

    def _static_writes(self, info: ClassInfo) -> None:
        """``obj.counter = 1`` of a static data member writes the class, as C++ writes it.

        The write goes to the class that declares the member, found along ``type(self)``'s
        lineage: a subclass inherits this ``__setattr__`` rather than getting one of its own,
        so ``D().counter = 1`` reaches the base's ``counter`` as C++'s does, not a field of
        ``D`` - and a subclass declaring its own ``counter`` hides the base's, nearest first.
        """
        names = sorted(member_name(name) for name in info.statics)
        if not names:
            return
        self.out.line(f"_cxx_statics_ = {names!r}", info.decl.where)
        self.out.blank()
        self.out.line("def __setattr__(self, name, value):", info.decl.where)
        with self.out.indented():
            self.out.line("for declared in type(self).__mro__:")
            with self.out.indented():
                self.out.line("if name in declared.__dict__.get('_cxx_statics_', ()):")
                with self.out.indented():
                    self.out.line("setattr(declared, name, value)")
                    self.out.line("return")
            self.out.line("object.__setattr__(self, name, value)")

    def _finalizer(self, info: ClassInfo) -> None:
        """``del obj`` runs the destructor, as cppyy's proxy does for the instance it owns."""
        if not any(func.body is not None for func in info.methods.get("~" + info.name, [])):
            return
        self.out.blank()
        self.out.line("def __del__(self):", info.decl.where)
        with self.out.indented():
            self.out.line("if self._cxx_released:")
            with self.out.indented():
                self.out.line("return")
            self.out.line("self._destruct()", info.decl.where)

    def _constructors(self, info: ClassInfo) -> None:
        ctors = [f for f in info.methods.get(info.name, []) if f.kind == "constructor"]
        defined = [func for func in ctors if func.body is not None]
        if not defined:
            self._default_init(info, aggregate=not ctors and not info.decl.bases)
            return
        self.out.blank()
        self.overloaded("__init__", "__init__", defined, self._constructor(info), method=True)

    def _constructor(self, info: ClassInfo) -> Callable[[Function, str], None]:
        """How each constructor of ``info`` is written: as it is, or - a member template, whose
        arguments a call made of a ``T *`` is not known to give - as a stand-in refusing the
        uses the translator cannot work out. A template standing in for itself alone leaves the
        class's other constructors there to call, which ROOT's ``TClass::New`` still needs."""

        def write(func: Function, py: str) -> None:
            def build() -> None:
                self.function(func, py, method=True, prologue=lambda: self.initialise(info, func))

            if func.template is None:
                build()
            else:
                self.generic(build, lambda why: self._uninstantiated_method(func, py, why))

        return write

    def _default_init(self, info: ClassInfo, aggregate: bool) -> None:
        """The constructor C++ writes when a class declares none."""
        fields = list(info.fields.values())
        params = ["self"]
        if aggregate:
            params += [f"{python_name(var.name)}=None" for var in fields]
        self.out.blank()
        self.out.line(f"def __init__({', '.join(params)}):", info.decl.where)
        with self.out.indented(), self.scoped("function"):
            self._base_inits(info, {})
            for var in fields:
                value = self.field_value(var, None)
                own = python_name(var.name)
                if aggregate:
                    value = f"{value} if {own} is None else {own}"
                self.out.line(f"self.{member_name(var.name)} = {value}", var.where)

    def initialise(self, info: ClassInfo, func: Function) -> None:
        """What runs before a constructor's body: its bases, then its members, in C++'s order."""
        inits = dict(func.inits)
        own = inits.get(info.name)
        if own is not None:
            args = ", ".join(self.value(arg) for arg in own)
            self.out.line(f"self.__init__({args})", func.where)
            return
        self._base_inits(info, inits)
        for var in info.fields.values():
            self.out.line(
                f"self.{member_name(var.name)} = {self.field_value(var, inits)}", var.where
            )

    def _base_inits(self, info: ClassInfo, inits: dict[str, list[Expr]]) -> None:
        origins = self.program.templates.origins
        # ``: ns::Box<int>(9)`` initialises the base ``Box<int>``, whose key is ``Box__int``.
        unqualified = {written.split("::")[-1]: args for written, args in inits.items()}
        for base in info.decl.bases:
            name = base.ctype.name
            short = origins.get(name, name).split("::")[-1]
            args = inits.get(name, unqualified.get(short, []))
            call = ", ".join(["self", *(self.value(arg) for arg in args)])
            self.out.line(f"{self._base(base.ctype)}.__init__({call})", base.where)

    def field_value(self, var: VarDecl, inits: dict[str, list[Expr]] | None) -> str:
        """A member's first value: from the initialiser list, the class body, or C++'s default."""
        ctype = var.ctype.value()
        if inits and var.name in inits:
            args = inits[var.name]
            if var.ctype.reference and len(args) == 1:
                return self.reference(args[0]) if ctype.scalar else self.value(args[0])
            return self.constructed(ctype, args, VarDecl(var.where, var.name, ctype, style="()"))
        if var.style is not None:
            return self.initial(var, self.declared_type(var))
        if ctype.dims:
            return self.array_initial(var, ctype)
        if ctype.name in self.template_names:
            # C++ leaves a member of a template parameter's type unset: nothing may read it yet.
            return "None"
        return self.default(ctype, var)

    def _method(self, info: ClassInfo, name: str, funcs: list[Function]) -> None:
        funcs = [func for func in funcs if func.kind not in ("constructor",)]
        # ``operator++()`` and ``operator++(int)`` are two Python methods, not one set.
        named: dict[str, list[Function]] = {}
        for func in funcs:
            named.setdefault(self._method_name(func), []).append(func)
        for py, group in named.items():
            self._method_set(name, py, group)

    def _method_set(self, name: str, py: str, funcs: list[Function]) -> None:
        static = funcs[0].static
        self.out.blank()
        decorator = "@staticmethod" if static and len(funcs) == 1 else None

        once = self._destroyed_once if py == "_destruct" else None

        def write() -> None:
            self.overloaded(
                name,
                py,
                funcs,
                lambda func, each: self.function(
                    func,
                    each,
                    method=not static,
                    decorator=decorator,
                    prologue=once,
                    epilogue=self._releases(func) if py != "_destruct" else None,
                ),
                method=not static,
            )

        if any(func.template is not None for func in funcs):
            self.generic(write, lambda why: self._uninstantiated_method(funcs[0], py, why))
        else:
            write()

    def _destroyed_once(self) -> None:
        """``delete``, a scope's end and ``del`` all reach ``_destruct``: the first wins."""
        self.out.line("if self._cxx_destroyed:")
        with self.out.indented():
            self.out.line("return")
        self.out.line("self._cxx_destroyed = True")

    def _releases(self, func: Function) -> Callable[[], None] | None:
        """The parameters the heuristic memory policy assumes C++ keeps: non-const pointers."""
        names = [
            python_name(param.name or f"arg{index}")
            for index, param in enumerate(func.params)
            if param.ctype.pointer and not param.ctype.const and not param.ctype.value().scalar
        ]
        if not names:
            return None

        def epilogue() -> None:
            for name in names:
                self.out.line(f"release({name})", func.where)

        return epilogue

    def _uninstantiated_method(self, func: Function, py: str, why: str) -> None:
        """A member template whose body needs its arguments, used where they are not known."""
        self.out.line(f"def {py}(*args, **kwargs):", func.where)
        with self.out.indented():
            what = f"the member template {func.name}"
            self.out.line(f"uninstantiated({what!r}, {why!r})", func.where)

    def _method_name(self, func: Function) -> str:
        if func.kind == "destructor":
            return "_destruct"
        if func.kind != "operator":
            return member_name(func.name)
        dunder = conversion(func.name)
        names = DUNDERS.get(func.name) or ((None, dunder) if dunder is not None else None)
        if names is None:
            spelled = " ".join(func.name[8:].split())
            raise self.refuse(f"the operator {spelled} defined for a class", func)
        binary, unary = names
        chosen = binary if func.params else unary
        if chosen is None:
            raise self.refuse(f"the operator {func.name[8:]} with this many operands", func)
        return chosen


def _data_symbols(klass: ClassInfo) -> list[Symbol]:
    """A class's members, static members and enumerators, as its methods see them."""
    found = [Symbol(n, "field", member_name(n), v.ctype.value(), cell=bool(
        v.ctype.reference and v.ctype.value().scalar)) for n, v in klass.fields.items()]
    found += [Symbol(n, "static", f"{klass.name}.{n}", v.ctype) for n, v in klass.statics.items()]
    found += [Symbol(n, "static", f"{klass.name}.{n}", CType("int")) for n in klass.constants]
    return found


def _method_symbols(klass: ClassInfo) -> list[Symbol]:
    """A class's methods - not its constructors - as ``self.m`` or, static, ``Class.m``."""
    found = []
    for name, funcs in klass.methods.items():
        if name in (klass.name, "~" + klass.name):
            continue
        static = any(func.static for func in funcs)
        kind, py = ("static", f"{klass.name}.{name}") if static else ("method", name)
        found.append(Symbol(name, kind, py, owner=klass.name))
    return found


def _constant(value: Expr) -> int | None:
    """An enumerator's value when it is a number written out, ``3`` or ``-1``."""
    if isinstance(value, Literal) and value.kind in ("int", "char"):
        return int(value.value)
    if isinstance(value, Unary) and value.op == "-" and isinstance(value.operand, Literal):
        inner = _constant(value.operand)
        return -inner if inner is not None else None
    return None


#: What a class tells a tree about itself: its C++ name, its bases', and each data member's
#: name, type as written, trailing comment and array dimensions, in declaration order.
Layout = tuple[str, tuple[str, ...], tuple[tuple[str, str, str, tuple[int | None, ...]], ...]]


#: One function a class declares, as a reflection dictionary describes it: its C++ name, its
#: return type as written, each parameter's type, and whether it is const, static, inline and
#: virtual, a template, and its kind - ``function``, ``constructor``, ``destructor``, ``operator``.
Method = tuple[str, str, tuple[str, ...], bool, bool, bool, bool, bool, str]


def _methods_layout(info: ClassInfo, spellings: dict[str, str]) -> tuple[Method, ...]:
    """A class's functions as a reflection dictionary describes them, so that the ``TMethod``
    made from one knows its signature - what a program searching the class by prototype and
    reading a method's return type asks about. A constructor returns the class it makes and a
    destructor nothing, as ROOT's ``TMethod`` says of them."""
    found: list[Method] = []
    for func in info.overloads.values():
        found.append((
            func.name, _function_return(info, func, spellings),
            tuple(_declared_type(param.ctype, spellings) for param in func.params),
            bool(func.const), bool(func.static), bool(func.inlined),
            bool(func.virtual), func.template is not None, func.kind,
        ))
    return tuple(found)


def _function_return(info: ClassInfo, func: Function, spellings: dict[str, str]) -> str:
    """What a function returns, as a signature writes it: the class a constructor builds."""
    if func.kind == "constructor":
        return spellings.get(info.name, info.name)
    if func.kind == "destructor":
        return "void"
    return _declared_type(func.returns, spellings)


def _declared_type(ctype: CType, spellings: dict[str, str]) -> str:
    """A type as a function's signature spells it: the typedef it was written as, a ``&`` and a
    leading ``const`` kept - which is what a ``TMethod`` reports its return type as."""
    text = _spelled(ctype, spellings) + "&" * int(ctype.reference)
    return f"const {text}" if ctype.const and not text.startswith("const ") else text


def _layout(info: ClassInfo, spellings: dict[str, str], values: dict[str, int]) -> Layout:
    """A class's data members as a reflection dictionary would describe them, for a library
    that writes objects member by member - an instance of a template by its C++ name."""
    bases = tuple(spellings.get(base.ctype.name, base.ctype.name) for base in info.decl.bases)
    members = tuple(
        (name, _spelled(var.ctype, spellings), var.comment, _dimensions(var, values))
        for name, var in info.fields.items()
    )
    return spellings.get(info.name, info.name), bases, members


def _dimensions(var: VarDecl, values: dict[str, int]) -> tuple[int | None, ...]:
    """A member's array lengths, each resolved as far as a compile-time constant goes."""
    return tuple(_dimension(each, values) for each in var.ctype.dims)


def _dimension(size: Any, values: dict[str, int]) -> int | None:
    """An array's length as declared: a number, the value of an enumerator, or ``None`` for one
    worked out at run time."""
    if isinstance(size, int):
        return size
    known = _constant(size)
    if known is not None:
        return known
    return values.get("::".join(size.parts)) or values.get(size.parts[-1]) \
        if isinstance(size, Name) else None


def _enumerators(lineage: list[ClassInfo]) -> dict[str, int]:
    """Every enumerator the class and its bases declare, with its value where it is a number.

    ``enum {kSize = 10}; Int_t fMeasures[kSize];`` is an array of ten, and ROOT's layout
    says ten - so a dimension written as an enumerator's name is resolved here rather
    than left to whatever reads the layout.
    """
    found: dict[str, int] = {}
    for info in lineage:
        for decl in info.decl.members:
            if isinstance(decl, EnumDecl):
                _counted_up(decl, info.name, found)
    return found


def _counted_up(decl: EnumDecl, owner: str, found: dict[str, int]) -> None:
    """One enum's enumerators, counted on from the last as C++ counts them, under both the
    name it is written by and the qualified one a macro may write instead."""
    last: int | None = -1
    for name, value in decl.items:
        last = None if last is None else last + 1 if value is None else _constant(value)
        if last is None:
            continue
        for spelled in (name, f"{decl.name}::{name}" if decl.name else f"{owner}::{name}"):
            found.setdefault(spelled, last)


def _template_argument(arg: Any, spellings: dict[str, str]) -> str:
    """One template argument as C++ writes it: a type, a number, or ``?`` when the argument is
    an expression the layout cannot say (``std::array<int, n>`` of a run-time length)."""
    if isinstance(arg, CType):
        return _spelled(arg, spellings)
    number = _constant(arg)
    return "?" if number is None else str(number)


def _spelled(ctype: CType, spellings: dict[str, str]) -> str:
    """A member's type as C++ spells it: the typedef it was written as, its arguments, stars."""
    args = ", ".join(_template_argument(arg, spellings) for arg in ctype.args)
    name = ctype.written or spellings.get(ctype.name, ctype.name)
    return (f"{name}<{args}>" if ctype.args else name) + "*" * ctype.pointer

