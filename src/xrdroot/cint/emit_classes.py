"""C++ classes and enums as Python classes and constants.

A class is a Python class deriving from the same bases - the macro's own,
or ROOT's (``class MySelector : public TSelector`` derives from
``ROOT.TSelector``). Its constructor is ``__init__``, running the base
constructors and the member initialisers C++ would, in the order C++ does;
a struct with no constructor takes its members positionally, as aggregate
initialisation does. Operators become the dunder methods Python calls, the
destructor ``_destruct`` (which ``delete`` calls). ``ClassDef`` and
``ClassImp`` have already gone, in the preprocessor.
"""

from __future__ import annotations

from .ctype import CType
from .emit_funcs import FunctionEmitter
from .nodes import ClassDecl, EnumDecl, Expr, Function, Literal, Unary, VarDecl
from .operators import DUNDERS
from .program import ClassInfo
from .symbols import Symbol, python_name

__all__ = ["ClassEmitter"]


class ClassEmitter(FunctionEmitter):
    """The emitter's layer for classes and enums."""

    def top(self, decl: object) -> None:
        raise NotImplementedError

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
        if decl.forward:
            return
        info = self.program.classes[decl.name]
        for nested in info.types:
            self.top(nested)
        py = self._class_symbol(decl.name).py
        bases = [self._base(base.ctype) for base in decl.bases]
        self.out.blank(2 if self.scope.kind == "module" else 1)
        self.out.line(f"class {py}({', '.join(bases)}):" if bases else f"class {py}:", decl.where)
        with self.out.indented(), self.scoped("class", klass=decl.name) as scope:
            self._members(info, scope.symbols)
            self._class_body(info)
        self.declarators(decl.declarators)

    def declarators(self, decls: list[VarDecl]) -> None:
        """``struct P {...} p;``: the variables declared with a class or enum."""
        raise NotImplementedError

    def _class_symbol(self, name: str) -> Symbol:
        symbol = self.lookup(name)
        if symbol is None:
            symbol = self.scope.add(Symbol(name, "class", python_name(name)))
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
        for name, var in info.statics.items():
            ctype = self.declared_type(var)
            self.out.line(f"{python_name(name)} = {self.initial(var, ctype)}", var.where)
        for decl in info.decl.members:
            if isinstance(decl, EnumDecl):
                for name, value in self.enum_values(decl):
                    self.out.line(f"{name} = {value}", decl.where)
        self._constructors(info)
        for name, funcs in info.methods.items():
            self._method(info, name, [func for func in funcs if func.body is not None])

    def _constructors(self, info: ClassInfo) -> None:
        ctors = [f for f in info.methods.get(info.name, []) if f.kind == "constructor"]
        defined = [func for func in ctors if func.body is not None]
        if not defined:
            self._default_init(info, aggregate=not ctors and not info.decl.bases)
            return
        self.out.blank()
        self.overloaded(
            "__init__",
            "__init__",
            defined,
            lambda func, py: self.function(
                func, py, method=True, prologue=lambda: self.initialise(info, func)
            ),
            method=True,
        )

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
                self.out.line(f"self.{own} = {value}", var.where)

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
                f"self.{python_name(var.name)} = {self.field_value(var, inits)}", var.where
            )

    def _base_inits(self, info: ClassInfo, inits: dict[str, list[Expr]]) -> None:
        for base in info.decl.bases:
            args = inits.get(base.ctype.name, inits.get(base.ctype.name.split("::")[-1], []))
            call = ", ".join(["self", *(self.value(arg) for arg in args)])
            self.out.line(f"{self._base(base.ctype)}.__init__({call})", base.where)

    def field_value(self, var: VarDecl, inits: dict[str, list[Expr]] | None) -> str:
        """A member's first value: from the initialiser list, the class body, or C++'s default."""
        ctype = var.ctype.value()
        if inits and var.name in inits:
            args = inits[var.name]
            if var.ctype.reference and len(args) == 1:
                return self.value(args[0])
            return self.constructed(ctype, args, VarDecl(var.where, var.name, ctype, style="()"))
        if var.style is not None:
            return self.initial(var, self.declared_type(var))
        if ctype.dims:
            return self.array_initial(var, ctype)
        return self.default(ctype, var)

    def _method(self, info: ClassInfo, name: str, funcs: list[Function]) -> None:
        funcs = [func for func in funcs if func.kind not in ("constructor",)]
        if not funcs:
            return
        py = self._method_name(funcs[0])
        static = funcs[0].static
        self.out.blank()
        decorator = "@staticmethod" if static and len(funcs) == 1 else None
        self.overloaded(
            name,
            py,
            funcs,
            lambda func, each: self.function(func, each, method=not static, decorator=decorator),
            method=not static,
        )

    def _method_name(self, func: Function) -> str:
        if func.kind == "destructor":
            return "_destruct"
        if func.kind != "operator":
            return python_name(func.name)
        names = DUNDERS.get(func.name)
        if names is None:
            raise self.refuse(f"the operator {func.name[8:]} defined for a class", func)
        binary, unary = names
        chosen = binary if func.params else unary
        if chosen is None:
            raise self.refuse(f"the operator {func.name[8:]} with this many operands", func)
        return chosen


def _data_symbols(klass: ClassInfo) -> list[Symbol]:
    """A class's members, static members and enumerators, as its methods see them."""
    found = [Symbol(n, "field", python_name(n), v.ctype.value()) for n, v in klass.fields.items()]
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
