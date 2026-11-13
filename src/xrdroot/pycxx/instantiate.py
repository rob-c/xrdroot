"""Templates instantiated: a class or function template made concrete for one set of arguments.

C++ compiles a template once for each set of template arguments it is used
with, and so does this translator. Before any Python is written, a pass over
the whole program finds every class template named with arguments that are
all known - ``Box<int>``, ``Grid<double, 3>``, not ``Box<T>`` inside another
template - and makes the class C++ would: the template's syntax tree copied
with each parameter replaced by its argument (:mod:`.substitute`), from the
partial specialisation that matches best if one does, or the explicit one
written for exactly those arguments. The copy is an ordinary class of the
program's, under a *key* that is a Python identifier, ``Box__int``, and every
type, constructor call and ``Box<int>::member`` that named it now names the
key; its C++ spelling, ``Box<int>``, is kept for ``typeid`` and for the
library's reflection. A function template is instantiated where it is
called, its arguments given or deduced from the types of the call's
(:mod:`.emit_templates`). Inside an instantiation ``T`` is a type like any
other, so ``sizeof(T)``, ``std::numeric_limits<T>``, ``T x;`` and the
arithmetic of a ``T`` are C++'s; each instantiation has static members of
its own.

The memo is the key: a set of arguments spelled the same way is one
instantiation, made once.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .ctype import CType
from .errors import Refusal, Where
from .nodes import Call, ClassDecl, Expr, Function, Literal, Name, Node, Typedef, VarDecl
from .sizes import constant
from .substitute import Bindings, as_literal, bound, copied, substituted

__all__ = ["ClassTemplate", "Templates", "argument_text", "mangled", "deduce_call"]


@dataclass(eq=False)
class ClassTemplate:
    """A class template: its primary definition and what is written for it elsewhere."""

    decl: ClassDecl
    #: The partial specialisations, ``template <class T> struct Box<T*>``.
    partials: list[ClassDecl] = field(default_factory=list)
    #: Its member functions defined outside it, ``template <class T> T Box<T>::get()``.
    members: list[Function] = field(default_factory=list)
    #: Its static members defined outside it, with that definition's parameter names.
    statics: list[tuple[list[str], VarDecl]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.decl.name

    @property
    def parameters(self) -> list[str]:
        return list(self.decl.template or [])


def mangled(spelling: str) -> str:
    """A C++ spelling made a Python identifier no C++ name has: ``Box<int*>`` is
    ``Box__int_p``, its double underscore a name C++ reserves to the implementation."""
    text = spelling.replace("*", " p ").replace("&", " r ").replace("-", " minus ")
    head, _, rest = text.partition("<")
    tail = re.sub(r"\W+", "_", rest).strip("_")
    return re.sub(r"\W+", "_", head).strip("_") + "__" + (tail or "0")


def argument_text(arg: Any, spellings: dict[str, str], program: Any) -> str | None:
    """A template argument as C++ spells it in a type's name: ``int``, ``std::vector<double>``,
    ``3`` - or ``None`` when it is a value this translator cannot work out."""
    if isinstance(arg, CType):
        return type_spelling(arg, spellings, program)
    if isinstance(arg, Literal) and arg.kind == "bool":
        return "true" if arg.value else "false"
    value = constant(arg, program)
    return str(value) if value is not None else None


def _numbered(arg: Any, program: Any) -> Any:
    """A value argument that names a constant as the constant - kept as written otherwise."""
    if isinstance(arg, Name):
        value = constant(arg, program)
        return arg if value is None else as_literal(value, arg.where)
    return arg


def type_spelling(ctype: CType, spellings: dict[str, str], program: Any) -> str | None:
    """A type as C++ spells it: ``const std::pair<int, double>*``, an instantiation's key
    spelled as the template-id it stands for."""
    name = ctype.enum or spellings.get(ctype.name, ctype.name)
    args = [argument_text(arg, spellings, program) for arg in ctype.args]
    dims = [constant(dim, program) for dim in ctype.dims]
    if None in args or None in dims:
        return None
    listed = f"<{', '.join(map(str, args))}>" if args else ""
    sizes = "".join(f"[{dim}]" for dim in dims)
    return "const " * ctype.const + name + listed + "*" * ctype.pointer + sizes


class Templates:
    """The program's templates, and every instantiation of them made so far."""

    def __init__(self, program: Any) -> None:
        self.program = program
        self.classes: dict[str, ClassTemplate] = {}
        #: The C++ spelling of each instantiation by its key, and its template's name.
        self.spellings: dict[str, str] = {}
        self.origins: dict[str, str] = {}
        #: The key of each spelling.
        self.keys: dict[str, str] = {}
        #: The members explicitly specialised for one instance, by the instance's spelling.
        self.members: dict[str, list[Function]] = {}
        #: The function instances, by key; the explicit specialisations, by their spelling.
        self.functions: dict[str, Function] = {}
        self.specialisations: dict[str, Function] = {}
        #: Each class instance's template and its arguments, by key.
        self.arguments: Known = {}
        #: The class instantiations, each as it was finished - so after those it needs.
        self.made: list[str] = []
        #: Where each class instantiation is written: before the declaration at this index of
        #: the program's, after its template's own.
        self.points: dict[str, int] = {}
        #: The index of the declaration being looked over, and of each template's definition.
        self.current = 0
        self.defined: dict[str, int] = {}

    # -- collecting ------------------------------------------------------------

    def add_class(self, decl: ClassDecl) -> None:
        self.classes[decl.name] = ClassTemplate(decl)
        self.defined[decl.name] = self.current

    def partial(self, decl: ClassDecl) -> None:
        """A partial specialisation, kept with its primary template."""
        template = self.classes.get(decl.name)
        if template is None:
            why = f"a partial specialisation of {decl.name}, a template the program does not define"
            raise Refusal(why, decl.where)
        template.partials.append(decl)

    def explicit(self, decl: ClassDecl) -> ClassDecl:
        """``template <> struct Box<int> {...}``: the class for exactly those arguments - an
        ordinary class, under its key."""
        assert decl.specialises is not None
        template = self.classes.get(decl.name)
        args = self._arguments(template, decl.specialises) if template else decl.specialises
        key = self.key(decl.name, args or decl.specialises)
        if key is None:
            why = f"a specialisation of {decl.name} for arguments this translator cannot work out"
            raise Refusal(why, decl.where)
        self.arguments[key] = (decl.name, list(args or decl.specialises))
        return _renamed(copied(decl, {decl.name: CType(key)}), key)

    def specialised(self, func: Function) -> None:
        """``template <> int f<int>(int)``: what ``f<int>`` is, in place of the template's."""
        assert func.specialises is not None
        texts = [argument_text(arg, self.spellings, self.program) for arg in func.specialises]
        self.specialisations[f"{func.name}<{', '.join(map(str, texts))}>"] = func

    def instance_member(self, func: Function) -> bool:
        """``Box<int>::f()`` defined outside the class: a member of an explicit specialisation
        - given to it now - or the explicit specialisation of one member, kept for the
        instance ``Box<int>`` to have in place of the template's. Is ``func`` one?"""
        if not func.scope or not func.scope_args or func.template:
            return False
        spelling = self._owner_spelling(func)
        info = self.program.classes.get(self.keys.get(spelling or ""))
        if info is not None:
            info.add_method(_renamed_function(func, func.scope[-1], info.name))
            return True
        if spelling is None or func.scope[-1] not in self.classes:
            return False
        self.members.setdefault(spelling, []).append(func)
        return True

    def _owner_spelling(self, func: Function) -> str | None:
        """How the class ``func`` is a member of is spelled: ``Box<int>`` of ``Box<int>::f``."""
        template = self.classes.get(func.scope[-1])
        args = self._arguments(template, func.scope_args) if template else func.scope_args
        return self._spelling(func.scope[-1], args) if args is not None else None

    def member(self, func: Function) -> bool:
        """An out-of-class member of a class template: kept, to be part of each instance."""
        template = self.classes.get(func.scope[-1]) if func.scope else None
        if template is None or func.template is None:
            return False
        template.members.append(func)
        return True

    def static(self, names: list[str], var: VarDecl, where: Where) -> None:
        """``template <class T> int Box<T>::count = 0;``: each instance's ``count``."""
        template = self.classes.get(var.name.split("::")[-2])
        if template is None:
            return
        if not names:
            why = f"a static member {var.name} defined for one instance of a class template"
            raise Refusal(why, where)
        template.statics.append((names, var))

    # -- keys ------------------------------------------------------------------

    def _spelling(self, name: str, args: list[Any]) -> str | None:
        texts = [argument_text(arg, self.spellings, self.program) for arg in args]
        if any(text is None for text in texts):
            return None
        return f"{name}<{', '.join(str(text) for text in texts)}>"

    def key(self, name: str, args: list[Any]) -> str | None:
        """The key of the instantiation of ``name`` for ``args``: made up the first time."""
        spelling = self._spelling(name, args)
        if spelling is None:
            return None
        found = self.keys.get(spelling)
        if found is not None:
            return found
        key = mangled(spelling)
        while key in self.spellings or key in self.program.classes:
            key += "_"
        self.keys[spelling], self.spellings[key], self.origins[key] = key, spelling, name
        return key

    # -- instantiating ---------------------------------------------------------

    def instance(self, name: str, args: list[Any], params: frozenset[str]) -> str | None:
        """The key of ``name<args>`` - instantiated now if it has not been - or ``None`` if
        ``name`` is no class template's, or ``args`` are not all known. A namespace the
        program declares is no part of a name, as the program's namespaces are flattened."""
        if not args or not all(_concrete(arg, params) for arg in args):
            return None
        own = name if name.startswith("std::") else name.split("::")[-1]
        template = self.classes.get(own)
        if template is None:
            return self._specialised(own, args)
        return self._made_for(template, args)

    def _made_for(self, template: ClassTemplate, args: list[Any]) -> str | None:
        """The key of the instance of ``template`` for ``args``, made now if it was not."""
        full = self._arguments(template, args)
        key = self.key(template.name, full) if full is not None else None
        if full is not None and key is not None and key not in self.program.classes:
            self._instantiate(template, full, key)
        return key

    def _specialised(self, name: str, args: list[Any]) -> str | None:
        """``std::hash<Mine>``, or a template declared and never defined: an explicit
        specialisation the program wrote, if there is one for ``args``."""
        spelling = self._spelling(name, args)
        known = self.keys.get(spelling) if spelling is not None else None
        return known if known in self.program.classes else None

    # -- function templates ------------------------------------------------------

    def function(
        self, func: Function, bindings: Bindings, owner: str | None = None
    ) -> tuple[str, Function | None] | None:
        """The key of the instance of the function template ``func`` - a member template of
        the class ``owner``, if given - for ``bindings``, and the instance itself the first
        time it is asked for; or ``None`` when an argument is a value not worked out here."""
        names = list(func.template or [])
        spelled = f"{owner}::{func.name}" if owner is not None else func.name
        key = self.key(spelled, [bindings[name] for name in names])
        if key is None:
            return None
        if key in self.functions:
            return key, None
        written = self.specialisations.get(self.spellings[key])
        made = copied(written, {}) if written is not None else copied(func, self.typed(bindings))
        made.name, made.template, made.values, made.defaults = key, None, [], []
        made.specialises = None
        self.functions[key] = made
        self.resolve(made, frozenset())
        return key, made

    # -- resolving ---------------------------------------------------------------

    def resolve(self, node: Any, params: frozenset[str]) -> None:
        """Every type and name in ``node`` that is a class template with known arguments made
        its instance's key, ``params`` being the template parameters in scope there."""
        if isinstance(node, CType):
            self._type(node, params)
        elif isinstance(node, list):
            for item in node:
                self.resolve(item, params)
        elif isinstance(node, Node) and not _partial(node):
            self._node(node, params | frozenset(getattr(node, "template", None) or ()))

    def _node(self, node: Node, params: frozenset[str]) -> None:
        if isinstance(node, Call):
            self._typed_test(node, params)
        if isinstance(node, Name):
            self._name(node, params)
        for item in dataclasses.fields(node):
            if item.name != "where":
                self.resolve(getattr(node, item.name), params)

    def _typed_test(self, call: Call, params: frozenset[str]) -> None:
        """GoogleTest's ``RegisterTyped<Test, ::testing::Types<int, double>>(...)``: the test
        class template instantiated for each type, the instances handed to the call - or,
        for ``RegisterTypedSuite<Types>(prefix, "Suite", ...)``, each test of the suite's,
        with its name and the type's place in the list."""
        func = call.func
        if not isinstance(func, Name) or func.last not in TYPED or not func.targs:
            return
        types = func.targs[-1]
        tests = func.targs[:-1] or self._suite(call)
        func.targs = None
        for test in tests:
            call.args.extend(self._typed(test, types, call, params))

    def _typed(self, test: Any, types: Any, call: Call, params: frozenset[str]) -> list[Expr]:
        """One typed test's instances, each named by its key - and, for a suite's, after
        the type's place in the list and the test's name."""
        named = isinstance(call.func, Name) and call.func.last == "RegisterTypedSuite"
        name = _test_name(test.name, call) if named else ""
        found: list[Expr] = []
        for index, kind in enumerate(types.args if isinstance(types, CType) else []):
            key = self.instance(test.name, [kind], params)
            if named:
                found += [
                    Literal(call.where, "int", index, "int"),
                    Literal(call.where, "str", name, "char*"),
                ]
            found.append(Name(call.where, [str(key)]))
        return found

    def _suite(self, call: Call) -> list[Any]:
        """The typed tests of the suite a ``RegisterTypedSuite`` call names, as written."""
        fixture = _fixture(call)
        return [
            CType(name)
            for name, template in self.classes.items()
            if template.decl.bases
            and template.decl.bases[0].ctype.name == fixture
            and name.startswith(f"{fixture}_")
            and name.endswith("_Test")
        ]

    def _type(self, ctype: CType, params: frozenset[str]) -> None:
        for arg in [*ctype.args, *ctype.scope_args]:
            self.resolve(arg, params)
        key = self.instance(ctype.name, ctype.args, params)
        if key is not None:
            ctype.name, ctype.args = key, []
        elif ctype.scope_args:
            self._nested_type(ctype, params)

    def _nested_type(self, ctype: CType, params: frozenset[str]) -> None:
        """``List<int>::Node``: the class ``List<int>`` has inside it."""
        head, _, last = ctype.name.rpartition("::")
        key = self.instance(head.split("::")[-1], ctype.scope_args, params)
        if key is not None:
            nested = f"{key}__{last}"
            ctype.name = nested if nested in self.program.classes else f"{key}::{last}"
            ctype.scope_args = []

    def _name(self, name: Name, params: frozenset[str]) -> None:
        """``Box<int>(1)`` and ``Box<int>::count``: the class the arguments are a template's."""
        self.resolve(name.targs or [], params)
        self.resolve(name.scope_targs or [], params)
        own = "::".join(name.parts) if name.parts[0] == "std" else name.last
        key = self.instance(own, name.targs or [], params)
        if key is not None and name.targs is not name.scope_targs:
            name.parts, name.targs = [key], None
        elif len(name.parts) > 1 and name.scope_targs:
            self._scope(name, params)

    def _scope(self, name: Name, params: frozenset[str]) -> None:
        """``Box<int>::count``: the class whose member is named, an instance."""
        key = self.instance(name.parts[-2], name.scope_targs or [], params)
        if key is not None:
            name.parts[-2] = key
            name.targs = None if name.targs is name.scope_targs else name.targs
            name.scope_targs = None

    def _arguments(self, template: ClassTemplate, args: list[Any]) -> list[Any] | None:
        """``args`` with the template's defaults for those not given."""
        args = [_numbered(arg, self.program) for arg in valued(template.decl, args)]
        bindings = bound(template.parameters, template.decl.defaults, args)
        if bindings is None:
            return None
        return [bindings[name] for name in template.parameters]

    def typed(self, bindings: Bindings) -> Bindings:
        """``bindings`` with the types each class of the program's given for a parameter
        declares: ``T::value_type`` of a ``T`` that is ``Traits`` is ``Traits``'s typedef."""
        found = dict(bindings)
        for name, value in bindings.items():
            info = self.program.classes.get(value.name) if isinstance(value, CType) else None
            for member in info.decl.members if info is not None else []:
                if isinstance(member, Typedef):
                    found[f"{name}::{member.name}"] = member.ctype
        return found

    def _instantiate(self, template: ClassTemplate, args: list[Any], key: str) -> None:
        chosen, bindings = self._chosen(template, args)
        bindings = self.typed(bindings)
        bindings[template.name] = CType(key)
        own = _own_types(chosen, bindings)
        bindings.update(own)
        nested = self._nested(chosen, key, bindings)
        decl = _renamed(copied(chosen, bindings), key)
        decl.members = [
            _renamed(member, nested[member.name])
            if isinstance(member, ClassDecl) and member.name in nested
            else member
            for member in decl.members
        ]
        decl.template, decl.specialises, decl.values, decl.defaults = None, None, [], []
        decl.members = [*decl.members, *self._members(template, args, key, chosen, own)]
        info = self.program.register(decl)
        info.origin = template.name
        self.arguments[key] = (template.name, args)
        for names, var in template.statics if chosen is template.decl else []:
            own = copied(var, {**dict(zip(names, args, strict=False)), template.name: CType(key)})
            own.name = own.name.split("::")[-1]
            info.statics[own.name] = own
        self.points[key] = max(self.current, self.defined.get(template.name, 0) + 1)
        self.resolve(decl, frozenset())
        self.made.append(key)

    def _nested(self, decl: ClassDecl, key: str, bindings: Bindings) -> dict[str, str]:
        """The classes declared inside a class template, each instance's own: ``Node`` of
        ``List<int>`` is a class named for it, and its name in the template's body names it."""
        nested = {}
        for member in decl.members:
            if isinstance(member, ClassDecl) and not member.template and not member.specialises:
                own = f"{key}__{member.name}"
                nested[member.name] = own
                bindings[member.name] = CType(own)
                self.spellings[own] = f"{self.spellings[key]}::{member.name}"
        return nested

    def _members(
        self,
        template: ClassTemplate,
        args: list[Any],
        key: str,
        chosen: ClassDecl,
        own: Bindings,
    ) -> Iterator[Function]:
        """The template's members defined outside it, for this instance: its own parameters
        named as the definition names them, and the class's own types - ``own`` - in."""
        if chosen is not template.decl:
            return
        for func in [*template.members, *self.members.get(self.spellings[key], [])]:
            names = func.template or []
            bindings: Bindings = {**own, **dict(zip(names, args, strict=False))}
            bindings[template.name] = CType(key)
            made = copied(func, bindings)
            made.template, made.scope = None, [*func.scope[:-1], key]
            yield _renamed_function(made, template.name, key)

    def _chosen(self, template: ClassTemplate, args: list[Any]) -> tuple[ClassDecl, Bindings]:
        """The declaration an instance is made from: the most specialised partial
        specialisation that matches, or the primary template."""
        matches = self._matches(template, args)
        if not matches:
            return template.decl, dict(zip(template.parameters, args, strict=False))
        ranks = [_specificity(partial.specialises or []) for partial, _ in matches]
        if ranks.count(max(ranks)) > 1:
            why = f"an instance of {template.name} more than one partial specialisation matches"
            raise Refusal(why, template.decl.where)
        return matches[ranks.index(max(ranks))]

    def _matches(
        self, template: ClassTemplate, args: list[Any]
    ) -> list[tuple[ClassDecl, Bindings]]:
        """The partial specialisations ``args`` fit, each with its parameters' values."""
        matches = []
        for partial in template.partials:
            patterns = self._arguments(template, partial.specialises or [])
            found = deduce(patterns, args, set(partial.template or []), self.arguments)
            if found is not None:
                matches.append((partial, found))
        return matches


def valued(decl: ClassDecl, args: list[Any]) -> list[Any]:
    """``args`` with each given for a value parameter as a value: ``B<N>`` names ``N``, which
    reads like a type until the template says it is a value."""
    values = {param.name for param in decl.values}
    names = decl.template or []
    return [
        Name(decl.where, arg.name.split("::")) if _named_value(arg, index, names, values) else arg
        for index, arg in enumerate(args)
    ]


def _named_value(arg: Any, index: int, names: list[str], values: set[str | None]) -> bool:
    plain = isinstance(arg, CType) and not (arg.args or arg.pointer or arg.const or arg.dims)
    return plain and index < len(names) and names[index] in values


#: GoogleTest's registrations of typed tests, which name class templates to instantiate.
TYPED = frozenset({"RegisterTyped", "RegisterTypedSuite"})


def _fixture(call: Call) -> str:
    """The suite a ``RegisterTypedSuite(prefix, "Suite", ...)`` call names."""
    suite = call.args[1] if len(call.args) > 1 else None
    return str(suite.value) if isinstance(suite, Literal) else ""


def _test_name(test: str, call: Call) -> str:
    """``Name`` of the class template ``Suite_Name_Test`` a typed test is."""
    return test[len(_fixture(call)) + 1 : -len("_Test")]


def _own_types(decl: ClassDecl, bindings: Bindings) -> Bindings:
    """The typedefs a class template declares, each the type it is in this instance: what
    its members' bodies mean by ``TypeParam``."""
    return {
        member.name: substituted(member.ctype, bindings)
        for member in decl.members
        if isinstance(member, Typedef)
    }


def _renamed(decl: ClassDecl, key: str) -> ClassDecl:
    """A class's copy named ``key``, its constructors and destructor with it."""
    old = decl.name
    decl = dataclasses.replace(decl, name=key)
    decl.members = [
        _renamed_function(member, old, key) if isinstance(member, Function) else member
        for member in decl.members
    ]
    return decl


def _renamed_function(func: Function, old: str, key: str) -> Function:
    if func.kind == "constructor" and func.name == old:
        func.name = key
    elif func.kind == "destructor" and func.name == "~" + old:
        func.name = "~" + key
    return func


def _concrete(arg: Any, params: frozenset[str]) -> bool:
    """Does a template argument name no template parameter of the templates around it?"""
    if isinstance(arg, CType):
        head = arg.name.split("::")[0]
        if arg.name in params or head in params or arg.is_auto:
            return False
        return all(_concrete(inner, params) for inner in arg.args)
    return not any(isinstance(node, Name) and node.parts[0] in params for node in _walk(arg))


def _walk(node: Any) -> Iterator[Any]:
    """``node`` and every node inside it."""
    yield node
    if isinstance(node, list):
        for item in node:
            yield from _walk(item)
    elif isinstance(node, Node):
        for item in dataclasses.fields(node):
            yield from _walk(getattr(node, item.name))


#: Each class instance's template and arguments, by its key: what ``Box<T>`` is matched to.
Known = dict[str, tuple[str, list[Any]]]


def deduce(
    patterns: list[Any] | None, args: list[Any], params: set[str], known: Known
) -> Bindings | None:
    """The values of ``params`` that make ``patterns`` the arguments ``args`` - ``T`` of
    ``<T*>`` for ``<int*>`` - or ``None`` if no values do, or there are no patterns."""
    if patterns is None:
        return None
    found: Bindings = {}
    for pattern, arg in zip(patterns, args, strict=True):
        if not match(pattern, arg, params, found, known):
            return None
    return found if params <= found.keys() else None


def match(pattern: Any, arg: Any, params: set[str], found: Bindings, known: Known) -> bool:
    """Does ``arg`` fit ``pattern``, with its parameters given the values in ``found`` -
    adding those it settles?"""
    if isinstance(pattern, CType) and isinstance(arg, CType):
        return _match_type(pattern, arg, params, found, known)
    if isinstance(pattern, Name) and len(pattern.parts) == 1 and pattern.last in params:
        return _settle(pattern.last, arg, found)
    return _same_value(pattern, arg)


def _match_type(
    pattern: CType, arg: CType, params: set[str], found: Bindings, known: Known
) -> bool:
    if not _match_dims(pattern, arg, params, found):
        return False
    if pattern.name in params:
        return _match_parameter(pattern, arg, found)
    name, args = known.get(arg.name, (arg.name, arg.args)) if pattern.args else (arg.name, [])
    shape = (pattern.name, pattern.pointer, pattern.const, len(pattern.args))
    if shape != (name, arg.pointer, arg.const, len(args)):
        return False
    pairs = zip(pattern.args, args, strict=True)
    return all(match(p, a, params, found, known) for p, a in pairs)


def _match_parameter(pattern: CType, arg: CType, found: Bindings) -> bool:
    """``T*`` against ``int**``: ``T`` is what is left of the argument, ``int*``."""
    if arg.pointer < pattern.pointer:
        return False
    # ``const`` is the constness of what is pointed at, so ``const T`` takes no pointer.
    if pattern.const and not (arg.const and arg.pointer == pattern.pointer):
        return False
    rest = dataclasses.replace(
        arg,
        pointer=arg.pointer - pattern.pointer,
        const=arg.const and not pattern.const,
        reference=arg.reference and not pattern.reference,
        dims=arg.dims[len(pattern.dims) :],
    )
    return _settle(pattern.name, rest, found)


def _match_dims(pattern: CType, arg: CType, params: set[str], found: Bindings) -> bool:
    """``T a[N]`` against ``double[3]``: each size the pattern has, matched to the argument's."""
    if not pattern.dims:
        return True
    if len(arg.dims) < len(pattern.dims):
        return False
    for want, have in zip(pattern.dims, arg.dims, strict=False):
        if isinstance(want, Name) and len(want.parts) == 1 and want.last in params:
            if not isinstance(have, int) or not _settle(want.last, have, found):
                return False
        elif want != have:
            return False
    return True


def _settle(name: str, value: Any, found: Bindings) -> bool:
    """``name`` takes ``value``, or already has one the same."""
    if name not in found:
        found[name] = value
        return True
    return _same_value(found[name], value)


def _same_value(one: Any, other: Any) -> bool:
    if isinstance(one, CType) or isinstance(other, CType):
        kinds = isinstance(one, CType) and isinstance(other, CType)
        return kinds and type_spelling(one, {}, None) == type_spelling(other, {}, None)
    return bool(_number(one) == _number(other))


def _number(value: Any) -> Any:
    """A value argument's number: a literal's, or one already worked out."""
    return value.value if isinstance(value, Literal) else value


def _specificity(patterns: list[Any]) -> int:
    """How much of its arguments a partial specialisation fixes: its pointers, ``const``s and
    named types, counted."""
    total = 0
    for pattern in patterns:
        if isinstance(pattern, CType):
            total += pattern.pointer + pattern.const + 1 + _specificity(pattern.args)
        total += isinstance(pattern, Literal)
    return total


def _partial(node: Node) -> bool:
    """A partial specialisation is a template of its own, instantiated only through its primary."""
    return isinstance(node, ClassDecl) and bool(node.template) and node.specialises is not None


def deduce_call(
    func: Function, explicit: list[Any], given: list[CType | None], known: Known
) -> Bindings | None:
    """The arguments of the function template ``func`` for a call that writes ``explicit`` in
    its ``<...>`` and passes arguments of the types ``given``: those written, those deduced
    from the arguments as C++ deduces them, then the defaults - or ``None`` if one is left
    unknown, or the arguments contradict each other."""
    names = list(func.template or [])
    params = set(names[len(explicit) :])
    found: Bindings = dict(zip(names, explicit, strict=False))
    for param, kind in zip(func.params, given, strict=False):
        pattern = _parameter_pattern(param.ctype)
        if kind is None or not _mentions(pattern, params):
            continue
        if not match(pattern, _argument_type(kind, param.ctype), params, found, known):
            return None
    return _defaulted(names, func.defaults, found)


def _defaulted(names: list[str], defaults: list[Any], found: Bindings) -> Bindings | None:
    """``found`` with the defaults of the parameters it lacks - ``None`` if one has none."""
    for index, name in enumerate(names):
        if name not in found:
            default = defaults[index] if index < len(defaults) else None
            if default is None:
                return None
            found[name] = copied(default, found)
    return found


def _parameter_pattern(ctype: CType) -> CType:
    """What an argument is matched against: the parameter without its reference and ``const``
    - ``const T &`` takes an ``int`` as ``T`` - except a reference to an array, which keeps
    the array's size, ``T (&)[N]``."""
    return dataclasses.replace(ctype, reference=False, const=ctype.const and bool(ctype.pointer))


def _argument_type(kind: CType, param: CType) -> CType:
    """An argument's type as deduction sees it: an array is a pointer to its first element,
    unless the parameter is a reference to an array; ``const`` of a value goes."""
    if kind.dims and not (param.reference and param.dims):
        kind = dataclasses.replace(kind, dims=kind.dims[1:], pointer=kind.pointer + 1)
    if not kind.pointer:
        kind = dataclasses.replace(kind, const=False)
    return dataclasses.replace(kind, reference=False)


def _mentions(ctype: Any, params: set[str]) -> bool:
    """Does a parameter's type name one of the template parameters still to be deduced?"""
    if isinstance(ctype, CType):
        if ctype.name in params or any(_mentions(arg, params) for arg in ctype.args):
            return True
        return any(_mentions(dim, params) for dim in ctype.dims)
    return isinstance(ctype, Name) and ctype.parts[0] in params
