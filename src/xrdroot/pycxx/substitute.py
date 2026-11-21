"""Template arguments put in for template parameters, in a type or a whole declaration.

A template's text names its parameters - ``T``, ``N`` - and an
instantiation is that text with each replaced by an argument: ``std::vector<T>``
is ``std::vector<double>``, ``const T &`` is ``const double &``, ``T a[N]`` is
``double a[3]``, and ``N`` in an expression is ``3``. :func:`substituted`
does it to a type, :func:`copied` to a copy of any part of a syntax tree,
leaving the template itself as it was.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .ctype import CType
from .dialect import active
from .errors import Where
from .nodes import Call, Cast, Literal, Name, Node

__all__ = ["Bindings", "substituted", "copied", "as_literal", "bound", "aliased"]

#: What each template parameter stands for: a type, or the expression of a value.
Bindings = dict[str, Any]


def substituted(ctype: CType, bindings: Bindings) -> CType:
    """``ctype`` with the parameters ``bindings`` gives arguments for put in."""
    dims = [_dimension(dim, bindings) for dim in ctype.dims]
    bound = bindings.get(ctype.name) if not ctype.args else None
    bound = bound if isinstance(bound, CType) else _member_type(ctype.name, bindings)
    if bound is not None:
        return dataclasses.replace(
            bound,
            pointer=bound.pointer + ctype.pointer,
            reference=bound.reference or ctype.reference,
            const=bound.const or ctype.const,
            dims=[*dims, *bound.dims],
            volatile=bound.volatile or ctype.volatile,
        )
    args = copied(ctype.args, bindings)
    return dataclasses.replace(
        ctype, args=args, dims=dims, scope_args=copied(ctype.scope_args, bindings)
    )


#: The member types of the standard's containers, and which template argument each is.
MEMBER_TYPES = {"value_type": 0, "key_type": 0, "element_type": 0, "mapped_type": 1}


def _member_type(name: str, bindings: Bindings) -> CType | None:
    """``typename T::value_type`` with ``T`` a container of the standard's: its element, as
    ``std::map<K, V>::mapped_type`` is ``V``; ``T::size_type`` is ``size_t``. A class of the
    dialect's library answers through :attr:`~.dialect.Dialect.member_types`."""
    head, _, rest = name.partition("::")
    bound = bindings.get(head)
    if not isinstance(bound, CType) or bound.pointer:
        return None
    if not bound.name.startswith("std::"):
        return _library_member(bound, rest)
    if rest == "size_type":
        return CType("unsigned long")
    index = MEMBER_TYPES.get(rest)
    found = bound.args[index] if index is not None and index < len(bound.args) else None
    return found if isinstance(found, CType) else None


def _library_member(bound: CType, rest: str) -> CType | None:
    """The member alias ``rest`` of the library class ``bound`` is an instance of, if the
    dialect knows what that class's aliases stand for."""
    asked = active().member_types
    found = asked(bound.name, list(bound.args)) if asked is not None else None
    alias = (found or {}).get(rest)
    return alias if isinstance(alias, CType) else None


def _dimension(dim: Any, bindings: Bindings) -> Any:
    """An array's size: ``N`` given ``3`` is the number ``3``, as a declared size is."""
    found = copied(dim, bindings)
    return found.value if isinstance(found, Literal) and found.kind == "int" else found


def as_literal(value: Any, where: Where) -> Node:
    """A value template argument known as a number, as the literal a template's text reads."""
    if isinstance(value, Node):
        return value
    return Literal(where, "int", int(value), "int")


def copied(node: Any, bindings: Bindings) -> Any:
    """A copy of ``node`` - a syntax tree, a type, or a list of them - with ``bindings`` put in.

    A name that is a value parameter becomes its value; one whose first part
    is a type parameter, ``T::kSize``, is looked up in the type the parameter
    stands for.
    """
    if isinstance(node, CType):
        return substituted(node, bindings)
    if isinstance(node, (list, tuple)):
        return type(node)(copied(item, bindings) for item in node)
    if isinstance(node, Name):
        return _name(node, bindings)
    if isinstance(node, Call) and _converted(node, bindings):
        return _conversion(node, bindings)
    return _node(node, bindings) if isinstance(node, Node) else node


def _node(node: Node, bindings: Bindings) -> Node:
    """A copy of any other node, each of its parts copied."""
    changes = {
        item.name: copied(getattr(node, item.name), bindings)
        for item in dataclasses.fields(node)
        if item.name != "where"
    }
    return dataclasses.replace(node, **changes)


def _converted(node: Call, bindings: Bindings) -> bool:
    """Is ``node`` ``T(x)`` or ``T()`` of a type parameter - or a type a parameter's class
    names, ``T::value_type(x)`` - that stands for a number or pointer?"""
    func = node.func
    if not isinstance(func, Name) or len(node.args) > 1:
        return False
    bound = bindings.get(func.text)
    return isinstance(bound, CType) and (bound.scalar or bound.is_pointer)


def _conversion(node: Call, bindings: Bindings) -> Cast:
    """``T(x)`` of a number type is the conversion ``double(x)``; ``T()`` is zero."""
    assert isinstance(node.func, Name)
    bound = bindings[node.func.text]
    zero = Literal(node.where, "int", 0, "int")
    operand = copied(node.args[0], bindings) if node.args else zero
    return Cast(node.where, bound.value(), operand, "functional")


def _name(node: Name, bindings: Bindings) -> Node:
    found = _named(node, bindings)
    if isinstance(found, Name) and node.scope_targs:
        found.scope_targs = copied(node.scope_targs, bindings)
        found.targs = found.scope_targs if node.targs is node.scope_targs else found.targs
    return found


def _named(node: Name, bindings: Bindings) -> Node:
    targs = copied(node.targs, bindings) if node.targs else node.targs
    bound = bindings.get(node.parts[0])
    if len(node.parts) == 1 and bound is not None and not isinstance(bound, CType):
        return as_literal(bound, node.where)
    if isinstance(bound, CType) and not bound.pointer:
        return _typed_name(node, bound, targs)
    return Name(node.where, list(node.parts), targs, node.rooted)


def _typed_name(node: Name, bound: CType, targs: Any) -> Name:
    """``T`` or ``T::member`` with ``T`` a type: the type's name - and its arguments, for
    ``T(...)`` of a class template's instance not made yet."""
    parts = [*bound.name.split("::"), *node.parts[1:]]
    given = bound.args if len(node.parts) == 1 and bound.args else targs
    found = Name(node.where, parts, list(given) if given else None, node.rooted)
    if len(node.parts) == 2 and bound.args:
        found.scope_targs = list(bound.args)  # ``T::count`` of ``Box<int>``: its instance's
    return found


def bound(names: list[str], defaults: list[Any], args: list[Any]) -> Bindings | None:
    """Each of a template's parameters ``names`` given ``args`` in order, the rest their
    defaults - which may name the parameters before them - or ``None`` when one has none."""
    bindings: Bindings = {}
    for index, name in enumerate(names):
        if index < len(args):
            bindings[name] = args[index]
            continue
        default = defaults[index] if index < len(defaults) else None
        if default is None:
            return None
        bindings[name] = copied(default, bindings)
    return bindings


def aliased(template: tuple[list[str], list[Any], CType], args: list[Any]) -> CType:
    """The type an alias template names given ``args``: ``Vec<int>`` of ``template <class T>
    using Vec = std::vector<T>`` is ``std::vector<int>``. Arguments short of its parameters
    leave those it has no default for as they are written."""
    names, defaults, ctype = template
    bindings = bound(names, defaults, args)
    if bindings is None:
        bindings = dict(zip(names, args, strict=False))
    return substituted(ctype, bindings)
