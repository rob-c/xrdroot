"""``ROOT::Detail::TRangeStaticCast``/``TRangeDynCast``, and the type a range's element has.

``for (auto bcl : *cl->GetListOfBases())`` hands its body a ``TObject *`` whatever each
object really is, because that is the type the collection's iterators name. ``Class()`` is
a *static* member of the class in C++, so the compiler resolves ``bcl->Class()`` to
``TObject::Class()`` and ROOT prints ``TObject`` for a ``TBaseClass`` a list holds. Python
finds ``Class`` on the object and so prints the class each object really is, which is right
for a call the macro wrote on a variable it declared itself and wrong for one a range gave
it. An element a collection hands out is therefore wrapped: the wrapper is the object in
every respect but ``Class``, which answers the range's static type.

``TRangeStaticCast<T>`` and ``TRangeDynCast<T>`` are what a macro writes to say that type
itself: the first gives every object as a ``T *`` as ``static_cast`` does - no check, the
object unchanged - and the second only the objects that really are a ``T``, ``nullptr``
for the rest, as ``dynamic_cast`` does.
"""

from __future__ import annotations

from typing import Any

__all__ = ["TypedElement", "TRangeStaticCast", "TRangeDynCast", "Detail"]


class TypedElement:
    """One object of a collection as C++'s range-for gives it: statically of the range's type.

    Everything asked of it is the object's own, save ``Class``, which is the *static*
    member of the type the range declared - so ``for (auto bcl : *list) bcl->Class()``
    answers what ROOT answers. ``isinstance`` sees the object's own class, so what the
    macro hands on to the library is still the object C++ would hand it.
    """

    __slots__ = ("_object", "_static")

    def __init__(self, obj: Any, static: Any) -> None:
        object.__setattr__(self, "_object", obj)
        object.__setattr__(self, "_static", static)

    @property  # type: ignore[misc]  # an object's class is not settable, as ROOT's is not
    def __class__(self) -> Any:
        return type(self._object)

    def Class(self) -> Any:
        """``bcl->Class()``: the declared type's own, which C++ resolves the call to."""
        return self._static.Class()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._object, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._object, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._object, name)

    def __eq__(self, other: object) -> bool:
        return bool(self._object == _unwrapped(other))

    def __ne__(self, other: object) -> bool:
        return bool(self._object != _unwrapped(other))

    def __hash__(self) -> int:
        return hash(self._object)

    def __str__(self) -> str:
        return str(self._object)

    def __repr__(self) -> str:
        return repr(self._object)

    def __bool__(self) -> bool:
        return bool(self._object)


def _unwrapped(other: object) -> Any:
    """What another wrapper of these holds, so two wrappers of one object compare as it does."""
    return other._object if isinstance(other, TypedElement) else other


def elements_of(collection: Any, static: Any) -> list[TypedElement]:
    """A collection's objects, each as the range gives it, statically of ``static``."""
    return [TypedElement(item, static) for item in collection]


def _class_named(name: str) -> Any:
    """The class ROOT has by that name, as the namespace offers it - else the refusal."""
    import xrdroot.pyroot as namespace  # asked for a name, it gathers the module holding it

    found = getattr(namespace, name, None)
    if not isinstance(found, type):
        raise TypeError(
            f"ROOT::Detail::TRangeStaticCast<{name}> names no class this namespace has; "
            f"name a ROOT class, such as TBaseClass, that it declares"
        )
    return found


class _RangeCast:
    """A typed range over a collection: ``TRangeStaticCast<T>``, or ``TRangeDynCast<T>``.

    A macro writes ``TRangeStaticCast<TBaseClass>(*list)``, which is this factory indexed
    by the class's name - the argument a template is named by here - and then called with
    the collection.
    """

    __slots__ = ("_checked",)

    def __init__(self, checked: bool) -> None:
        self._checked = checked

    def __getitem__(self, name: str) -> Any:
        return lambda collection: self._range(str(name), collection)

    def _range(self, name: str, collection: Any) -> list[Any]:
        static = _class_named(name)
        if not self._checked:
            return elements_of(collection, static)
        return [item if isinstance(item, static) else None for item in collection]


#: ``ROOT::Detail::TRangeStaticCast<T>(c)``: ``c``'s objects as ``T *``, unchecked.
TRangeStaticCast = _RangeCast(checked=False)

#: ``ROOT::Detail::TRangeDynCast<T>(c)``: ``c``'s objects that are a ``T``, ``None`` for the
#: rest, as ``dynamic_cast`` gives ``nullptr``.
TRangeDynCast = _RangeCast(checked=True)


class Detail:
    """``ROOT::Detail::``: the namespace the range casts are ROOT's under."""

    TRangeStaticCast = TRangeStaticCast
    TRangeDynCast = TRangeDynCast
