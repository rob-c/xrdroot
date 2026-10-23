"""C++ vector elements are values; a copied pointer still names its original object."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class _Kind:
    element: Any
    numeric: bool
    bases: tuple[type, ...]


#: Origin is recorded outside concrete classes so their ordinary metadata stays unchanged.
_KINDS: dict[type, _Kind] = {}


def register(kind: type, element: Any, numeric: bool) -> type:
    """Only the native vector factories register a concrete declared value kind."""
    _KINDS[kind] = _Kind(element, numeric, kind.__bases__)
    return kind


def native(kind: Any) -> bool:
    """Pointer, user-class, subclass and object-dtype elements keep ordinary conversion."""
    if type(kind) is not type:
        return False
    found = _KINDS.get(kind)
    if found is None or kind.__bases__ != found.bases:
        return False
    state = vars(kind)
    if found.numeric:
        return state.get("dtype") is found.element
    return state.get("element") is found.element and native(found.element)


def converted(kind: Any, value: Any, ordinary: Callable[[Any, Any], Any]) -> Any:
    """Each by-value occurrence copies independently, without a shared graph memo."""
    if type(value) is not kind or not native(kind):
        return ordinary(kind, value)
    found = _KINDS[kind]
    # The ndarray path avoids Python scalar conversion of signaling NaNs and payloads.
    return kind(value.data()) if found.numeric else kind(value)


def assigned(kind: Any, target: Any, source: Any, ordinary: Callable[[Any, Any], Any]) -> bool:
    """Assign into an existing element object so references to it remain live."""
    if type(target) is not kind or type(source) is not kind or not native(kind):
        return False
    if target is source:
        return True
    found = _KINDS[kind]
    if found.numeric:
        target.assign(source.data())
    else:
        _assign_nested(found.element, target.data(), source.data(), ordinary)
    return True


def _assign_nested(
    element: type, target: list[Any], source: list[Any], ordinary: Callable[[Any, Any], Any],
) -> None:
    common = min(len(target), len(source))
    for index in range(common):
        if not assigned(element, target[index], source[index], ordinary):
            target[index] = converted(element, source[index], ordinary)
    for value in source[common:]:
        target.append(converted(element, value, ordinary))
    del target[len(source):]


def numeric_input(kind: type, source: Any, readers: dict[str, Any], array_type: type) -> Any:
    """An unchanged same-kind numeric vector supplies its logical array without scalar promotion."""
    if type(source) is not kind or not native(kind):
        return source
    found = _KINDS[kind]
    if not found.numeric or not _original_readers(kind, readers):
        return source
    storage = object.__getattribute__(source, "_data")
    size = object.__getattribute__(source, "_size")
    if not _numeric_storage(storage, size, found.element, array_type):
        return source
    return cast(Any, source).data()


def _original_readers(kind: type, readers: dict[str, Any]) -> bool:
    states = [vars(base) for base in kind.__mro__]
    return all(next((state[name] for state in states if name in state), None) is value
               for name, value in readers.items())


def _numeric_storage(storage: Any, size: Any, dtype: Any, array_type: type) -> bool:
    if type(storage) is not array_type:
        return False
    storage = cast(Any, storage)
    if storage.ndim != 1 or storage.dtype != dtype:
        return False
    return bool(type(size) is int and 0 <= size <= storage.size)
