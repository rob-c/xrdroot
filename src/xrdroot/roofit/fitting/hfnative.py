"""Const optimization requires native methods and data that cannot run model callbacks."""
from __future__ import annotations

import types
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np

#: Native Python methods live below this package, including the fitting helpers.
_PACKAGE = str(Path(__file__).parents[2]) + "/"
#: Class dictionaries are checked by identity; user-defined equality is never invoked.
_CLASSES: dict[type[Any], dict[str, Any]] = {}
#: Repeated nodes share class checks only within one synchronous planning pass.
_CHECKING: ContextVar[dict[type[Any], set[str]] | None] = ContextVar(
    "native_model_validation", default=None)
#: Scalar storage whose arithmetic is the engine's double-precision arithmetic.
NUMBERS = (float, int, np.float64)
#: Builtin callable descriptors do not enter a Python model callback.
_BUILTIN_METHODS = (types.BuiltinFunctionType, types.MethodDescriptorType,
                    types.WrapperDescriptorType, types.ClassMethodDescriptorType)
#: Class metadata may carry ordinary values without evaluating a descriptor.
_METADATA = (str, int, float, bool, tuple, list, dict, set, frozenset, type(None),
             types.GetSetDescriptorType, types.MemberDescriptorType)


def _function(value: Any) -> bool:
    if isinstance(value, (classmethod, staticmethod)):
        return _function(value.__func__)
    if type(value) is types.FunctionType:
        return value.__code__.co_filename.startswith(_PACKAGE)
    return type(value) in _BUILTIN_METHODS


def _attribute(value: Any) -> bool:
    if isinstance(value, property):
        return all(_function(fn) for fn in (value.fget, value.fset, value.fdel) if fn is not None)
    if callable(value) or isinstance(value, (classmethod, staticmethod)):
        return _function(value)
    # A foreign descriptor can run Python merely when a field is read.
    return type(value) in _METADATA


def _same_class(kind: type[Any], expected: dict[str, Any]) -> bool:
    current = kind.__dict__
    return current.keys() == expected.keys() and all(current[key] is value
                                                     for key, value in expected.items())


def _class(kind: type[Any]) -> bool:
    module = kind.__module__
    if type(module) is not str or not (module == "builtins" or module.startswith("xrdroot.")):
        return False
    expected = _CLASSES.get(kind)
    if expected is not None and _same_class(kind, expected):
        return True
    if not all(_attribute(value) for value in kind.__dict__.values()):
        return False
    _CLASSES[kind] = dict(kind.__dict__)
    return True


def instance(value: Any, kinds: tuple[type[Any], ...]) -> bool:
    """A known concrete class with its own native methods, including inherited methods."""
    kind = type(value)
    if kind not in kinds:
        return False
    methods = _methods(kind)
    return methods is not None and methods.isdisjoint(getattr(value, "__dict__", {}))


def _methods(kind: type[Any]) -> set[str] | None:
    cache = _CHECKING.get()
    if cache is not None and kind in cache:
        return cache[kind]
    methods = _native_methods(kind)
    if cache is not None and methods is not None:
        cache[kind] = methods
    return methods


def _native_methods(kind: type[Any]) -> set[str] | None:
    if not all(_class(base) for base in kind.__mro__):
        return None
    return {name for base in kind.__mro__ for name, member in base.__dict__.items()
            if callable(member) or isinstance(member, (classmethod, staticmethod, property))}


@contextmanager
def checking() -> Iterator[None]:
    token = _CHECKING.set({})
    try:
        yield
    finally:
        _CHECKING.reset(token)


def number(value: Any, *, finite: bool = True) -> bool:
    if type(value) not in NUMBERS:
        return False
    try:
        return not finite or bool(np.isfinite(value))
    except (TypeError, OverflowError):
        return False


def array(value: Any, *, floating: bool = False) -> bool:
    if type(value) is not np.ndarray or value.ndim != 1:
        return False
    return value.dtype == np.float64 if floating else value.dtype.kind in "biufUS"
