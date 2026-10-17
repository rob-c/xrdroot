"""Native collection rows can copy their storage without repeating template dispatch."""
from __future__ import annotations

import inspect
import types
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ...tree import Jagged
from .. import stl
from . import rvec

#: Native implementation functions live within this installed package.
_PACKAGE = str(Path(__file__).parents[2]) + '/'
#: The normal sequence iterator calls Jagged's checked integer indexing.
_ITERATOR = inspect.getattr_static(Sequence, '__iter__')
#: Generated numeric vectors contain metadata and inherit their behavior.
_VECTOR_FIELDS = {'__slots__', 'dtype', 'value_type', '__cpp_name__', '__module__', '__doc__'}


def _function(value: Any, module: str, name: str) -> bool:
    if type(value) is not types.FunctionType:
        return False
    return (type(value.__module__) is str and type(value.__qualname__) is str
            and (value.__module__, value.__qualname__) == (module, name)
            and value.__code__.co_filename.startswith(_PACKAGE))


def _method(kind: type[Any], name: str, module: str, owner: str) -> bool:
    return _function(inspect.getattr_static(kind, name, None), module, owner + '.' + name)


def _access(kind: type[Any]) -> bool:
    expected = (('__getattribute__', object.__getattribute__),
                ('__setattr__', object.__setattr__), ('__getattr__', None), ('__del__', None))
    return all(inspect.getattr_static(kind, name, None) is value for name, value in expected)


def _slot(kind: type[Any], name: str) -> bool:
    value = inspect.getattr_static(kind, name, None)
    return (type(value) is types.MemberDescriptorType
            and value.__objclass__ is kind and value.__name__ == name)


def _jagged(column: Any) -> bool:
    if type(column) is not Jagged or not _access(Jagged):
        return False
    methods = all(_method(Jagged, name, 'xrdroot.tree', 'Jagged')
                  for name in ('__getitem__', '__len__'))
    return (methods and _slot(Jagged, 'content') and _slot(Jagged, 'offsets')
            and inspect.getattr_static(Jagged, '__iter__') is _ITERATOR)


def _arrays(column: Any) -> bool:
    content, offsets = column.content, column.offsets
    if type(content) is not np.ndarray or type(offsets) is not np.ndarray:
        return False
    return (content.ndim == offsets.ndim == 1 and content.dtype.kind in 'biuf'
            and offsets.dtype.kind == 'i' and offsets.dtype.itemsize == 8 and len(offsets) > 1)


def _mapping(value: Any, key_type: type[Any]) -> bool:
    return type(value) is dict and all(type(k) is key_type and type(v) is str
                                      for k, v in value.items())


def _keys(cache: Any) -> bool:
    if type(cache) is not dict:
        return False
    return all(type(key) is tuple and all(type(part) is str for part in key) for key in cache)


def _namespace() -> bool:
    functions = ((vars(stl).get('cpp_name'), 'xrdroot.pyroot.stl', 'cpp_name'),
                 (vars(stl).get('element_type'), 'xrdroot.pyroot.stl', 'element_type'),
                 (vars(stl).get('_numeric_dtype'), 'xrdroot.pyroot.stl', '_numeric_dtype'))
    return (all(_function(value, module, name) for value, module, name in functions)
            and _mapping(vars(stl).get('CPP_NAMES'), str)
            and _mapping(vars(stl).get('_DTYPE_NAMES'), int))


def _numpy() -> bool:
    functions = ((vars(np).get('asarray'), 'asarray'), (vars(np).get('zeros'), 'zeros'))
    return all(type(value) is types.BuiltinFunctionType and value.__name__ == name
               and type(value.__module__) is str and value.__module__ == 'numpy'
               for value, name in functions)


def _template() -> Any:
    template = vars(rvec).get('RVec')
    if type(template) is not stl._Template or not _dictionary(stl._Template):
        return None
    if not _access(stl._Template):
        return None
    state = vars(template)
    if not _template_state(state) or not _template_methods(state):
        return None
    return template


def _dictionary(kind: Any) -> bool:
    if type(kind) is not type:
        return False
    descriptor = inspect.getattr_static(kind, '__dict__', None)
    return (type(descriptor) is types.GetSetDescriptorType
            and descriptor.__objclass__ is kind)


def _template_state(state: dict[str, Any]) -> bool:
    if type(state.get('_arity')) is not int or state['_arity'] != 1:
        return False
    return _keys(state.get('_made')) and _namespace() and _numpy()


def _template_methods(state: dict[str, Any]) -> bool:
    methods = all(_method(stl._Template, name, 'xrdroot.pyroot.stl', '_Template')
                  for name in ('__getitem__', '_general'))
    return (methods and _template_base()
            and _function(state.get('_build'), 'xrdroot.pyroot.rdf.rvec', '_rvec_class'))


def _template_base() -> bool:
    return (type(rvec._RVec) is type
            and inspect.getattr_static(rvec._RVec, '__init_subclass__')
            is inspect.getattr_static(object, '__init_subclass__'))


def _vector(kind: Any) -> bool:
    if type(kind) is not type:
        return False
    bases = (rvec._RVec, np.lib.mixins.NDArrayOperatorsMixin, stl._Vector, stl._Container, object)
    if kind.__mro__[1:] != bases or set(kind.__dict__) != _VECTOR_FIELDS:
        return False
    methods = all(_method(kind, name, 'xrdroot.pyroot.stl', '_Vector')
                  for name in ('__init__', '_construct', 'assign'))
    return (methods and _access(kind) and _slot(stl._Vector, '_data')
            and _slot(stl._Vector, '_size')
            and inspect.getattr_static(kind, '__new__') is object.__new__)


def _storage(kind: Any, dtype: Any) -> Any:
    if not _vector(kind):
        return None
    held = inspect.getattr_static(kind, 'dtype')
    if not stl._numeric_dtype(held):
        return None
    # Equal precision and kind permit byte swapping without numerical conversion warnings.
    return held if (held.kind, held.itemsize) == (dtype.kind, dtype.itemsize) else None


def _prepared(column: Any, maker: Any) -> Any:
    if not _jagged(column) or not _function(maker, 'xrdroot.pyroot.rdf.rvec', '_made'):
        return None
    if not _arrays(column):
        return None
    template = _template()
    if template is None or not stl._numeric_dtype(column.content.dtype):
        return None
    kind = template[column.content.dtype]
    dtype = _storage(kind, column.content.dtype)
    return None if dtype is None else (kind, dtype)


def rows(column: Any, maker: Any) -> list[Any] | None:
    """Refuse before copying when ordinary construction could enter a user callback."""
    prepared = _prepared(column, maker)
    if prepared is None:
        return None
    kind, dtype = prepared
    made = []
    offsets, content = column.offsets, column.content
    for start, stop in pairwise(offsets):
        vector = object.__new__(kind)
        vector._data = content[start:stop].astype(dtype, copy=True).reshape(-1)
        vector._size = len(vector._data)
        made.append(vector)
    return made
