"""Batch selection retains custom readers, vector constructors and live function lookup."""
from __future__ import annotations

import inspect
import sys
import types
from typing import Any

import numpy as np

from ...tree import Jagged
from .. import stl
from . import rvec, vectorrows
from .vectorrows import _dictionary, _function, _method


def _compact(column: Any) -> bool:
    offsets = column.offsets
    return (offsets[0] == 0 and offsets[-1] == len(column.content)
            and bool((offsets[1:] >= offsets[:-1]).all()))


def accepts(columns: tuple[Any, ...], maker: Any) -> bool:
    methods = ((rvec._RVec, '__getitem__', 'xrdroot.pyroot.rdf.rvec', '_RVec'),
               (rvec._RVec, '__array__', 'xrdroot.pyroot.stl', '_Vector'),
               (rvec._RVec, 'data', 'xrdroot.pyroot.stl', '_Vector'),
               (rvec._RVec, '__len__', 'xrdroot.pyroot.stl', '_Vector'),
               (Jagged, '__init__', 'xrdroot.tree', 'Jagged'))
    return (vars(stl).get('np') is np and vars(rvec).get('np') is np and _mask_kernel()
            and inspect.getattr_static(Jagged, '__new__') is object.__new__
            and all(_method(*method) for method in methods)
            and all(vectorrows._prepared(column, maker) is not None and _compact(column)
                    for column in columns))


def _numpy_function(name: str) -> bool:
    value = vars(np).get(name)
    kind = type(value)
    builtin = kind is types.BuiltinFunctionType
    dispatcher = kind is _DISPATCHER
    if not (builtin or dispatcher):
        return False
    return (type(value.__module__) is str and value.__module__ == 'numpy'
            and type(value.__name__) is str and value.__name__ == name)


def _mask_kernel() -> bool:
    from ...rdf import kernels
    functions = ('masked', 'rows_of', 'offsets_of')
    names = ('asarray', 'zeros', 'bincount', 'repeat', 'arange', 'diff', 'cumsum', 'array_equal')
    return (vars(kernels).get('np') is np
            and all(_function(vars(kernels).get(name), 'xrdroot.rdf.kernels', name)
                    for name in functions)
            and all(_numpy_function(name) for name in names))


def _dispatcher_type(value: Any) -> Any:
    kind = type(value)
    if type(kind) is not type:
        return None
    module = kind.__module__
    valid = type(module) is str and module == 'numpy'
    return kind if valid and kind.__name__ == '_ArrayFunctionDispatcher' else None


def callee(original: Any, name: str, rooted: bool) -> Any:
    if not rooted:
        found = original.__globals__.get(name)
    else:
        found = _root_callee(original.__globals__.get('ROOT'), name)
    if type(found) is not types.FunctionType or type(found.__module__) is not str:
        return None
    native = found.__module__ == '__cint__'
    return found if native and found.__code__.co_filename.startswith('<translation of ') else None


def _root_ready(root: Any) -> bool:
    from ...cint.runtime.root import RootProxy
    if type(root) is not RootProxy or not _dictionary(RootProxy):
        return False
    access = inspect.getattr_static(RootProxy, '__getattribute__') is object.__getattribute__
    descriptor = inspect.getattr_static(RootProxy, '__dict__')
    methods = ('__getattr__', 'namespace', '_used')
    return (access and type(descriptor) is types.GetSetDescriptorType
            and descriptor.__objclass__ is RootProxy
            and all(_method(RootProxy, name, 'xrdroot.cint.runtime.root', 'RootProxy')
                    for name in methods)
            and inspect.getattr_static(RootProxy, '_bound', None) is None
            and inspect.getattr_static(RootProxy, '_resolved', None) is None)


def _root_state(root: Any) -> Any:
    if not _root_ready(root):
        return None
    state = vars(root)
    if not all(type(key) is str for key in state):
        return None
    methods = {'__getattr__', 'namespace', '_used', '__getattribute__'}
    return state if methods.isdisjoint(state) else None


def _root_callee(root: Any, name: str) -> Any:
    state = _root_state(root)
    if state is None:
        return None
    namespace = _root_namespace(state)
    if namespace is None:
        return None
    resolved = state['_resolved']
    if not _resolved_keys(resolved):
        return None
    cached = resolved.get((id(namespace), name))
    return cached if cached is not None else vars(namespace).get(name)


def _root_namespace(state: dict[str, Any]) -> Any:
    from ...cint.runtime.root import DEFAULT
    if type(DEFAULT) is not str:
        return None
    bound, resolved = state.get('_bound'), state.get('_resolved')
    if type(bound) is not list or type(resolved) is not dict or type(sys.modules) is not dict:
        return None
    namespace = bound[-1] if bound else sys.modules.get(DEFAULT)
    return namespace if type(namespace) is types.ModuleType else None


def _resolved_keys(resolved: dict[Any, Any]) -> bool:
    return all(type(key) is tuple and len(key) == 2
               and type(key[0]) is int and type(key[1]) is str for key in resolved)


#: Recognize native NumPy dispatchers before inspecting callable metadata.
_DISPATCHER = _dispatcher_type(np.repeat)
