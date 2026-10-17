"""Prepare condition-vector arguments in batches while scalar callees retain entry order."""
from __future__ import annotations

import types
from typing import Any

import numpy as np

from ...cint import nodes as cxx
from ...cint.ctype import CType
from ...rdf.kernels import masked
from ...tree import Jagged
from .batched import _Batched
from .entrywise import entrywise


def _numeric_vector(ctype: CType) -> bool:
    if ctype.pointer or ctype.reference or ctype.const:
        return False
    name = ctype.name.split('::')[-1]
    if name in ('RVecF', 'RVecD', 'RVecI', 'RVecL', 'RVecB'):
        return not ctype.args
    if name != 'RVec' or len(ctype.args) != 1:
        return False
    item = ctype.args[0]
    return isinstance(item, CType) and item.arithmetic


def _call(function: cxx.Function) -> tuple[str, list[cxx.Expr]] | None:
    assert function.body is not None
    body = function.body.body
    if len(body) != 1 or not isinstance(body[0], cxx.Return):
        return None
    call = body[0].value
    if not isinstance(call, cxx.Call) or not isinstance(call.func, cxx.Name) or not call.args:
        return None
    return call.func.text, call.args


def _argument(arg: cxx.Expr, parameters: dict[str, CType],
              positions: dict[str, int]) -> tuple[int, int] | None:
    names = _index_names(arg)
    if names is None:
        return None
    if not all(name in parameters and name in positions for name in names):
        return None
    if not all(_numeric_vector(parameters[name]) for name in names):
        return None
    return positions[names[0]], positions[names[1]]



def _index_names(arg: cxx.Expr) -> tuple[str, str] | None:
    if not isinstance(arg, cxx.Index):
        return None
    if not isinstance(arg.obj, cxx.Name) or not isinstance(arg.index, cxx.Name):
        return None
    return arg.obj.text, arg.index.text


def _arguments(function: cxx.Function, args: list[cxx.Expr],
               used: list[str]) -> list[tuple[int, int]] | None:
    parameters = {param.name: param.ctype for param in function.params if param.name is not None}
    positions = {name: at for at, name in enumerate(used)}
    found = []
    for arg in args:
        part = _argument(arg, parameters, positions)
        if part is None:
            return None
        found.append(part)
    return found


def compiled(function: cxx.Function, used: list[str], original: Any) -> _Batched | None:
    call = _call(function)
    if call is None or type(original) is not types.FunctionType:
        return None
    name, args = call
    arguments = _arguments(function, args, used)
    if arguments is None:
        return None
    code = original.__code__
    if code.co_names not in ((name,), ('ROOT', name)):
        return None
    return _bound(original, name, code.co_names == ('ROOT', name), arguments, used)


def _bound(original: Any, name: str, rooted: bool,
           arguments: list[tuple[int, int]], used: list[str]) -> _Batched:
    from . import entrywise as wrappers
    from . import masknative, rvec

    code = original.__code__

    def invoke(*args: Any) -> Any:
        namespace = original.__globals__
        function = getattr(namespace['ROOT'], name) if rooted else namespace[name]
        return function(*args)

    native = entrywise(invoke)
    fallback = entrywise(original)

    def batch(*columns: Any) -> Any:
        if (original.__code__ is not code
                or vars(wrappers).get('_made') is not vars(rvec).get('_made')
                or masknative.callee(original, name, rooted) is None
                or not masknative.accepts(columns, vars(rvec).get('_made'))):
            return fallback(*columns)
        if not _aligned(columns, arguments):
            return fallback(*columns)
        selected = [Jagged(*masked(columns[t].content, columns[t].offsets, columns[c].content))
                    for t, c in arguments]
        return native(*selected)

    return _Batched(batch, used)


def _aligned(columns: tuple[Any, ...], arguments: list[tuple[int, int]]) -> bool:
    return all(np.array_equal(columns[t].offsets, columns[c].offsets) for t, c in arguments)
