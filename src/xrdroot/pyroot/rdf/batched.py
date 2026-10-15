"""Pure C++ array construction and reductions can operate on columns directly.

Only expressions whose operations already have batch semantics qualify. Other
function bodies keep the translated, entry-by-entry implementation.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ...cint import nodes as cxx
from ...cint.ctype import CType
from ...cint.errors import CintError
from ...cint.parse_decl import parse
from ...cint.tokens import tokenize
from ...rdf import language
from ...rdf.expression import Evaluator, Scope
from ...rdf.values import Value, to_column
from ...tree import Jagged
from .entrywise import ENTRY
from .randomexpr import _RandomDraw, random_draw


class _NotBatched(Exception):
    """The function needs the complete translator's entry-by-entry semantics."""


def _converted(node: Any, names: set[str]) -> language.Node:
    make = _CONVERTERS.get(type(node))
    if make is None:
        raise _NotBatched
    return make(node, names)


def _column(node: cxx.Name, names: set[str]) -> language.Column:
    if node.text not in names:
        raise _NotBatched
    return language.Column(node.text)


def _literal(node: cxx.Literal, names: set[str]) -> language.Literal:
    if node.kind not in ("int", "float"):
        raise _NotBatched
    # Numeric literal types are normalized to these built-in types by the C++ parser.
    dtype = language.TYPES[node.ctype]
    return language.Literal(np.asarray(node.value, dtype=dtype)[()])


def _binary(node: cxx.Binary, names: set[str]) -> language.Binary:
    if node.op not in ("+", "-", "*"):
        raise _NotBatched
    return language.Binary(node.op, _converted(node.left, names), _converted(node.right, names))


def _unary(node: cxx.Unary, names: set[str]) -> language.Unary:
    if node.op not in ("+", "-") or node.postfix:
        raise _NotBatched
    return language.Unary(node.op, _converted(node.operand, names))


def _call(node: cxx.Call, names: set[str]) -> language.Call:
    if not isinstance(node.func, cxx.Name) or node.func.text not in ("sqrt", "std::sqrt"):
        raise _NotBatched
    if len(node.args) != 1:
        raise _NotBatched
    return language.Call("sqrt", (_converted(node.args[0], names),))


#: A deliberately small subset of pure expressions with existing batch semantics.
_CONVERTERS: dict[type, Callable[..., language.Node]] = {
    cxx.Name: _column, cxx.Literal: _literal, cxx.Binary: _binary,
    cxx.Unary: _unary, cxx.Call: _call,
}


def _declaration(node: Any) -> cxx.VarDecl:
    if not isinstance(node, cxx.DeclStmt) or len(node.decls) != 1:
        raise _NotBatched
    return node.decls[0]


def _named(node: Any, name: str) -> bool:
    return isinstance(node, cxx.Name) and node.text == name


def _scope(columns: dict[str, Any], count: int) -> Scope:
    return Scope(lambda name: Value(np.asarray(columns[name])), count, np.arange(count))


class _Batched:
    """A recognized body and its declared result, including when no entries are requested."""

    def __init__(self, evaluate: Callable[..., Any], columns: list[str],
                 result_type: str | None = None) -> None:
        self.evaluate = evaluate
        self.columns = columns
        self._rdf_return_type = result_type

    @property
    def batched(self) -> _Batched:
        return self

    def __call__(self, *columns: Any) -> Any:
        return self.evaluate(*columns)


def _array_size(ctype: CType) -> int:
    if ctype.name not in ("std::array", "array") or len(ctype.args) != 2:
        raise _NotBatched
    kind, size = ctype.args
    if not isinstance(kind, CType) or kind.name != "double":
        raise _NotBatched
    if not isinstance(size, cxx.Literal) or size.kind != "int" or size.value < 0:
        raise _NotBatched
    return int(size.value)


def _body(function: cxx.Function, count: int) -> list[Any]:
    assert function.body is not None
    body = function.body.body
    if len(body) != count or not isinstance(body[-1], cxx.Return):
        raise _NotBatched
    return body


def _packed_decl(function: cxx.Function, body: list[Any]) -> tuple[cxx.VarDecl, int]:
    decl = _declaration(body[0])
    width = _array_size(decl.ctype)
    if decl.static or decl.style != "{}" or len(decl.args) > width:
        raise _NotBatched
    if not _named(body[-1].value, decl.name):
        raise _NotBatched
    return decl, width


def _packed(function: cxx.Function, used: list[str]) -> _Batched:
    body = _body(function, 2)
    decl, width = _packed_decl(function, body)
    parts = [_converted(arg, set(used)) for arg in decl.args]
    columns = list(dict.fromkeys(name for part in parts for name in language.columns_of(part)))
    parameters = {p.name: p.ctype.name for p in function.params}
    if any(parameters[name] != "double" for name in columns):
        raise _NotBatched
    columns = columns or [ENTRY]

    def batch(*given: Any) -> Any:
        count = len(given[0])
        scope = _scope(dict(zip(columns, given, strict=True)), count)
        evaluator = Evaluator(scope, "array initializer")
        out = np.zeros((count, width), dtype=np.float64)
        for at, part in enumerate(parts):
            out[:, at] = evaluator.top(part).data
        return Jagged(out.reshape(-1), np.arange(count + 1) * width)

    return _Batched(batch, columns, f"array<double,{width}>")


def _double_collection(ctype: CType) -> bool:
    if ctype.name.split("::")[-1] not in ("array", "vector", "RVec") or not ctype.args:
        return False
    return isinstance(ctype.args[0], CType) and ctype.args[0].name == "double"


def _iterator(loop: cxx.RangeFor, accumulator: str, used: list[str]) -> int:
    if not isinstance(loop.iterable, cxx.Name) or loop.iterable.text not in used:
        raise _NotBatched
    if loop.decl.ctype.name not in ("auto", "double") or loop.decl.ctype.pointer:
        raise _NotBatched
    if loop.decl.name == accumulator:
        raise _NotBatched
    return used.index(loop.iterable.text)


def _update(loop: cxx.RangeFor, accumulator: str) -> Any:
    body = loop.body
    if isinstance(body, cxx.Block) and len(body.body) == 1:
        body = body.body[0]
    if not isinstance(body, cxx.ExprStmt) or not isinstance(body.expr, cxx.Assign):
        raise _NotBatched
    update = body.expr
    if update.op != "+=" or not _named(update.target, accumulator):
        raise _NotBatched
    return _converted(update.value, {loop.decl.name})


def _accumulator(node: Any) -> cxx.VarDecl:
    decl = _declaration(node)
    if decl.ctype.name != "double" or decl.ctype.const or decl.static:
        raise _NotBatched
    if not isinstance(decl.init, cxx.Literal) or decl.init.kind not in ("int", "float"):
        raise _NotBatched
    return decl


def _folded(function: cxx.Function, used: list[str]) -> _Batched:
    body = _body(function, 3)
    if not isinstance(body[1], cxx.RangeFor):
        raise _NotBatched
    decl = _accumulator(body[0])
    position = _iterator(body[1], decl.name, used)
    transform = _update(body[1], decl.name)
    if not _double_collection(function.params[position].ctype):
        raise _NotBatched
    finish = _converted(body[-1].value, {decl.name})
    assert isinstance(decl.init, cxx.Literal)
    initial, item = float(decl.init.value), body[1].decl.name

    def batch(*given: Any) -> Any:
        content, offsets = _rows(given[0])
        scope = _scope({item: content}, len(content))
        mapped = np.broadcast_to(Evaluator(scope, "range-for expression").top(transform).data,
                                 content.shape)
        result = _summed(mapped, offsets, initial)
        value = Evaluator(_scope({decl.name: result}, len(result)), "range-for return").top(finish)
        return to_column(value, len(result))

    return _Batched(batch, [used[position]])


def _rows(column: Any) -> tuple[Any, Any]:
    if isinstance(column, Jagged):
        return column.content, column.offsets
    matrix = np.asarray(column)
    width = matrix.shape[1]
    return matrix.reshape(-1), np.arange(len(matrix) + 1) * width


def _summed(mapped: Any, offsets: Any, initial: float) -> Any:
    """Each accumulator starts at its initializer and adds in the C++ loop's order."""
    lengths = np.diff(offsets)
    out = np.full(len(lengths), initial, dtype=np.float64)
    for at in range(lengths.max(initial=0)):
        rows = np.flatnonzero(lengths > at)
        out[rows] = out[rows] + mapped[offsets[rows] + at]
    return out


def compiled(source: str, used: list[str], original: Any) -> _Batched | _RandomDraw | None:
    try:
        function = parse(tokenize(source)).decls[0]
    except CintError:
        return None  # the complete translator may have needed preprocessing
    assert isinstance(function, cxx.Function) and function.body is not None
    if not used:
        drawn = random_draw(original, function.body.body)
        if drawn is not None:
            return drawn
    for make in (_packed, _folded):
        try:
            return make(function, used)
        except _NotBatched:
            pass
    return None
