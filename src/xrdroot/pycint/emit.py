"""ROOT's own patterns, written the ROOT dialect's way: the emitter hooks of :data:`.dialect.ROOT`.

Each is a small function the translator calls with itself - the emitter -
where the dialect says a construct of ROOT's is not what plain C++ would
make of it: ``TString::Format`` and ``s.Form(...)`` build a ``TString``, a
counted loop filling histograms from ``gRandom`` draws every turn at once,
and a loop over ``ROOT::TSeq`` reading an ntuple into one histogram fills it
in a batch when it can.
"""

from __future__ import annotations

from typing import Any

from ..pycxx.base import Out, P
from ..pycxx.ctype import CType
from ..pycxx.nodes import Block, Call, Expr, For, Name, RangeFor, VarDecl
from .fill_loops import VectorDraw, vector_fill
from .tree_loops import NtupleFill, ntuple_fill

__all__ = ["formatted", "changing", "counted_loop", "range_loop", "SEQUENCES"]

#: ``ROOT::TSeq``'s names, whose ``for`` walks integers as ``range`` does.
SEQUENCES = frozenset({"TSeq", "TSeqI", "TSeqU", "TSeqL", "TSeqUL"})


def formatted(emitter: Any, func: Name, node: Call) -> Out | None:
    """``TString::Format(fmt, ...)``: the ``TString`` of the C formatting."""
    if func.parts != ["TString", "Format"]:
        return None
    return f"ROOT.TString(cformat({emitter.arguments(node)}))", P.POSTFIX


def changing(emitter: Any, owner: CType, target: str, name: str, args: list[str]) -> str | None:
    """``s.Form(fmt, ...)`` of a ``TString``: ``s`` made anew from the formatting."""
    if owner.name == "TString" and not owner.pointer and name == "Form":
        return f"{target} = ROOT.TString(cformat({', '.join(args)}))"
    return None


def counted_loop(emitter: Any, node: For, decl: VarDecl, turns: str) -> bool:
    """A counted loop filling histograms from a random draw, written all at once if it is one."""
    draw = vector_fill(node.body, decl.name, emitter.typeof)
    if draw is None:
        return False
    _all_at_once(emitter, node, draw, turns)
    return True


def range_loop(emitter: Any, node: RangeFor) -> bool:
    """A loop over ``ROOT::TSeq``: filling histograms from draws all at once, or an ntuple's
    column into a histogram in a batch - else not written here."""
    turns = _sequence(node.iterable, emitter.value)
    draw = None if turns is None else vector_fill(node.body, node.decl.name, emitter.typeof)
    if draw is not None and node.decl.binding is None:
        assert turns is not None
        _all_at_once(emitter, node, draw, turns)
        return True
    fill = None if turns is None else ntuple_fill(node.body, node.decl.name)
    if fill is not None and node.decl.binding is None:
        _ntuple_fill(emitter, node, fill)
        return True
    return False


def _all_at_once(emitter: Any, node: For | RangeFor, draw: VectorDraw, turns: str) -> None:
    """A loop filling histograms from a random draw, as the draws of every turn at once
    and one ``Fill`` of them per histogram (see :func:`.fill_loops.vector_fill`): the same
    bins and moments to the last bit, without a turn of Python for each."""
    count = emitter.fresh("count")
    emitter.out.line(f"{count} = len({turns})", node.where)
    stmts = node.body.body if isinstance(node.body, Block) else [node.body]
    emitter.vector_draw, emitter.vector_count = draw.call, count
    try:
        with emitter.scoped():
            arrays = _pair_drawn(emitter, draw, count) if draw.cells else []
            for stmt in stmts[1:] if draw.cells else stmts:
                emitter.statement(stmt)
            _pair_kept(emitter, draw.cells, arrays, count, node)
    finally:
        emitter.vector_draw = emitter.vector_count = None
        emitter.vector_names = {}


def _pair_drawn(emitter: Any, draw: VectorDraw, count: str) -> list[str]:
    """``Rannor(a, b)`` of every turn at once: arrays standing for ``a`` and ``b`` in
    the fills, rounded as a ``Float_t`` would have been."""
    arrays = [emitter.fresh(name) for name in draw.cells]
    emitter.vector_names = dict(zip(draw.cells, arrays, strict=True))
    where = draw.call.where
    emitter.out.line(f"{', '.join(arrays)} = {emitter.value(draw.call.func)}(n={count})", where)
    for name, array in emitter.vector_names.items():
        symbol = emitter.lookup(name)
        if symbol is not None and symbol.ctype is not None and symbol.ctype.name == "float":
            emitter.out.line(f"{array} = f32({array})", where)
    return arrays


def _pair_kept(
    emitter: Any, cells: tuple[str, ...], arrays: list[str], count: str, node: For | RangeFor
) -> None:
    """After the loop, ``a`` and ``b`` hold the last turn's draw - unless there was none."""
    if not cells:
        return
    emitter.out.line(f"if {count}:", node.where)
    with emitter.out.indented():
        for name, array in zip(cells, arrays, strict=True):
            symbol = emitter.lookup(name)
            assert symbol is not None  # the analysis found its type, so it is declared
            emitter.out.line(f"{emitter.use(symbol)[0]} = {array}[-1]", node.where)


def _ntuple_fill(emitter: Any, node: RangeFor, fill: NtupleFill) -> None:
    """A loop reading one ntuple column into one histogram: the runtime fills it in a batch
    when the objects it gets are those, and the loop runs as written when they are not."""
    sequence = emitter.fresh("entries")
    emitter.out.line(f"{sequence} = {emitter.value(node.iterable)}", node.where)
    arguments = (emitter.value(fill.tree), emitter.value(fill.histogram), sequence,
                 str(fill.column))  # fmt: skip
    emitter.out.line(f"if not fill_ntuple_histogram({', '.join(arguments)}):", node.where)
    with emitter.out.indented():
        emitter.range_loop(node, sequence)


def _sequence(node: Expr, value: Any) -> str | None:
    """``range(...)`` for a ``for`` over ``ROOT::TSeqI(n)`` - or ``(begin, end[, step])`` -
    which a fill loop's every turn at once is written over; ``None`` for any other range."""
    if not isinstance(node, Call) or not isinstance(node.func, Name):
        return None
    if node.func.last not in SEQUENCES or not 1 <= len(node.args) <= 3:
        return None
    return f"range({', '.join(value(arg) for arg in node.args)})"
