"""In a task: the recorded graph called on the task's frame, and run.

:func:`generate_computation_graph` calls each node's operation on its
parent's frame, top down, making every action lazy - a ``Snapshot`` into a
file of the task's own - and gives back the booked results in the order of
the graph's actions. :func:`trigger_computation_graph` does that the first
time a run's graph comes to this process, books the same results again on
the same frame every other time (see :mod:`._graph_cache`), and runs the
frame's loop over the task's range.
"""

from __future__ import annotations

import copy
from functools import singledispatch
from typing import TYPE_CHECKING, Any, cast

from ._graph_cache import _ACTIONS_REGISTER, ExecutionIdentifier
from .Backends import Utils
from .Operation import Action, AsNumpy, InstantAction, Operation, Snapshot, VariationsFor
from .PythonMergeables import SnapshotResult

if TYPE_CHECKING:
    from .Node import Node

__all__ = ["generate_computation_graph", "trigger_computation_graph"]


@singledispatch
def append_node_to_results(operation: Operation, promise: Any, results: list[Any]) -> None:
    """Keep an action's booked result; a transformation books none."""


@append_node_to_results.register(Action)
@append_node_to_results.register(InstantAction)
def _(operation: Operation, promise: Any, results: list[Any]) -> None:
    results.append(promise)


@append_node_to_results.register
def _(operation: Snapshot, promise: Any, results: list[Any]) -> None:
    results.append(SnapshotResult(operation.args[0], [operation.args[1]], promise))


@singledispatch
def _create_lazy_op_if_needed(operation: Operation, range_id: int) -> Operation:
    """The operation as a task calls it: an action that runs at once made lazy."""
    return operation


@_create_lazy_op_if_needed.register
def _(operation: AsNumpy, range_id: int) -> Operation:
    operation.kwargs["lazy"] = True
    return operation


@_create_lazy_op_if_needed.register
def _(operation: Snapshot, range_id: int) -> Operation:
    """A copy writing into ``name_<range id>.root``, lazily, with the options given."""
    from ..rdf.snapshot import RSnapshotOptions

    made = copy.deepcopy(operation)
    made.args[1] = f"{made.args[1].partition('.root')[0]}_{range_id}.root"
    if len(made.args) == 2:
        made.args.append("")
    if len(made.args) == 4:
        made.args[3].fLazy = True
    else:
        made.args.append(RSnapshotOptions(lazy=True))
    return made


@singledispatch
def _call_rdf_operation(op: Operation, parent_rdf_node: Any, range_id: int) -> Any:
    """The operation called on its parent's frame, and the operation as it was called."""
    in_task_op = _create_lazy_op_if_needed(op, range_id)
    made = getattr(parent_rdf_node, op.name)(*in_task_op.args, **in_task_op.kwargs)
    return made, in_task_op


@_call_rdf_operation.register
def _(op: VariationsFor, parent_rdf_node: Any, range_id: int) -> Any:
    from ...errors import UnsupportedFeatureError

    raise UnsupportedFeatureError(
        "VariationsFor gives the results of the variations Vary booked, and xrdroot's "
        "RDataFrame has no Vary yet; book each variation as a result of its own instead"
    )


def generate_computation_graph(
    graph: dict[int, Node], starting_node: Any, range_id: int
) -> list[Any]:
    """Call every node's operation on its parent's frame - the head's is ``starting_node`` - and
    give back the booked results, an action's after its parents'."""
    nodes = iter(graph.items())
    head_id, _ = next(nodes)
    frames = {head_id: starting_node}
    promises: list[Any] = []
    for node_id, node in nodes:
        parent = frames[node.parent_id]  # type: ignore[index]
        made, in_task_op = _call_rdf_operation(cast(Operation, node.operation), parent, range_id)
        frames[node_id] = made
        append_node_to_results(in_task_op, made, promises)
    return promises


def trigger_computation_graph(
    graph: dict[int, Node], starting_node: Any, range_id: int, exec_id: ExecutionIdentifier
) -> list[Any]:
    """Book the graph's results on this task's frame - or book again the ones booked the
    last time this run's graph came here - and run the frame's loop."""
    if exec_id not in _ACTIONS_REGISTER or exec_id.rdf_uuid == "RNTuple":
        _ACTIONS_REGISTER[exec_id] = generate_computation_graph(graph, starting_node, range_id)
    else:
        _ACTIONS_REGISTER[exec_id] = [
            Utils.clone_action(action, range_id) for action in _ACTIONS_REGISTER[exec_id]
        ]
    Utils.trigger_run(starting_node)
    return _ACTIONS_REGISTER[exec_id]
