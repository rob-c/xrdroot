"""What a user holds of a distributed graph: a proxy of a node, standing for a frame or result.

A :class:`NodeProxy` is a frame - ``RNode`` - whose methods record a new node
and give back a proxy of it; a :class:`ResultPtrProxy` is a booked result -
``RResultPtr`` - whose value, when first asked for, runs the graph. A proxy
going out of scope tells its node nobody holds it any more, which lets the
node be pruned from the graph sent to the tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import singledispatch
from typing import Any

from . import Operation
from .Node import Node

__all__ = [
    "Proxy",
    "NodeProxy",
    "ResultPtrProxy",
    "ResultMapProxy",
    "execute_graph",
    "get_proxy_for",
]


def execute_graph(node: Node) -> None:
    """Run the graph ``node`` is in on its backend, unless its value is already there."""
    if node.value is None:
        node.get_head().execute_graph()


def _update_internal_df_with_transformation(node: Node, operation: Operation.Operation) -> None:
    """Call a transformation on the parent's frame too, so the node knows its columns."""
    parent = node.parent if node.parent is not None else node
    node.rdf_node = getattr(parent.rdf_node, operation.name)(*operation.args, **operation.kwargs)


def _create_new_node(parent: Node, operation: Operation.Operation) -> Node:
    """A node for ``operation``, a child of ``parent``, added to the graph."""
    headnode = parent.get_head()
    headnode.node_counter += 1
    made = Node(parent.get_head, headnode.node_counter, operation, parent)
    parent.nchildren += 1
    headnode.graph_nodes.appendleft(made)
    return made


class Proxy(ABC):
    """A proxy of a node; when it goes, the node is no longer held by the user."""

    def __init__(self, node: Node) -> None:
        self.proxied_node = node

    @abstractmethod
    def __getattr__(self, attr: str) -> Any:
        """How the proxy answers for what it stands for."""

    def __del__(self) -> None:
        node = self.__dict__.get("proxied_node")
        if node is not None:
            node.has_user_references = False


class ResultMapProxy(Proxy):
    """``RResultMap``: the varied results of an action - which xrdroot's frame cannot make."""

    def __getattr__(self, attr: str) -> Any:
        raise AttributeError(f"'ResultMapProxy' object has no attribute '{attr}'")


class ResultPtrProxy(Proxy):
    """``RResultPtr``: a result, computed - with every other result of the graph - when first
    asked for; any method of the value is the value's, called then."""

    def __getattr__(self, attr: str) -> Any:
        self._cur_attr = attr
        return self._call_action_result

    def GetValue(self) -> Any:
        """The value, running the graph first if it has not run."""
        execute_graph(self.proxied_node)
        return self.proxied_node.value

    def _call_action_result(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(self.GetValue(), self._cur_attr)(*args, **kwargs)

    def create_variations(self) -> ResultMapProxy:
        """``VariationsFor``: refused, since xrdroot's frame has no ``Vary`` to vary it."""
        from ...errors import UnsupportedFeatureError

        raise UnsupportedFeatureError(
            "VariationsFor gives the results of the variations Vary booked, and xrdroot's "
            "RDataFrame has no Vary yet; book each variation as a result of its own instead"
        )


class NodeProxy(Proxy):
    """``RNode``: a frame of the graph; a supported operation records a node and gives back
    its proxy, and any other name is the node's own attribute."""

    def __getattr__(self, attr: str) -> Any:
        if attr in Operation.SUPPORTED_OPERATIONS:
            self.proxied_node._new_op_name = attr
            return self._create_new_op
        try:
            return getattr(self.proxied_node, attr)
        except AttributeError:
            operation = self.proxied_node.operation
            kind = operation.name if operation is not None else "RDataFrame"
            raise AttributeError(f"'{kind}' object has no attribute '{attr}'") from None

    def GetColumnNames(self) -> Any:
        return self.proxied_node.rdf_node.GetColumnNames()

    def GetColumnType(self, column: str) -> Any:
        return self.proxied_node.rdf_node.GetColumnType(column)

    def _create_new_op(self, *args: Any, **kwargs: Any) -> Any:
        op = Operation.create_op(self.proxied_node._new_op_name, *args, **kwargs)
        return get_proxy_for(op, _create_new_node(self.proxied_node, op))


@singledispatch
def get_proxy_for(operation: Operation.Operation, node: Node) -> Any:
    """What the user gets for a new node: a frame's proxy for a transformation."""
    _update_internal_df_with_transformation(node, operation)
    return NodeProxy(node)


@get_proxy_for.register
def _(operation: Operation.Action, node: Node) -> Any:
    return ResultPtrProxy(node)


@get_proxy_for.register
def _(operation: Operation.InstantAction, node: Node) -> Any:
    execute_graph(node)
    return node.value


@get_proxy_for.register
def _(operation: Operation.Snapshot, node: Node) -> Any:
    if len(operation.args) == 4:
        options = operation.args[3]
        if str(options.fMode).lower() == "update":
            raise ValueError(
                "Opening a file in UPDATE mode is not supported in distributed Snapshot."
            )
        if options.fLazy:
            return ResultPtrProxy(node)
    return get_proxy_for.dispatch(Operation.InstantAction)(operation, node)


@get_proxy_for.register
def _(operation: Operation.AsNumpy, node: Node) -> Any:
    if operation.kwargs.get("lazy", False):
        return ResultPtrProxy(node)
    return get_proxy_for.dispatch(Operation.InstantAction)(operation, node)
