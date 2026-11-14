"""A node of distributed ``RDataFrame``'s computation graph: one operation, and its place.

The graph a user builds is recorded here, not run: each ``Define``,
``Filter`` or action is a :class:`Node` holding its :class:`~.Operation`,
its parent and a count of its children. When a result is asked for, the
nodes that still matter - see :meth:`Node.is_prunable` - are sent to every
task, which calls each operation on the frame of its own range of entries.
A node travels as its operation and its parent's id alone.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .HeadNode import HeadNode
    from .Operation import Operation

__all__ = ["Node"]


class Node:
    """A node of the graph.

    ``get_head`` gives the graph's head node and ``node_id`` numbers this
    node in the order nodes were made. ``value`` is an action's result once
    the graph has run; ``has_user_references`` turns false when the proxy a
    user held is collected; ``rdf_node`` is the frame - in this process, the
    head node's frame of the whole data - the operation was called on, so
    that column names and types are known before anything runs.
    """

    def __init__(
        self,
        get_head: Callable[[], HeadNode],
        node_id: int = 0,
        operation: Operation | None = None,
        parent: Node | None = None,
    ) -> None:
        self.get_head = get_head
        self.node_id = node_id
        self.operation = operation
        self.parent = parent
        self.nchildren = 0
        self._new_op_name = ""
        self.value: Any = None
        self.has_user_references = True
        self.rdf_node: Any = None
        self._parent_id = parent.node_id if parent is not None else None

    @property
    def parent_id(self) -> int | None:
        return self._parent_id

    @parent_id.setter
    def parent_id(self, value: int | None) -> None:
        self._parent_id = value

    def __getstate__(self) -> dict[str, Any]:
        """What a task needs of this node: its operation, and which node it is called on."""
        return {"operation": self.operation, "parent_id": self.parent_id}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.operation = state["operation"]
        self.parent_id = state["parent_id"]

    def is_prunable(self) -> bool:
        """Can this node be left out of the graph sent to the tasks?

        A node with no children can, when the user holds it no longer or its
        value is already known; its parent then has one child fewer, so that
        going over the nodes from the newest prunes whole branches.
        """
        if self.nchildren == 0 and (not self.has_user_references or self.value is not None):
            if self.parent is not None:
                self.parent.nchildren -= 1
            return True
        return False
