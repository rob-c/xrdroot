"""``graphVizTree``: a model's graph of nodes and the servers each is made of, as GraphViz DOT.

``RooAbsArg::graphVizTree("model.dot")`` writes ``digraph "model"{``, then
each node once - in the order ``treeNodeServerList`` walks them, the node
before what it is made of - as ``"name" [ color=..., label="..."];``, red
for a function, blue for a variable, the label its class, its name (or
title) and a real-valued node's value, each on a line of its own; then an
edge ``"client" -> "server";`` for every node and each of its servers, and
``}``. ROOT keeps the edges in a set ordered by the objects' addresses, an
order no model's meaning fixes; here they go in the nodes' order.
"""

from __future__ import annotations

from typing import Any

from .printing import g

__all__ = ["graphviz"]


def _nodes(top: Any) -> list[Any]:
    """``treeNodeServerList``: the node, then depth-first each server's, every node once."""
    found: list[Any] = []

    def walk(node: Any) -> None:
        if any(node is seen for seen in found):
            return
        found.append(node)
        for server in node.servers():
            walk(server)

    walk(top)
    return found


def _label(node: Any, delimiter: str, use_title: bool) -> str:
    title = str(node.GetTitle())
    text = title if use_title and title else str(node.GetName())
    if hasattr(node, "getVal") and not getattr(node, "isCategory", lambda: False)():
        text += delimiter + g(float(node.getVal()))
    return f"{node.ClassName()}{delimiter}{text}"


def graphviz(top: Any, delimiter: str = "\n", use_title: bool = False) -> str:
    """The DOT text of ``top``'s graph."""
    nodes = _nodes(top)
    lines = [f'digraph "{top.GetName()}"{{']
    for node in nodes:
        colour = "blue" if node.isFundamental() else "red"
        lines.append(
            f'"{node.GetName()}" [ color={colour}, label="{_label(node, delimiter, use_title)}"];'
        )
    for node in nodes:
        lines += [f'"{node.GetName()}" -> "{server.GetName()}";' for server in node.servers()]
    return "\n".join(lines) + "\n}\n"
