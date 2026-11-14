"""``RDataFrame`` of a distributed graph: the proxy of its head node, under the frame's name."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

from . import Proxy
from .Operation import SUPPORTED_OPERATIONS

if TYPE_CHECKING:
    from .HeadNode import HeadNode

__all__ = ["RDataFrame"]


class RDataFrame:
    """A distributed frame: every operation and attribute is its head node's proxy's."""

    def __init__(self, headnode: HeadNode) -> None:
        self._headnode = headnode
        self._headproxy = Proxy.NodeProxy(headnode)

    def __dir__(self) -> list[str]:
        return sorted(itertools.chain(SUPPORTED_OPERATIONS, super().__dir__()))

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__") or attr in ("_headnode", "_headproxy"):
            raise AttributeError(attr)
        return getattr(self._headproxy, attr)

    def __repr__(self) -> str:
        return f"<distributed RDataFrame on {type(self._headnode.backend).__name__}>"
