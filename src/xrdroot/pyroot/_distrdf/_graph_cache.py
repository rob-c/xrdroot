"""The frames and results a process keeps from one task to the next of the same run.

Tasks of one run of one graph that land in the same process share its
frame: the first builds the graph on a frame of its range, the next point
that frame at their own ranges and book the results again, so the graph is
built once per process rather than once per task - ROOT's way of not
compiling it again. The keys are :class:`ExecutionIdentifier`\\ s, and the
backend clears them when the run is over.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any

__all__ = ["ExecutionIdentifier", "_RDF_REGISTER", "_ACTIONS_REGISTER", "LOCK"]


@dataclass(frozen=True)
class ExecutionIdentifier:
    """One run of the graph of one distributed frame: the frame's ``rdf_uuid`` and the run's
    ``graph_uuid``."""

    rdf_uuid: uuid.UUID | str
    graph_uuid: uuid.UUID


#: Each run's frame in this process.
_RDF_REGISTER: dict[ExecutionIdentifier, Any] = {}
#: Each run's booked results in this process, in the order of the graph's actions.
_ACTIONS_REGISTER: dict[ExecutionIdentifier, list[Any]] = {}
#: Held by a task while it uses the registers: tasks run side by side in one process's
#: threads take turns, since they would share a frame.
LOCK = threading.Lock()
