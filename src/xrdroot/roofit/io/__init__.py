"""Reading a ``RooWorkspace`` from a ROOT file, into the engine's own objects.

A workspace is written as one object graph: every node of every model,
each pointing at the nodes it is made of and at those made of it, the
datasets, the named sets, the snapshots and whatever else was imported -
a ``ModelConfig``. :mod:`.stream` walks those bytes by the file's streamer
information, with RooFit's own streamers - the workspace's code
repository, ``RooRealVar``, ``RooLinkedList``, ``RooRefArray`` - written
out by hand, into plain :class:`~.stream.Streamed` records; :mod:`.build`
makes the engine's classes of them, and refuses by name a class it has not.

A workspace that carries C++ code for classes of its own - its code
repository - is read only when the repository is empty or holds classes
the engine has: compiling somebody's C++ is not something this does.
"""

from __future__ import annotations
