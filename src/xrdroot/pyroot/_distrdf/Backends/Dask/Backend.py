"""The Dask backend: each range a ``dask.delayed`` task, reduced pairwise on the cluster.

Dask is optional - ``pip install "xrdroot[distrdf]"`` - and imported only
here, when a distributed frame is made. Headers, shared libraries and files
reach the workers with ``Client.upload_file``, into each worker's local
directory, where every task declares or loads them before it starts.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from .....errors import UnsupportedFeatureError
from ... import DataFrame, HeadNode
from .. import Base, Utils

if TYPE_CHECKING:
    from ..._graph_cache import ExecutionIdentifier

__all__ = ["DaskBackend", "dask_modules", "get_total_cores"]


def dask_modules() -> tuple[Any, Any]:
    """``dask`` and ``dask.distributed``, or the ``pip install`` that brings them."""
    try:
        import dask
        import dask.distributed as distributed
    except ImportError:
        raise UnsupportedFeatureError(
            "distributed RDataFrame runs on Dask, which is not installed: "
            'pip install "xrdroot[distrdf]" - or pip install "dask[distributed]"'
        ) from None
    return dask, distributed


def get_total_cores(client: Any) -> int:
    """The cores the Dask scheduler knows of: every worker's threads."""
    return int(sum(client.ncores().values()))


def _root_for_scripts() -> None:
    """``import ROOT`` in a worker - in a user's initialization, a callable of theirs - is
    this namespace, as it is where the frame was made."""
    if "ROOT" not in sys.modules:
        from ..... import pyroot

        sys.modules["ROOT"] = pyroot


class DaskBackend(Base.BaseBackend):
    """Distributed ``RDataFrame`` on a Dask cluster, through ``daskclient`` - or, given none,
    a cluster of a worker process per core of this machine."""

    def __init__(self, daskclient: Any = None) -> None:
        super().__init__()
        _, distributed = dask_modules()
        self.client = daskclient
        if daskclient is None:
            cluster = distributed.LocalCluster(n_workers=os.cpu_count(), threads_per_worker=1,
                                               processes=True)  # fmt: skip
            self.client = distributed.Client(cluster)

    def optimize_npartitions(self) -> int:
        """A partition per core of the cluster."""
        return get_total_cores(self.client)

    @staticmethod
    def dask_mapper(
        current_range: Any,
        headers: Iterable[str],
        shared_libraries: Iterable[str],
        pcms: Iterable[str],
        files: Iterable[str],
        mapper: Callable[[Any], Base.TaskResult],
    ) -> Base.TaskResult:
        """A task on a worker: the headers and libraries sent there declared and loaded from
        its local directory, then the range mapped."""
        _root_for_scripts()
        _, distributed = dask_modules()
        localdir = distributed.get_worker().local_directory
        Utils.distribute_headers(os.path.join(localdir, os.path.basename(path)) for path in headers)
        Utils.distribute_shared_libraries(
            os.path.join(localdir, os.path.basename(path)) for path in shared_libraries
        )
        return mapper(current_range)

    def ProcessAndMerge(
        self,
        ranges: list[Any],
        mapper: Callable[[Any], Base.TaskResult],
        reducer: Callable[[Base.TaskResult, Base.TaskResult], Base.TaskResult],
    ) -> Base.TaskResult:
        """A task per range, their results reduced two at a time, all computed on the cluster."""
        dask, _ = dask_modules()
        for paths in (self.headers, self.shared_libraries, self.pcms, self.files):
            self.distribute_unique_paths(paths)
        dmapper = dask.delayed(DaskBackend.dask_mapper)
        dreducer = dask.delayed(reducer)
        sent = [sorted(paths) for paths in (self.headers, self.shared_libraries, self.pcms,
                                            self.files)]  # fmt: skip
        pending = [dmapper(each, *sent, mapper) for each in ranges]
        while len(pending) > 1:
            pending.append(dreducer(pending.pop(0), pending.pop(0)))
        return self.client.compute(pending.pop()).result()  # type: ignore[no-any-return]

    def distribute_unique_paths(self, paths: Iterable[str]) -> None:
        """Each file uploaded to every worker's local directory."""
        for path in paths:
            self.client.upload_file(path, load=False)

    def make_dataframe(self, *args: Any, **kwargs: Any) -> Any:
        """A distributed frame on this cluster, cut into ``npartitions`` if given."""
        npartitions = kwargs.pop("npartitions", None)
        return DataFrame.RDataFrame(HeadNode.get_headnode(self, npartitions, *args))

    def cleanup_cache(self, exec_id: ExecutionIdentifier) -> None:
        """Every worker lets go of the run's frame and results."""
        self.client.run(_forget, exec_id)


def _forget(exec_id: ExecutionIdentifier) -> None:
    from ..._graph_cache import _ACTIONS_REGISTER, _RDF_REGISTER

    _ACTIONS_REGISTER.pop(exec_id, None)
    _RDF_REGISTER.pop(exec_id, None)
