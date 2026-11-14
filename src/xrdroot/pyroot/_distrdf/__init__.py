"""Distributed ``RDataFrame``: one graph, run over ranges of the data on a Dask cluster.

    >>> from dask.distributed import Client                      # doctest: +SKIP
    >>> df = ROOT.RDataFrame("Events", files, executor=Client(), npartitions=8)  # doctest: +SKIP
    >>> h = df.Filter("nMuon == 2").Histo1D(("m", "m", 100, 0, 200), "m")        # doctest: +SKIP
    >>> h.GetValue()                                             # doctest: +SKIP

This is ROOT's ``ROOT._distrdf``, and ``ROOT.RDF.Distributed``: a frame made
with an ``executor`` records the graph built on it (:mod:`.Proxy`,
:mod:`.Node`) and, when a result is asked for, cuts the data into ranges
(:mod:`.Ranges`), runs the whole graph over each in a task of its own on the
cluster - each on a frame of xrdroot's over its range (:mod:`.HeadNode`,
:mod:`.ComputationGraphGenerator`) - and adds the tasks' results up, as
ROOT's mergeable results are added: histograms bin by bin, counts and sums,
means from their sums and counts, ``AsNumpy``'s columns end to end, and a
``Snapshot`` as a file per task. Only Dask is a backend here; Spark is
refused. Importing this imports neither, nor anything of ``cppyy``.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import logging
import textwrap
import types
import warnings
from collections.abc import Iterable
from typing import Any

from ...errors import UnsupportedFeatureError
from .Backends import build_backends_submodules

__all__ = [
    "initialize",
    "DistributeCppCode",
    "DistributeHeaders",
    "DistributeFiles",
    "DistributeSharedLibs",
    "RunGraphs",
    "VariationsFor",
    "FromSpec",
    "LiveVisualize",
    "create_distributed_module",
    "distributed_module",
    "RDataFrame",
]

logger = logging.getLogger(__name__)


def initialize(fun: Any, *args: Any, **kwargs: Any) -> None:
    """Run ``fun(*args, **kwargs)`` first in every task, and here, now."""
    from .Backends import Base

    Base.BaseBackend.register_initialization(fun, *args, **kwargs)


def DistributeCppCode(code_to_declare: str) -> None:
    """Declare C++ here and in every task."""
    from .Backends import Base

    Base.BaseBackend.register_declaration(code_to_declare)


def DistributeHeaders(paths_to_headers: str | Iterable[str]) -> None:
    """Declare headers here, and send them to every worker to be declared there."""
    from .Backends import Base

    Base.BaseBackend.register_headers(paths_to_headers)


def DistributeFiles(paths_to_files: str | Iterable[str]) -> None:
    """Send files to every worker's local directory."""
    from .Backends import Base

    Base.BaseBackend.register_files(paths_to_files)


def DistributeSharedLibs(paths_to_shared_libraries: str | Iterable[str]) -> None:
    """Load shared libraries - xrdroot's ``.xrdlib`` - here, and send them to every worker
    to be loaded there."""
    from .Backends import Base

    Base.BaseBackend.register_shared_lib(paths_to_shared_libraries)


def RunGraphs(proxies: Iterable[Any]) -> int:
    """Run the graphs these results belong to, side by side; how many graphs ran."""
    from .Proxy import execute_graph

    unique = list({proxy.proxied_node.get_head(): proxy for proxy in proxies}.values())
    if not unique:
        logger.warning("RunGraphs: Got an empty list of handles, now quitting.")
        return 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(unique)) as pool:
        for done in [pool.submit(execute_graph, proxy.proxied_node) for proxy in unique]:
            done.result()
    return len(unique)


def VariationsFor(actionproxy: Any) -> Any:
    """``VariationsFor`` of a distributed result."""
    return actionproxy.create_variations()


def FromSpec(jsonfile: str, *args: Any, **kwargs: Any) -> Any:
    """``FromSpec`` with an executor: refused, as xrdroot reads no dataset specification
    from JSON yet."""
    raise UnsupportedFeatureError(
        "RDF.Experimental.FromSpec reads a dataset's specification from JSON, which xrdroot "
        "does not yet; build an RDF.Experimental.RDatasetSpec and pass it to RDataFrame"
    )


def LiveVisualize(drawables: Any, callback: Any = None) -> None:
    """``LiveVisualize``: refused, as xrdroot draws a distributed result once it is complete."""
    raise UnsupportedFeatureError(
        "LiveVisualize draws partial results while a distributed graph runs, which xrdroot "
        "does not; draw the result once GetValue gives it"
    )


class _DeprecatedModule(types.ModuleType):
    """``ROOT.RDF.Experimental.Distributed``: warns, as ROOT 6.36 on does, on every use."""

    def __getattribute__(self, name: str) -> Any:
        warnings.warn(
            textwrap.dedent(
                """
                In ROOT 6.36, the ROOT.RDF.Experimental.Distributed module has become just
                ROOT.RDF.Distributed. You can now give the distributed Dask client to
                ROOT.RDataFrame directly: ROOT.RDataFrame(..., executor = connection).
                """
            ),
            FutureWarning,
            stacklevel=2,
        )
        return super().__getattribute__(name)


def create_distributed_module(parentmodule: Any, experimental: bool = False) -> types.ModuleType:
    """``ROOT.RDF.Distributed``: a module of the backends and the functions above."""
    distributed = types.ModuleType("ROOT.RDF.Distributed")
    distributed.__file__ = "<module ROOT.RDF>"
    distributed.__path__ = []
    distributed.__package__ = str(parentmodule)
    build_backends_submodules(distributed)
    for name in FUNCTIONS:
        setattr(distributed, name, globals()[name])
    if experimental:
        distributed.__class__ = _DeprecatedModule
    return distributed


#: The functions ``ROOT.RDF.Distributed`` has, beside its backends.
FUNCTIONS = (
    "initialize", "RunGraphs", "VariationsFor", "LiveVisualize", "DistributeHeaders",
    "DistributeFiles", "DistributeSharedLibs", "DistributeCppCode", "FromSpec",
)  # fmt: skip

#: ``ROOT.RDF.Distributed`` and ``ROOT.RDF.Experimental.Distributed``, made when asked for.
_MODULES: dict[bool, types.ModuleType] = {}


def distributed_module(experimental: bool = False) -> types.ModuleType:
    """``ROOT.RDF.Distributed`` - or, ``experimental``, its old name's module, which warns."""
    if experimental not in _MODULES:
        _MODULES[experimental] = create_distributed_module("ROOT.RDF", experimental)
    return _MODULES[experimental]


def RDataFrame(*args: Any, **kwargs: Any) -> Any:
    """``ROOT.RDataFrame(..., executor=client)``: a distributed frame on the client's cluster."""
    executor = kwargs.get("executor")
    if executor is None:
        raise ValueError(
            "Missing keyword argument 'executor'. Please provide a connection object to one "
            "of the schedulers supported by distributed RDataFrame."
        )
    if importlib.util.find_spec("distributed") is not None:
        from distributed import Client

        from .Backends.Dask import RDataFrame as dask_frame

        if isinstance(executor, Client):
            return dask_frame(*args, **kwargs)
    if type(executor).__name__ == "SparkContext":
        from .Backends.Spark import RDataFrame as spark_frame

        return spark_frame(*args, **kwargs)
    raise TypeError(
        f"The client object of type '{type(executor)}' is not a supported connection type "
        "for distributed RDataFrame."
    )
