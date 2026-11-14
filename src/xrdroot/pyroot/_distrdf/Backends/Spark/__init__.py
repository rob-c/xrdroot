"""``ROOT.RDF.Distributed.Spark``: refused - xrdroot runs distributed ``RDataFrame`` on Dask."""

from __future__ import annotations

from typing import Any

from .....errors import UnsupportedFeatureError

__all__ = ["RDataFrame"]


def RDataFrame(*args: Any, **kwargs: Any) -> Any:
    """Refused, saying what to use instead."""
    raise UnsupportedFeatureError(
        "Distributed RDataFrame on Spark is not supported by xrdroot; run it on Dask "
        "instead, giving ROOT.RDataFrame a dask.distributed Client as its executor"
    )
